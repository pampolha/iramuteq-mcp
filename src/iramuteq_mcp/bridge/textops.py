"""Corpus operations and analysis runners over the IRaMuTeQ sources.

Every function here calls the IRaMuTeQ entry points directly
(``BuildFromAlceste``, ``Reinert``, ``Stat``, ...) — no analysis logic is
reimplemented.  Corpus objects are expensive to build (Lexique 3 + parse),
so they are cached by (path, mtime, language, encoding, options).
"""

from __future__ import annotations

import configparser
import contextlib
import hashlib
import json
import logging
import os
import tempfile
import threading
from types import SimpleNamespace

from . import bootstrap, patches

log = logging.getLogger("iramuteq_mcp")

_corpus_cache: dict[str, tuple] = {}
_cache_lock = threading.Lock()


# --------------------------------------------------------------------- corpus


def _corpus_key(path: str, parametres: dict, output_dir: str | None) -> str:
    """Key on the *resolved* parameter dict (corpus.cfg defaults + overrides).

    ``pathout``/``corpus_name`` are excluded: they only locate the output and
    ``PathOut.mkdirout`` picks a fresh number as soon as a corpus directory
    exists, which would defeat reuse.  ``output_dir`` is kept separately for
    the same reason (it is an explicit caller choice).
    """
    st = os.stat(path)
    resolved = {k: v for k, v in parametres.items() if k not in ("pathout", "corpus_name")}
    blob = json.dumps(
        {"path": path, "mtime": st.st_mtime_ns, "output_dir": output_dir, "parametres": resolved},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(blob.encode()).hexdigest()


def build_corpus(
    corpus_path: str,
    lang: str = "french",
    encoding: str = "utf8",
    output_dir: str | None = None,
    options: dict | None = None,
):
    """Build (or fetch cached) IRaMuTeQ Corpus for a .txt corpus file.

    Returns ``(corpus, summary)``.  Mirrors ``autres/iracmd.py``.
    """
    from chemins import PathOut
    from corpus import BuildFromAlceste, copycorpus
    from functions import DoConf

    patches.sweep()  # corpus module now imported -> install the sqlite shim
    ira = bootstrap.get()
    ira.load_lexicon(lang)
    path = os.path.abspath(corpus_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"corpus not found: {path}")
    options = dict(options or {})

    # corpus.cfg defaults -> overrides (what CorpusPref would collect in the GUI)
    parametres = DoConf(ira.ConfigPath["corpus"]).getoptions("corpus")
    for k, v in options.items():
        if v is not None:
            parametres[k] = v
    parametres["filename"] = os.path.basename(path)
    parametres["originalpath"] = path
    parametres["encoding"] = encoding
    parametres["syscoding"] = "utf8"
    parametres["lang"] = lang
    if output_dir:
        parametres["pathout"] = output_dir
    else:
        parametres["pathout"] = PathOut(path, "corpus").mkdirout()
    parametres["corpus_name"] = os.path.basename(parametres["pathout"])

    key = _corpus_key(path, parametres, output_dir)
    with _cache_lock:
        cached = _corpus_cache.get(key)
        if cached is not None:
            return cached

    builder = BuildFromAlceste(path, parametres, ira.lexique, ira.expressions)
    corpus = builder.corpus
    ira.history.add(corpus.parametres)
    corpus = copycorpus(corpus)
    corpus.conn_all()

    summary = corpus_summary(corpus)
    with _cache_lock:
        _corpus_cache[key] = (corpus, summary)
    return corpus, summary


def corpus_summary(corpus) -> dict:
    p = corpus.parametres
    return {
        "corpus_uuid": p.get("uuid"),
        "corpus_name": p.get("corpus_name"),
        "pathout": p.get("pathout"),
        "ira": p.get("ira"),
        "documents_ucis": p.get("ucinb"),
        "segments_uces": p.get("ucenb"),
        "occurrences": p.get("occurrences"),
        "forms": p.get("formesnb"),
        "hapax": p.get("hapax"),
        "build_time": p.get("time"),
    }


def get_corpus(corpus_path: str, lang: str = "french", encoding: str = "utf8", **kwargs):
    return build_corpus(corpus_path, lang=lang, encoding=encoding, **kwargs)


def _prep_lemmas(corpus, lem: bool = True):
    """make_lems + parse_active with key.cfg — what AnalyseText does per analysis."""
    from functions import DoConf

    ira = bootstrap.get()
    keys = DoConf(ira.ConfigPath["key"]).getoptions()
    gramact = [k for k in keys if keys[k] == 1]
    gramsup = [k for k in keys if keys[k] == 2]
    corpus.make_lems(lem=lem)
    corpus.parse_active(gramact, gramsup)
    return keys


# ------------------------------------------------------------ input bounds

MAX_ACTIVES_RANGE = (50, 50000)


def validate_max_actives(value: int) -> int:
    """Check the number of active forms an ALCESTE run may analyse (memory bound)."""
    lo, hi = MAX_ACTIVES_RANGE
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"max_actives must be an integer between {lo} and {hi}, got {value!r}")
    return value


# ------------------------------------------------------------ key overrides

KEY_VALUES = {"0": "eliminated", "1": "active", "2": "supplementary"}


@contextlib.contextmanager
def key_overrides(active_keys=None, supplementary_keys=None):
    """Temporarily override grammatical-category keys (key.cfg) for one run.

    AnalyseText reads ``ira.ConfigPath['key']`` at construction time, so we
    point it at a temporary copy of key.cfg with the requested categories
    flipped to 1 (active) or 2 (supplementary).  Every analysis runs under
    ``jobs.analysis_lock()`` — job runners and synchronous tools alike — so
    no other run can read key.cfg while the swap is active.
    """
    if not active_keys and not supplementary_keys:
        yield
        return
    ira = bootstrap.get()
    conf = configparser.RawConfigParser()
    conf.read(ira.ConfigPath["key"], encoding="utf8")
    section = conf.sections()[0]
    known = set(conf.options(section))
    wanted = {k: "1" for k in (active_keys or [])}
    wanted.update({k: "2" for k in (supplementary_keys or [])})
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise ValueError(f"unknown analysis keys: {unknown}; available: {sorted(known)}")
    for k, v in wanted.items():
        conf.set(section, k, v)
    fd, tmp = tempfile.mkstemp(prefix="key-", suffix=".cfg", dir=ira.TEMPDIR)
    with os.fdopen(fd, "w", encoding="utf8") as f:
        conf.write(f)
    original = ira.ConfigPath["key"]
    ira.ConfigPath["key"] = tmp
    try:
        yield
    finally:
        ira.ConfigPath["key"] = original
        try:
            os.remove(tmp)
        except OSError:
            pass


def key_summary() -> dict:
    """Current key.cfg values, as {category: 'active'|'supplementary'|'eliminated'}."""
    from functions import DoConf

    keys = DoConf(bootstrap.get().ConfigPath["key"]).getoptions()
    keys.pop("type", None)
    return {k: KEY_VALUES.get(str(v), str(v)) for k, v in keys.items()}


# ------------------------------------------------------- modality resolution


def resolve_listet(corpus, modalities=None, single_variable=True):
    """Map user-given modalities to the ``*variable_modality`` forms IRaMuTeQ uses.

    Without ``modalities`` the modalities of the first variable are used.  When
    ``single_variable`` is set (Lexico, Labbé) all modalities must belong to one
    variable, because those analyses compare UCI groups defined by it.
    """
    from functions import treat_var_mod

    all_mods = list(corpus.make_etoiles())
    if modalities:
        wanted = {m.lstrip("*") for m in modalities}
        listet = [m for m in all_mods if m.lstrip("*") in wanted]
        missing = wanted - {m.lstrip("*") for m in listet}
        if missing:
            raise ValueError(
                f"unknown modalities: {sorted(missing)}; available: {sorted(m.lstrip('*') for m in all_mods)}"
            )
    else:
        by_var = treat_var_mod([m.lstrip("*") for m in all_mods])
        if not by_var:
            raise ValueError("corpus has no *variable_modality entries")
        first_var = sorted(by_var)[0]
        listet = [m for m in all_mods if m.lstrip("*") in set(by_var[first_var])]
    if not listet:
        raise ValueError("no modalities selected")
    if single_variable:
        used = treat_var_mod([m.lstrip("*") for m in listet])
        if len(used) > 1:
            raise ValueError(
                "modalities must belong to a single variable (got %s); the analysis "
                "groups UCIs by one variable" % sorted(used)
            )
    return sorted(listet)


# ------------------------------------------------------------------- validate

_CHECK_MESSAGES = {
    1: "starred variable contains a space",
    2: "theme (-*) contains a space",
    3: "star (*) inside a text line",
    4: "bad token in UCI header (must be *variable, spaces not allowed inside)",
}


def validate_corpus(corpus_path: str, encoding: str = "utf8", max_errors: int = 100) -> dict:
    """Validate ALCESTE formatting with IRaMuTeQ's own checkline()."""
    from textcheckcorpus import checkline

    path = os.path.abspath(corpus_path)
    errors = []
    with open(path, encoding=encoding, errors="replace") as f:
        for lnb, line in enumerate(f, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            res = checkline(line, lnb)
            if not res[0]:
                errors.append(
                    {
                        "line": lnb,
                        "code": res[1],
                        "message": _CHECK_MESSAGES.get(res[1], "invalid"),
                        "text": line[:120],
                    }
                )
                if len(errors) >= max_errors:
                    break
    return {"path": path, "ok": not errors, "errors": errors, "errors_reported": len(errors)}


# ----------------------------------------------------------------- concordance


def concordance(corpus, lem: str, limit: int = 50, width: int = 60) -> dict:
    """KWIC concordance for a lemma via corpus.getlemconcorde().

    The ``uces`` table holds the surface text, so a lemma is searched through
    the written forms of ``corpus.lems[lem].formes``, not the lemma literal:
    "ecrire" never matches a UCE containing "écrit".  Each hit reports the
    form that matched.
    """
    if not corpus.lems:
        _prep_lemmas(corpus)
    if lem not in corpus.lems:
        return {"lem": lem, "found": False, "hits": [], "hint": "lemma not in corpus (try corpus_stats)"}
    if corpus.idformes is None:
        corpus.make_idformes()
    idformes = corpus.idformes or {}
    forms = sorted({idformes[i].forme for i in corpus.lems[lem].formes if i in idformes})
    targets: list[str] = []
    for candidate in [lem, *forms]:
        low = candidate.lower()
        if low not in targets:
            targets.append(low)

    hits: list[dict[str, object]] = []
    rows = list(corpus.getlemconcorde(lem))
    for uce_id, text in rows:
        uci = corpus.getucefromid(uce_id)
        low = text.lower()
        matches: list[tuple[int, int, str]] = []
        for target in targets:
            start = 0
            while True:
                i = low.find(target, start)
                if i < 0:
                    break
                j = i + len(target)
                start = j
                if (i == 0 or not low[i - 1].isalnum()) and (j == len(low) or not low[j].isalnum()):
                    matches.append((i, j, target))
        for i, j, target in sorted(matches):
            if len(hits) >= limit:
                break
            if any(h["uce"] == uce_id and h["start"] == i for h in hits):
                continue
            hits.append(
                {
                    "uce": uce_id,
                    "uci": uci.uci if uci else None,
                    "form": target,
                    "left": text[max(0, i - width) : i],
                    "match": text[i:j],
                    "right": text[j : j + width],
                    "start": i,
                }
            )
        if len(hits) >= limit:
            break
    out: dict[str, object] = {
        "lem": lem,
        "found": True,
        "forms": forms,
        "uce_count": len(rows),
        "hits": hits,
    }
    if rows and not hits:
        out["note"] = (
            "the lemma occurs in the corpus but none of its forms appear verbatim in the "
            "UCE text (cleaning or segmentation removed them)"
        )
    return out


# -------------------------------------------------------------------- analyses


def run_stats(
    corpus_path: str,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = True,
    active_keys: list[str] | None = None,
    supplementary_keys: list[str] | None = None,
    ucemethod: int | None = None,
    ucesize: int | None = None,
) -> dict:
    """Textual statistics (IRaMuTeQ Stat) — synchronous, fast."""
    from textstat import Stat

    corpus, csum = get_corpus(
        corpus_path, lang=lang, encoding=encoding, options={"ucemethod": ucemethod, "ucesize": ucesize}
    )
    patches.sweep()
    with key_overrides(active_keys, supplementary_keys):
        obj = Stat(bootstrap.get(), corpus, {"type": "stat", "lem": lem}, dlg=False, lemdial=False)
    from . import results

    return results.parse_stats(obj, csum)


def run_alceste(
    corpus_path: str,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = True,
    output_dir: str | None = None,
    options: dict | None = None,
    active_keys: list[str] | None = None,
    supplementary_keys: list[str] | None = None,
    ucemethod: int | None = None,
    ucesize: int | None = None,
) -> dict:
    """Reinert/ALCESTE descending hierarchical classification (job payload)."""
    from functions import DoConf
    from textreinert import Reinert

    ira = bootstrap.get()
    corpus, csum = get_corpus(
        corpus_path, lang=lang, encoding=encoding, options={"ucemethod": ucemethod, "ucesize": ucesize}
    )
    patches.sweep()

    # defaults = reinert.cfg [ALCESTE], overridden by tool options (the dialog
    # would have collected exactly these keys)
    parametres = DoConf(ira.ConfigPath["reinert"]).getoptions("ALCESTE")
    parametres.pop("type", None)
    parametres.update({k: v for k, v in (options or {}).items() if v is not None})
    parametres["max_actives"] = validate_max_actives(parametres.get("max_actives", 20000))
    parametres["type"] = "alceste"
    parametres["lem"] = lem
    if output_dir:
        parametres["pathout"] = output_dir

    with key_overrides(active_keys, supplementary_keys):
        obj = Reinert(ira, corpus, parametres=parametres, dlg=False, lemdial=False)
    from . import results

    return results.parse_alceste(obj, csum)


def run_lexico(
    corpus_path: str,
    modalities: list[str] | None = None,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = True,
    output_dir: str | None = None,
    options: dict | None = None,
    active_keys: list[str] | None = None,
    supplementary_keys: list[str] | None = None,
    ucemethod: int | None = None,
    ucesize: int | None = None,
) -> dict:
    """Specificities / "Comme Lexico" (IRaMuTeQ Lexico) — job payload.

    ``modalities`` must all belong to one ``*variable`` (see resolve_listet).
    """
    from textaslexico import Lexico

    corpus, csum = get_corpus(
        corpus_path, lang=lang, encoding=encoding, options={"ucemethod": ucemethod, "ucesize": ucesize}
    )
    patches.sweep()
    listet = resolve_listet(corpus, modalities, single_variable=True)

    parametres = {"type": "spec", "lem": lem}
    parametres.update({k: v for k, v in (options or {}).items() if v is not None})
    if output_dir:
        parametres["pathout"] = output_dir

    class HeadlessLexico(Lexico):
        def make_config(self, config):
            # the GUI dialog (OptLexi) would set these; inject from tool args
            self.listet = listet
            config.setdefault("mineff", 10)
            config.setdefault("indice", "hypergeo")
            config.setdefault("typeformes", 0)
            config.setdefault("clnb", 0)
            return config

    with key_overrides(active_keys, supplementary_keys):
        obj = HeadlessLexico(bootstrap.get(), corpus, parametres=parametres, dlg=False, lemdial=False)
    from . import results

    return results.parse_lexico(obj, csum)


def run_similitude(
    corpus_path: str,
    forms: list[str] | None = None,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = True,
    output_dir: str | None = None,
    options: dict | None = None,
    active_keys: list[str] | None = None,
    supplementary_keys: list[str] | None = None,
    ucemethod: int | None = None,
    ucesize: int | None = None,
) -> dict:
    """Similarity graph between active forms (IRaMuTeQ SimiTxt) — job payload."""
    from textsimi import SimiTxt

    corpus, csum = get_corpus(
        corpus_path, lang=lang, encoding=encoding, options={"ucemethod": ucemethod, "ucesize": ucesize}
    )
    patches.sweep()
    # makesimiparam is a SimiTxt method (defaults the GUI would show); calling it
    # unbound on a holder keeps the defaults in sync with the source tree.
    holder = SimpleNamespace(parametres={})
    SimiTxt.makesimiparam(holder)
    parametres = {"type": "simitxt", "lem": lem}
    parametres.update(holder.parametres)
    parametres.update({k: v for k, v in (options or {}).items() if v is not None})
    if output_dir:
        parametres["pathout"] = output_dir

    with key_overrides(active_keys, supplementary_keys):
        # Headless SimiTxt never collects 'selected' itself: its doanalyse writes
        # selected.csv as ``order_actives[val][0] for val in parametres['selected']``,
        # i.e. positions in the frequency-sorted list of ``make_actives_limit(3)``.
        _prep_lemmas(corpus, lem=lem)
        actives = corpus.make_actives_limit(3)
        eff = [corpus.getlemeff(a) for a in actives]
        order = sorted(range(len(actives)), key=lambda i: -eff[i])
        if not forms:
            parametres["selected"] = list(range(len(actives)))
        else:
            missing = [f for f in forms if f not in set(actives)]
            if missing:
                raise ValueError(f"forms not among the active forms: {missing}; active forms: {sorted(actives)[:50]}")
            rank = {i: p for p, i in enumerate(order)}
            wanted = set(forms)
            parametres["selected"] = sorted(rank[i] for i, a in enumerate(actives) if a in wanted)
        obj = SimiTxt(bootstrap.get(), corpus, parametres=parametres, dlg=False, lemdial=False)
    from . import results

    return results.parse_similitude(obj, csum)
