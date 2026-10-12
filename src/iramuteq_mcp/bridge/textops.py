"""Corpus operations and analysis runners over the IRaMuTeQ sources.

Every function here calls the IRaMuTeQ entry points directly
(``BuildFromAlceste``, ``Reinert``, ``Stat``, ...) — no analysis logic is
reimplemented.  Corpus objects are expensive to build (Lexique 3 + parse),
so they are cached by (path, mtime, language, encoding, options).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading

from . import bootstrap, patches

log = logging.getLogger("iramuteq_mcp")

_corpus_cache: dict[str, object] = {}
_cache_lock = threading.Lock()


# --------------------------------------------------------------------- corpus


def _corpus_key(path: str, lang: str, encoding: str, options: dict) -> str:
    st = os.stat(path)
    blob = json.dumps(
        {"path": path, "mtime": st.st_mtime_ns, "lang": lang, "enc": encoding, **options},
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

    key = _corpus_key(path, lang, encoding, options)
    with _cache_lock:
        cached = _corpus_cache.get(key)
        if cached is not None:
            return cached

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
    with open(path, "r", encoding=encoding, errors="replace") as f:
        for lnb, line in enumerate(f, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            res = checkline(line, lnb)
            if not res[0]:
                errors.append(
                    {"line": lnb, "code": res[1], "message": _CHECK_MESSAGES.get(res[1], "invalid"), "text": line[:120]}
                )
                if len(errors) >= max_errors:
                    break
    return {"path": path, "ok": not errors, "errors": errors, "total_errors": len(errors)}


# ----------------------------------------------------------------- concordance


def concordance(corpus, lem: str, limit: int = 50, width: int = 60) -> dict:
    """KWIC concordance for a lemma via corpus.getlemconcorde()."""
    if not corpus.lems:
        _prep_lemmas(corpus)
    if lem not in corpus.lems:
        return {"lem": lem, "found": False, "hits": [], "hint": "lemma not in corpus (try corpus_stats)"}
    hits = []
    rows = list(corpus.getlemconcorde(lem))
    for uce_id, text in rows:
        uci = corpus.getucefromid(uce_id)
        start = 0
        low = text.lower()
        target = lem.lower()
        while len(hits) < limit:
            i = low.find(target, start)
            if i < 0:
                break
            j = i + len(target)
            if (i == 0 or not low[i - 1].isalnum()) and (j == len(low) or not low[j].isalnum()):
                hits.append(
                    {
                        "uce": uce_id,
                        "uci": uci.uci if uci else None,
                        "left": text[max(0, i - width) : i],
                        "match": text[i:j],
                        "right": text[j : j + width],
                    }
                )
            start = j
        if len(hits) >= limit:
            break
    return {"lem": lem, "found": True, "uce_count": len(rows), "hits": hits}


# -------------------------------------------------------------------- analyses


def run_stats(corpus_path: str, lang: str = "french", encoding: str = "utf8", lem: bool = True) -> dict:
    """Textual statistics (IRaMuTeQ Stat) — synchronous, fast."""
    from textstat import Stat

    corpus, csum = get_corpus(corpus_path, lang=lang, encoding=encoding)
    patches.sweep()
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
) -> dict:
    """Reinert/ALCESTE descending hierarchical classification (job payload)."""
    from functions import DoConf
    from textreinert import Reinert

    ira = bootstrap.get()
    corpus, csum = get_corpus(corpus_path, lang=lang, encoding=encoding)
    patches.sweep()

    # defaults = reinert.cfg [ALCESTE], overridden by tool options (the dialog
    # would have collected exactly these keys)
    parametres = DoConf(ira.ConfigPath["reinert"]).getoptions("ALCESTE")
    parametres.pop("type", None)
    parametres.update({k: v for k, v in (options or {}).items() if v is not None})
    parametres["type"] = "alceste"
    parametres["lem"] = lem
    if output_dir:
        parametres["pathout"] = output_dir

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
) -> dict:
    """Specificities / "Comme Lexico" (IRaMuTeQ Lexico) — job payload.

    ``modalities`` are ``*variable_modality`` entries and must all belong to
    a single variable (the analysis compares UCI groups defined by it); when
    omitted, the modalities of the first variable found are used.
    """
    from functions import treat_var_mod
    from textaslexico import Lexico

    corpus, csum = get_corpus(corpus_path, lang=lang, encoding=encoding)
    patches.sweep()

    all_mods = list(corpus.make_etoiles())
    if modalities:
        wanted = set(modalities)
        listet = [m for m in all_mods if m.lstrip("*") in wanted]
        missing = wanted - {m.lstrip("*") for m in listet}
        if missing:
            raise ValueError(f"unknown modalities: {sorted(missing)}; available: {sorted(m.lstrip('*') for m in all_mods)}")
    else:
        by_var = treat_var_mod([m.lstrip("*") for m in all_mods])
        if not by_var:
            raise ValueError("corpus has no *variable_modality entries")
        first_var = sorted(by_var)[0]
        wanted_forms = set(by_var[first_var])
        listet = [m for m in all_mods if m.lstrip("*") in wanted_forms]
    if not listet:
        raise ValueError("no modalities selected")
    vars_used = treat_var_mod([m.lstrip("*") for m in listet])
    if len(vars_used) > 1:
        raise ValueError(
            "modalities must belong to a single variable (got %s); "
            "the Lexico comparison groups UCIs by one variable" % sorted(vars_used)
        )

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

    obj = HeadlessLexico(bootstrap.get(), corpus, parametres=parametres, dlg=False, lemdial=False)
    from . import results

    return results.parse_lexico(obj, csum)


def run_similitude(
    corpus_path: str,
    selected: list[int] | None = None,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = True,
    output_dir: str | None = None,
    options: dict | None = None,
) -> dict:
    """Similarity graph between active forms (IRaMuTeQ SimiTxt) — job payload."""
    from textsimi import SimiTxt, makesimiparam

    corpus, csum = get_corpus(corpus_path, lang=lang, encoding=encoding)
    patches.sweep()
    parametres = {"type": "simitxt", "lem": lem}
    parametres.update(makesimiparam())
    parametres.update({k: v for k, v in (options or {}).items() if v is not None})
    if selected is not None:
        parametres["selected"] = selected
    if output_dir:
        parametres["pathout"] = output_dir

    obj = SimiTxt(bootstrap.get(), corpus, parametres=parametres, dlg=False, lemdial=False)
    from . import results

    return results.parse_similitude(obj, csum)
