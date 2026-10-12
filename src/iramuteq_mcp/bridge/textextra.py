"""Remaining text analyses: Labbé distance, word clouds, Tgen profiles.

Plumbing (glue only, no analysis logic is duplicated):

* WordCloud/ClusterCloud open dialogs in ``make_option`` and call
  ``SelectColumn(dlg=True)`` inside ``doanalyse``.  The subclasses below fill
  the options from tool arguments and return ``wx.ID_OK``; the
  ``SelectColumn`` call is routed to its own ``dlg=False`` branch, which
  writes ``selected.csv`` (all words, or the ``forms`` requested).
* Labbé and Tgen need the same single-variable modality selection as Lexico.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from . import bootstrap, patches, textops

_tls = threading.local()
_selectcolumn_patched = False


# ------------------------------------------------------------------ plumbing


def _install_headless_selectcolumn() -> None:
    """Route textwordcloud.SelectColumn(dlg=True) to the dlg=False branch."""
    global _selectcolumn_patched
    if _selectcolumn_patched:
        return
    import textwordcloud
    from guifunct import SelectColumn as RealSelectColumn

    def headless_select(parent, dictcol, actives, pathout, selected=None, dlg=False):
        sel = selected
        forms = getattr(_tls, "forms", None)
        if sel is None and forms:
            missing = [f for f in forms if f not in actives]
            if missing:
                raise ValueError(f"forms not available for this selection: {missing}")
            sel = [actives.index(f) for f in forms]
        return RealSelectColumn(parent, dictcol, actives, pathout, selected=sel, dlg=False)

    textwordcloud.SelectColumn = headless_select
    _selectcolumn_patched = True


def new_outdir(base: str, kind: str) -> str:
    """Create ``<base>/<kind>_<N>`` with the first free N (IRaMuTeQ naming style)."""
    n = 1
    while os.path.exists(os.path.join(base, f"{kind}_{n}")):
        n += 1
    path = os.path.join(base, f"{kind}_{n}")
    os.makedirs(path)
    return path


def _nb_graph(outdir: str, svg: bool) -> str:
    end = ".svg" if svg else ".png"
    nb = 1
    while os.path.exists(os.path.join(outdir, f"nuage_{nb}{end}")):
        nb += 1
    return os.path.join(outdir, f"nuage_{nb}{end}")


def _rgb(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        value = [int(v) for v in value.replace("(", "").replace(")", "").split(",")]
    r, g, b = (int(v) for v in value)
    if not all(0 <= c <= 255 for c in (r, g, b)):
        raise ValueError("colours are RGB triples with components in 0..255")
    return (r, g, b)


# --------------------------------------------------------------------- labbé


def run_labbe(
    corpus_path: str,
    modalities: list[str] | None = None,
    mineff: int = 10,
    indice: str = "hypergeo",
    typeformes: int = 0,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = True,
    output_dir: str | None = None,
    active_keys: list[str] | None = None,
    supplementary_keys: list[str] | None = None,
) -> dict:
    """Labbé lexical distance between modalities (IRaMuTeQ DistLabbe) — job payload."""
    from textlabbe import DistLabbe

    if indice not in ("hypergeo", "chi2"):
        raise ValueError("indice must be 'hypergeo' or 'chi2'")
    corpus, csum = textops.get_corpus(corpus_path, lang=lang, encoding=encoding)
    patches.sweep()
    listet = textops.resolve_listet(corpus, modalities, single_variable=True)
    parametres = {
        "type": "labbe",
        "lem": lem,
        "mineff": int(mineff),
        "indice": indice,
        "typeformes": int(typeformes),
        "clnb": len(listet),
    }
    if output_dir:
        parametres["pathout"] = output_dir

    class HeadlessLabbe(DistLabbe):
        def make_config(self, config):
            self.listet = listet  # DistLabbe.preferences() would collect this
            return config

    with textops.key_overrides(active_keys, supplementary_keys):
        obj = HeadlessLabbe(bootstrap.get(), corpus, parametres=parametres, dlg=False, lemdial=False)
    from . import results

    return results.parse_labbe(obj, csum)


# ----------------------------------------------------------------- wordclouds

_CLOUD_DEFAULTS = {
    "width": 800,
    "height": 800,
    "maxword": 100,
    "mincex": 0.3,
    "maxcex": 3.0,
    "col_text": (0, 0, 0),
    "col_bg": (255, 255, 255),
    "svg": False,
}


def _cloud_options(opts: dict | None) -> dict:
    o: dict[str, Any] = dict(_CLOUD_DEFAULTS)
    o.update({k: v for k, v in (opts or {}).items() if v is not None})
    o["col_text"] = _rgb(o["col_text"], _CLOUD_DEFAULTS["col_text"])
    o["col_bg"] = _rgb(o["col_bg"], _CLOUD_DEFAULTS["col_bg"])
    o["svg"] = bool(o["svg"])
    o["mincex"] = float(o["mincex"])
    o["maxcex"] = float(o["maxcex"])
    return o


def _make_cloud_option_filler(opts: dict, mode: int | None = None, indice: int | None = None):
    """Build the make_option replacement shared by WordCloud and ClusterCloud."""

    def make_option(self, fromcluster=False):
        import wx

        p = self.parametres
        p.update(
            {
                "width": opts["width"],
                "height": opts["height"],
                "maxword": int(opts["maxword"]),
                "mincex": opts["mincex"],
                "maxcex": opts["maxcex"],
                "col_text": opts["col_text"],
                "col_bg": opts["col_bg"],
                "svg": 1 if opts["svg"] else 0,
            }
        )
        if mode is not None:
            p["mode"] = mode
        if fromcluster:
            p["indice"] = indice if indice is not None else 0
        p["graphout"] = _nb_graph(os.path.dirname(self.pathout["zipf.png"]), opts["svg"])
        return wx.ID_OK

    return make_option


def run_wordcloud(
    corpus_path: str,
    mode: int = 0,
    forms: list[str] | None = None,
    options: dict | None = None,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = True,
    output_dir: str | None = None,
    active_keys: list[str] | None = None,
    supplementary_keys: list[str] | None = None,
) -> dict:
    """Word cloud of active (0), supplementary (1) or all (2) forms."""
    from textwordcloud import WordCloud

    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0 (active), 1 (supplementary) or 2 (both)")
    _install_headless_selectcolumn()
    opts = _cloud_options(options)
    corpus, csum = textops.get_corpus(corpus_path, lang=lang, encoding=encoding)
    patches.sweep()
    parametres = {"type": "wordcloud", "lem": lem}
    if output_dir:
        parametres["pathout"] = output_dir

    class HeadlessWordCloud(WordCloud):
        make_option = _make_cloud_option_filler(opts, mode=mode)

    _tls.forms = forms
    try:
        with textops.key_overrides(active_keys, supplementary_keys):
            obj = HeadlessWordCloud(bootstrap.get(), corpus, parametres=parametres, dlg=False, lemdial=False)
    finally:
        _tls.forms = None
    from . import results

    return results.parse_wordcloud(obj, csum, cluster=None)


def run_class_cloud(
    corpus_path: str,
    analysis_dir: str,
    class_id: int,
    indice: int = 0,
    options: dict | None = None,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = True,
    output_dir: str | None = None,
) -> dict:
    """Word cloud of one ALCESTE class, weighted by its profile.

    ``indice`` 0 weights forms by their class frequency (profile ``eff``),
    1 by their chi-square with the class.  The profile is read from
    ``analysis_dir/profiles.csv`` (an ALCESTE run's output).
    """
    from functions import ReadProfileAsDico
    from textwordcloud import ClusterCloud

    if indice not in (0, 1):
        raise ValueError("indice must be 0 (frequency) or 1 (chi2)")
    prof_file = os.path.join(os.path.abspath(analysis_dir), "profiles.csv")
    if not os.path.isfile(prof_file):
        raise FileNotFoundError(f"no profiles.csv in {analysis_dir}; run corpus_alceste first")
    profiles = ReadProfileAsDico(prof_file, Alceste=True)
    if str(class_id) not in profiles:
        raise ValueError(f"class {class_id} not in profile (classes: {sorted(profiles)})")
    # rows after treat_line_alceste: [i, rank, eff, freq, chi2, '', form, p]
    clusterprof = [
        [row[6], row[2], row[4]]
        for row in profiles[str(class_id)][1:]
        if isinstance(row, list) and len(row) == 8 and row[6]
    ]
    if not clusterprof:
        raise ValueError(f"class {class_id} has no forms in its profile")

    _install_headless_selectcolumn()
    opts = _cloud_options(options)
    corpus, csum = textops.get_corpus(corpus_path, lang=lang, encoding=encoding)
    patches.sweep()
    parametres = {"type": "clustercloud", "lem": lem, "clusterprof": clusterprof, "indice": indice}
    if output_dir:
        parametres["pathout"] = output_dir

    class HeadlessClusterCloud(ClusterCloud):
        make_option = _make_cloud_option_filler(opts, indice=indice)

    obj = HeadlessClusterCloud(bootstrap.get(), corpus, parametres=parametres, dlg=False, lemdial=False)
    from . import results

    return results.parse_wordcloud(obj, csum, cluster={"class_id": class_id, "indice": indice})


# ---------------------------------------------------------------------- tgen


def _write_filtered_tgen(src: str, dest: str, corpus) -> dict:
    """Keep only lemmas present in the corpus; report the rest."""
    kept, dropped = {}, []
    with open(src, encoding="utf8") as f:
        for line in f:
            parts = [p.strip() for p in line.rstrip("\r\n").split("\t") if p.strip()]
            if len(parts) < 2:
                continue
            group, lems = parts[0], parts[1:]
            present = [lem for lem in lems if lem in corpus.lems]
            dropped += [lem for lem in lems if lem not in corpus.lems]
            if present:
                kept[group] = present
    if not kept:
        raise ValueError("none of the Tgen lemmas occur in the corpus")
    with open(dest, "w", encoding="utf8") as f:
        f.write("\n".join("\t".join([g, *lems]) for g, lems in kept.items()) + "\n")
    return {"groups": sorted(kept), "dropped_lemmas": sorted(set(dropped))}


def run_tgen_spec(
    corpus_path: str,
    tgen_path: str,
    modalities: list[str] | None = None,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = True,
    output_dir: str | None = None,
    active_keys: list[str] | None = None,
    supplementary_keys: list[str] | None = None,
) -> dict:
    """Specificity of Tgen groups across modalities (IRaMuTeQ TgenSpec)."""
    from textaslexico import TgenSpec

    tgen_src = os.path.abspath(tgen_path)
    if not os.path.isfile(tgen_src):
        raise FileNotFoundError(f"tgen file not found: {tgen_src}")
    corpus, csum = textops.get_corpus(corpus_path, lang=lang, encoding=encoding)
    patches.sweep()
    base = output_dir or csum["pathout"]
    outdir = new_outdir(base, "tgenspec")
    with textops.key_overrides(active_keys, supplementary_keys):
        textops._prep_lemmas(corpus, lem=lem)
    used = os.path.join(outdir, "tgen_used.txt")
    report = _write_filtered_tgen(tgen_src, used, corpus)
    etoiles = textops.resolve_listet(corpus, modalities, single_variable=False)
    parametres = {
        "type": "tgenspec",
        "tgenpath": used,
        "etoiles": list(etoiles),
        "pathout": outdir,
        "lem": lem,
    }
    obj = TgenSpec(bootstrap.get(), corpus, parametres)
    from . import results

    return results.parse_tgen(obj, csum, "spec", report)


def run_tgen_prof(
    analysis_dir: str,
    corpus_path: str,
    tgen_path: str,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = True,
    active_keys: list[str] | None = None,
    supplementary_keys: list[str] | None = None,
) -> dict:
    """Profile Tgen groups over the classes of an ALCESTE run (IRaMuTeQ TgenProf) — job payload."""
    from textreinert import TgenProf

    tgen_src = os.path.abspath(tgen_path)
    uce_file = os.path.join(os.path.abspath(analysis_dir), "uce.csv")
    if not os.path.isfile(tgen_src):
        raise FileNotFoundError(f"tgen file not found: {tgen_src}")
    if not os.path.isfile(uce_file):
        raise FileNotFoundError(f"no uce.csv in {analysis_dir}; run corpus_alceste first")
    corpus, csum = textops.get_corpus(corpus_path, lang=lang, encoding=encoding)
    patches.sweep()
    outdir = new_outdir(os.path.abspath(analysis_dir), "tgenprof")
    with textops.key_overrides(active_keys, supplementary_keys):
        textops._prep_lemmas(corpus, lem=lem)
    corpus.make_ucecl_from_R(uce_file)
    used = os.path.join(outdir, "tgen_used.txt")
    report = _write_filtered_tgen(tgen_src, used, corpus)
    parametres = {"type": "tgenprof", "tgenpath": used, "pathout": outdir, "lem": lem}
    obj = TgenProf(bootstrap.get(), corpus, parametres, None)
    from . import results

    return results.parse_tgen(obj, csum, "prof", report)
