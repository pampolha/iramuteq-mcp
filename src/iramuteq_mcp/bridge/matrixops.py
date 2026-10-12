"""Matrix (individuals x variables) analysis runners over the IRaMuTeQ sources.

Drives ``Tableau`` + ``AnalyseMatrix`` subclasses (Frequences, ChiSquare,
AnalyseQuest, DoSimi) with headless ``doparametres`` overrides that inject
the options the GUI dialogs would have collected.  Analyses are passed a
:class:`patches.StubProgress` as ``dlg`` because several ``doanalyse``
implementations call ``self.dlg.Update`` unconditionally; ``OpenAnalyse``
and ``ira.tree`` are neutralised by :mod:`patches` / :mod:`bootstrap`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading

from . import bootstrap, patches

log = logging.getLogger("iramuteq_mcp")

_matrix_cache: dict[str, object] = {}
_cache_lock = threading.Lock()

# graph options for similitude (values as produced by DoSimi.make_param defaults)
_SIMI_DEFAULTS = {
    "coeff": 0,
    "layout": 2,
    "type_graph": 1,
    "arbremax": 1,
    "coeff_tv": 0,
    "coeff_tv_nb": 10,
    "tvprop": 1,
    "tvmin": 5,
    "tvmax": 30,
    "coeff_te": 1,
    "coeff_temin": 1,
    "coeff_temax": 10,
    "label_v": 1,
    "label_e": 1,
    "vcex": 0,
    "vcexmin": 8,
    "vcexmax": 25,
    "cex": 10,
    "seuil_ok": 0,
    "seuil": 1,
    "cols": (255, 0, 0),
    "cola": (200, 200, 200),
    "width": 800,
    "height": 800,
    "first": True,
    "keep_coord": False,
    "alpha": 10,
    "film": False,
    "svg": 0,
    "halo": 0,
    "com": 0,
    "communities": 0,
}

_CHI_OPTIONS = (
    "valobs",
    "valtheo",
    "resi",
    "contrib",
    "pourcent",
    "pourcentl",
    "pourcentc",
    "graph",
    "bw",
)


# ---------------------------------------------------------------------- open


def _matrix_key(path: str, opts: dict) -> str:
    st = os.stat(path)
    blob = json.dumps({"path": path, "mtime": st.st_mtime_ns, **opts}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def open_matrix(
    matrix_path: str,
    filetype: str | None = None,
    encoding: str = "utf-8",
    colsep: str = ";",
    sheetnb: int = 1,
    firstrowiscolnames: bool = True,
    firstcolisrownames: bool = True,
    output_dir: str | None = None,
):
    """Load a .csv/.xls/.ods table as an IRaMuTeQ Tableau (cached).

    Returns ``(tableau, summary)``.
    """
    from tableau import Tableau

    ira = bootstrap.get()
    path = os.path.abspath(matrix_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"matrix not found: {path}")
    if filetype is None:
        ext = os.path.splitext(path)[1].lower()
        filetype = {".xls": "xls", ".xlsx": "xls", ".ods": "ods"}.get(ext, "csv")

    opts = {
        "filetype": filetype,
        "encoding": encoding,
        "colsep": colsep,
        "sheetnb": sheetnb,
        "firstrowiscolnames": firstrowiscolnames,
        "firstcolisrownames": firstcolisrownames,
    }
    key = _matrix_key(path, opts)
    with _cache_lock:
        cached = _matrix_cache.get(key)
        if cached is not None:
            return cached

    tableau = Tableau(ira, path, filetype=filetype, encodage=encoding)
    tableau.parametres.update(opts)
    tableau.parametres["syscoding"] = "utf8"
    if output_dir:
        tableau.parametres["pathout"] = output_dir
    tableau.firstrowiscolnames = firstrowiscolnames
    tableau.firstcolisrownames = firstcolisrownames
    tableau.make_content()

    summary = matrix_summary(tableau)
    with _cache_lock:
        _matrix_cache[key] = (tableau, summary)
    return tableau, summary


def matrix_summary(tableau) -> dict:
    p = tableau.parametres
    return {
        "matrix_uuid": p.get("uuid"),
        "matrix_name": p.get("matrix_name"),
        "pathout": p.get("pathout"),
        "ira": p.get("ira"),
        "csvfile": p.get("csvfile"),
        "columns": tableau.get_colnames(),
        "rows": tableau.rownb,
    }


def _get_matrix(matrix_path: str, **kwargs):
    return open_matrix(matrix_path, **kwargs)


# ------------------------------------------------------------------ analyses


def run_frequencies(
    matrix_path: str,
    columns: list[int],
    include_na: bool = False,
    header: bool = True,
    **open_kwargs,
) -> dict:
    """Frequencies of selected columns (IRaMuTeQ Frequences)."""
    from tabfrequence import Frequences

    tableau, msum = _get_matrix(matrix_path, **open_kwargs)
    patches.sweep()

    class HeadlessFreq(Frequences):
        def doparametres(self, dlg=None):
            self.parametres["colsel"] = list(columns)
            self.parametres["header"] = header
            self.parametres["NA"] = include_na

    parametres = {"type": "freq", "colsel": list(columns), "header": header, "NA": include_na}
    obj = HeadlessFreq(bootstrap.get(), tableau, parametres, dlg=patches.StubProgress())
    from . import results

    return results.parse_frequencies(obj, msum)


def run_chi2(
    matrix_path: str,
    columns_a: list[int],
    columns_b: list[int],
    options: dict | None = None,
    **open_kwargs,
) -> dict:
    """Chi-square independence tests between two column sets (IRaMuTeQ ChiSquare)."""
    from tabchi2 import ChiSquare, chioption

    tableau, msum = _get_matrix(matrix_path, **open_kwargs)
    patches.sweep()

    opts = {k: bool(v) for k, v in (options or {}).items() if k in _CHI_OPTIONS}
    opts = {**{k: chioption.get(k, True) for k in _CHI_OPTIONS}, **opts}

    class HeadlessChi2(ChiSquare):
        def doparametres(self, dlg=None):
            self.colsel1 = list(columns_a)
            self.colsel2 = list(columns_b)
            self.chioption = opts
            self.parametres.update(opts)

    parametres = {"type": "chi2", **opts}
    obj = HeadlessChi2(bootstrap.get(), tableau, parametres, dlg=patches.StubProgress())
    from . import results

    return results.parse_chi2(obj, msum)


def run_mcnemar(
    matrix_path: str,
    columns_a: list[int],
    columns_b: list[int],
    options: dict | None = None,
    **open_kwargs,
) -> dict:
    """McNemar chi-square tests between two column sets."""
    from tabchi2mcnemar import McNemar

    tableau, msum = _get_matrix(matrix_path, **open_kwargs)
    patches.sweep()

    opts = {**{k: True for k in _CHI_OPTIONS}, **{k: bool(v) for k, v in (options or {}).items() if k in _CHI_OPTIONS}}

    class HeadlessMcNemar(McNemar):
        def doparametres(self, dlg=None):
            self.colsel1 = list(columns_a)
            self.colsel2 = list(columns_b)
            self.chioption = opts
            self.parametres.update(opts)

    parametres = {"type": "chi2mcnemar", **opts}
    obj = HeadlessMcNemar(bootstrap.get(), tableau, parametres, dlg=patches.StubProgress())
    from . import results

    return results.parse_chi2(obj, msum)


def run_matrix_reinert(
    matrix_path: str,
    active_columns: list[int] | None = None,
    supplementary_columns: list[int] | None = None,
    formatted: bool = False,
    nbcl_p1: int = 10,
    mincl: int = 2,
    output_dir: str | None = None,
    **open_kwargs,
) -> dict:
    """Reinert classification on a matrix (IRaMuTeQ AnalyseQuest) — job payload."""
    from tabchdalc import AnalyseQuest

    tableau, msum = _get_matrix(matrix_path, **open_kwargs)
    patches.sweep()

    parametres = {"type": "reinertmatrix", "nbcl_p1": nbcl_p1, "mincl": mincl}
    if formatted:
        parametres["formatted"] = 1
    else:
        if not active_columns:
            raise ValueError("active_columns is required (or use formatted=True)")
        ncol = len(tableau.get_colnames())
        active = list(active_columns)
        # IRaMuTeQ's profile stage always reads a supplementary table; default
        # to the remaining columns (illustrative variables) so it is never empty
        sup = list(supplementary_columns) if supplementary_columns is not None else [
            c for c in range(ncol) if c not in active
        ]
        if not sup:
            raise ValueError(
                "no supplementary columns left; pass supplementary_columns explicitly"
            )
        parametres["listact"] = active
        parametres["listsup"] = sup
    if output_dir:
        parametres["pathout"] = output_dir

    class HeadlessQuest(AnalyseQuest):
        def doparametres(self, dlg=None):
            pass  # everything comes in self.parametres

    obj = HeadlessQuest(bootstrap.get(), tableau, parametres, dlg=patches.StubProgress())
    from . import results

    return results.parse_matrix_reinert(obj, msum)


def run_matrix_similitude(
    matrix_path: str,
    columns: list[int],
    words: list[str] | None = None,
    options: dict | None = None,
    output_dir: str | None = None,
    **open_kwargs,
) -> dict:
    """Similarity graph on matrix columns (IRaMuTeQ DoSimi) — job payload."""
    from chemins import PathOut
    from functions import indices_simi
    from tabsimi import DoSimi

    tableau, msum = _get_matrix(matrix_path, **open_kwargs)
    patches.sweep()

    paramsimi = dict(_SIMI_DEFAULTS)
    paramsimi.update({k: v for k, v in (options or {}).items() if k in _SIMI_DEFAULTS})
    paramsimi["first"] = True

    class HeadlessSimi(DoSimi):
        def doparametres(self, dlg=None):
            p = self.parametres
            self.fromprof = False
            self.wordgraph = p.get("wordgraph", False)
            self.listactives = False
            self.actives = False
            self.openfromprof = False
            self.cmd = p.get("cmd", False)
            self.dirout = p.get("pathout", False)
            self.paramsimi = dict(paramsimi)
            self.indices = indices_simi
            self.Source = None
            if self.dirout:
                self.pathout = PathOut(dirout=self.dirout)
            self.tableau.parametres["mineff"] = 0
            self.tableau.selected_col = list(columns)
            self.tableau.getactlistfromselection(self.tableau.selected_col)
            self.tableau.make_listactives()
            if words:
                wanted = {w: i for i, w in enumerate(self.tableau.listactives)}
                self.column = sorted(wanted[w] for w in words if w in wanted)
            else:
                self.column = list(range(len(self.tableau.listactives)))
            self.parametres.update(self.paramsimi)
            if not self.parametres.get("pathout"):
                self.parametres["pathout"] = self.pathout.dirout
            self.pathout.createdir(self.parametres["pathout"])
            self.pathout.dirout = self.parametres["pathout"]

    parametres = {"type": "simimatrix", **paramsimi}
    if output_dir:
        parametres["pathout"] = output_dir
    obj = HeadlessSimi(bootstrap.get(), tableau, parametres, dlg=patches.StubProgress())
    from . import results

    return results.parse_similitude(obj, msum)
