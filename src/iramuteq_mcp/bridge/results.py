"""Parse IRaMuTeQ analysis outputs into structured summaries.

Reuses the source's own readers (``functions.ReadProfileAsDico``,
``functions.read_list_file``, ``functions.DoConf``) wherever they exist;
only shapes them for MCP responses.  Every result carries the artifact
paths so agents can open files the GUI would display.
"""

from __future__ import annotations

import csv
import logging
import os

log = logging.getLogger("iramuteq_mcp")


def _artifacts(dirpath: str) -> dict:
    out = {}
    if not dirpath or not os.path.isdir(dirpath):
        return out
    for name in sorted(os.listdir(dirpath)):
        p = os.path.join(dirpath, name)
        if os.path.isfile(p):
            out[name] = p
    return out


def _read_csv(path: str, sep: str = ";") -> list[list[str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf8", errors="replace") as f:
        return [row for row in csv.reader(f, delimiter=sep)]


# ----------------------------------------------------------------------- stat


def parse_stats(obj, corpus_summary: dict) -> dict:
    res = getattr(obj, "result", {}) or {}
    outdir = obj.parametres.get("pathout")

    def summarize(d, top=50):
        rows = sorted(d.values(), key=lambda v: v[1], reverse=True) if d else []
        return [
            {"form": v[0], "freq": v[1], "gram": v[2] if len(v) > 2 else ""} for v in rows[:top]
        ]

    glob = res.get("glob", "")
    if isinstance(glob, list):
        glob = "\n".join(str(x) for x in glob)
    return {
        "analysis": "stat",
        "output_dir": outdir,
        "corpus": corpus_summary,
        "summary": glob,
        "counts": {
            "forms_total": len(res.get("total", {})),
            "hapax": len(res.get("hapax", {})),
            "active_forms": len(res.get("formes_actives", {})),
            "supplementary_forms": len(res.get("formes_supplémentaires", {})),
        },
        "top_forms": summarize(res.get("total", {})),
        "top_active_forms": summarize(res.get("formes_actives", {})),
        "artifacts": _artifacts(outdir),
    }


# -------------------------------------------------------------------- alceste

# fields produced by functions.treat_line_alceste for a profiles.csv row
# (row columns: rank;eff;freq%;chi2;form;p — add_type appends the POS slot)
_PROFILE_FIELDS = ["index", "rank", "eff", "freq_percent", "chi2", "typo", "form", "p"]


def _read_profiles(path: str, top: int = 30) -> dict:
    from functions import ReadProfileAsDico

    if not os.path.isfile(path):
        return {}
    try:
        raw = ReadProfileAsDico(path, Alceste=True)
    except Exception:
        log.exception("ReadProfileAsDico failed on %s", path)
        return {}
    out = {}
    for cluster, rows in raw.items():
        head = rows[0] if rows else []
        body = []
        for row in rows[1:top + 1]:
            if len(row) >= 8:
                body.append(dict(zip(_PROFILE_FIELDS, row)))
        out[str(cluster)] = {
            "cluster_header": [str(v) for v in head],
            "top_forms": body,
        }
    return out


def _class_sizes(uce_csv: str) -> dict:
    rows = _read_csv(uce_csv)
    sizes: dict[str, int] = {}
    for row in rows[1:] if rows else []:
        if len(row) >= 2 and row[0] != "uce":
            sizes[row[1]] = sizes.get(row[1], 0) + 1
    return sizes


def _maybe(path: str) -> str | None:
    return path if path and os.path.isfile(path) else None


def _read_ira(path: str) -> dict:
    from functions import DoConf

    try:
        return DoConf(path).getoptions()
    except Exception:
        return {}


def parse_alceste(obj, corpus_summary: dict) -> dict:
    outdir = obj.parametres.get("pathout")
    po = obj.pathout
    return {
        "analysis": "alceste",
        "output_dir": outdir,
        "corpus": corpus_summary,
        "parameters": {
            k: obj.parametres.get(k)
            for k in (
                "classif_mode",
                "nbcl_p1",
                "mincl",
                "max_actives",
                "eff_min_forme",
                "nbactives",
                "tailleuc1",
                "tailleuc2",
                "svdmethod",
                "lem",
                "clnb",
                "time",
            )
        },
        "class_sizes": _class_sizes(po["uce"]),
        "profiles": _read_profiles(po["profile"]),
        "antiprofiles": _read_profiles(po["antiprofile"]),
        "afc_factors": _read_csv(po["afc_facteur"]),
        "chisq_table": _read_csv(po["chisqtable"])[:50],
        "graphs": {
            "afc": _maybe(po["AFC2DL_OUT"]),
            "afc_sup": _maybe(po["AFC2DSL_OUT"]),
            "afc_et": _maybe(po["AFC2DEL_OUT"]),
            "afc_classes": _maybe(po["AFC2DCL_OUT"]),
            "dendro1": _maybe(po["dendro1"]),
            "dendro2": _maybe(po["dendro2"]),
        },
        "artifacts": _artifacts(outdir),
    }


# --------------------------------------------------------------------- lexico


def parse_lexico(obj, corpus_summary: dict) -> dict:
    outdir = obj.parametres.get("pathout")
    # Lexico writes via StatTxtPathOut-style absolute names (with .csv)
    j = lambda name: os.path.join(outdir, name) if outdir else ""
    return {
        "analysis": "lexico",
        "output_dir": outdir,
        "corpus": corpus_summary,
        "parameters": {k: obj.parametres.get(k) for k in ("mineff", "indice", "typeformes", "clnb", "time")},
        "specificities_forms": _read_csv(j("tablespecf.csv"))[:100],
        "specificities_types": _read_csv(j("tablespect.csv"))[:100],
        "banalites": _read_csv(j("banalites.csv"))[:50],
        "contingency_forms": _read_csv(j("tableafcm.csv"))[:50],
        "artifacts": _artifacts(outdir),
    }


# ---------------------------------------------------------------- similitude


def parse_similitude(obj, corpus_summary: dict) -> dict:
    outdir = obj.parametres.get("pathout")
    from functions import read_list_file

    graphs = []
    liste = os.path.join(outdir, "liste_graph.txt") if outdir else None
    if liste and os.path.isfile(liste):
        try:
            graphs = read_list_file(liste)
        except Exception:
            graphs = []
    return {
        "analysis": "similitude",
        "output_dir": outdir,
        "corpus": corpus_summary,
        "parameters": {k: obj.parametres.get(k) for k in ("coeff", "layout", "seuil", "arbremax", "type_graph", "time")},
        "graphs": [[str(v) for v in row] for row in graphs],
        "artifacts": _artifacts(outdir),
    }


# ------------------------------------------------------------- matrix results


def _read_tsv(path: str) -> list[list[str]]:
    return _read_csv(path, sep="\t")


def parse_frequencies(obj, matrix_summary: dict) -> dict:
    """Parse the R outframe of Frequences (groups separated by '***' marker rows)."""
    rows = _read_tsv(getattr(obj, "outframe", ""))
    titles = list(getattr(obj, "listtitre", []))
    tables: list[dict] = []
    current: list[list[str]] = []
    for row in rows[1:] if rows else []:
        if row and row[0] == "***":
            if current:
                tables.append(current)
            current = []
        else:
            current.append(row)
    if current:
        tables.append(current)

    out_tables = []
    for i, table in enumerate(tables):
        entries = []
        for row in table:
            if len(row) >= 3:
                entries.append({"label": row[2], "effectif": row[0], "pourcentage": row[1]})
        title = titles[i] if i < len(titles) else f"column_{i}"
        out_tables.append({"column": title, "frequencies": entries})
    return {
        "analysis": "freq",
        "output_dir": obj.parametres.get("pathout"),
        "matrix": matrix_summary,
        "tables": out_tables,
        "artifacts": _artifacts(obj.parametres.get("pathout")),
    }


def parse_chi2(obj, matrix_summary: dict) -> dict:
    """Parse the tagged R outframe of ChiSquare/McNemar plus its HTML report."""
    rows = _read_tsv(getattr(obj, "OutFrame", ""))
    tags: dict[str, list[list[str]]] = {}
    results_rows: list[list[str]] = []
    for row in rows:
        if not row:
            continue
        if row[0] == "res":
            results_rows.append(row[1:])
        elif len(row) > 1 and row[1].startswith("*") and row[1].endswith("*"):
            tags.setdefault(row[1].strip("*"), []).append(row[2:])
    return {
        "analysis": getattr(obj, "parametres", {}).get("type", "chi2"),
        "output_dir": obj.parametres.get("pathout"),
        "matrix": matrix_summary,
        "crosstabs": [str(x) for x in getattr(obj, "TextCroise", [])],
        "results": results_rows,
        "tables": tags,
        "artifacts": _artifacts(obj.parametres.get("pathout")),
    }


def parse_matrix_reinert(obj, matrix_summary: dict) -> dict:
    outdir = obj.parametres.get("pathout")
    po = obj.pathout
    return {
        "analysis": "reinertmatrix",
        "output_dir": outdir,
        "matrix": matrix_summary,
        "parameters": {
            k: obj.parametres.get(k) for k in ("nbcl_p1", "mincl", "listact", "listsup", "formatted", "clnb", "time")
        },
        "class_sizes": _class_sizes(po["uce"]),
        "profiles": _read_profiles(po["profile"]),
        "antiprofiles": _read_profiles(po["antiprofile"]),
        "afc_factors": _read_csv(po["afc_facteur"]),
        "chisq_table": _read_csv(po["chisqtable"])[:50],
        "graphs": {
            "afc": _maybe(po["AFC2DL_OUT"]),
            "afc_classes": _maybe(po["AFC2DCL_OUT"]),
            "dendro1": _maybe(po["dendro1"]),
        },
        "artifacts": _artifacts(outdir),
    }


# ------------------------------------------------------------- generic reader


def read_analysis(path: str) -> dict:
    """Summarize any finished analysis directory (reads Analyse.ira + outputs)."""
    path = os.path.abspath(path)
    if os.path.isfile(path) and path.endswith(".ira"):
        path = os.path.dirname(path)
    ira_file = os.path.join(path, "Analyse.ira")
    info = _read_ira(ira_file)
    res = {
        "analysis_dir": path,
        "type": info.get("type"),
        "parameters": info,
        "artifacts": _artifacts(path),
    }
    if os.path.isfile(os.path.join(path, "uce.csv")):
        res["class_sizes"] = _class_sizes(os.path.join(path, "uce.csv"))
    if os.path.isfile(os.path.join(path, "profiles.csv")):
        res["profiles"] = _read_profiles(os.path.join(path, "profiles.csv"))
    return res


def list_history() -> dict:
    from . import bootstrap

    ira = bootstrap.get()
    corpora = []
    for entry in ira.history.history:
        item = dict(entry)
        item["analyses"] = [dict(a) for a in entry.get("analyses", [])]
        corpora.append(item)
    matrices = []
    for entry in ira.history.matrix:
        item = dict(entry)
        item["analyses"] = [dict(a) for a in entry.get("analyses", [])]
        matrices.append(item)
    return {"corpora": corpora, "matrices": matrices, "jobs": ira.history.filein}


# ------------------------------------------------- package A/B result parsers


def parse_freqmulti(obj, matrix_summary: dict) -> dict:
    outdir = obj.parametres.get("pathout")
    rows = getattr(obj, "freq", []) or []
    header, body = (rows[0], rows[1:]) if rows else ([], [])
    return {
        "analysis": "freqmulti",
        "output_dir": outdir,
        "matrix": matrix_summary,
        "columns": header,
        "frequencies": [dict(zip(header, r)) for r in body],
        "artifacts": _artifacts(outdir),
    }


def parse_splitvar(obj, matrix_summary: dict, modalities: list[str]) -> dict:
    outdir = obj.parametres.get("pathout")
    created = {}
    for mod in modalities:
        found = sorted(
            os.path.join(outdir, d, "csvfile.csv")
            for d in os.listdir(outdir)
            if d.startswith(f"{mod}_matrix_") and os.path.isfile(os.path.join(outdir, d, "csvfile.csv"))
        ) if outdir and os.path.isdir(outdir) else []
        created[mod] = found[-1] if found else None
    return {
        "analysis": "splitvar",
        "output_dir": outdir,
        "matrix": matrix_summary,
        "modalities": created,
        "artifacts": _artifacts(outdir),
    }


def parse_labbe(obj, corpus_summary: dict) -> dict:
    outdir = obj.parametres.get("pathout")
    j = lambda name: os.path.join(outdir, name) if outdir else ""
    return {
        "analysis": "labbe",
        "output_dir": outdir,
        "corpus": corpus_summary,
        "parameters": {k: obj.parametres.get(k) for k in ("mineff", "indice", "typeformes", "clnb")},
        "modalities": list(getattr(obj, "listet", [])),
        "distances": _read_csv(j("listdist.csv"))[:200],
        "distance_matrix": _read_csv(j("distmat.csv"))[:50],
        "artifacts": _artifacts(outdir),
    }


def _count_lines(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path, encoding="utf8") as f:
        return sum(1 for line in f if line.strip())


def parse_wordcloud(obj, corpus_summary: dict, cluster: dict | None) -> dict:
    outdir = obj.parametres.get("pathout")
    graph = obj.parametres.get("graphout")
    actives = list(getattr(obj, "actives", []) or [])
    return {
        "analysis": "clustercloud" if cluster else "wordcloud",
        "output_dir": outdir,
        "corpus": corpus_summary,
        "cluster": cluster,
        "graph": graph if graph and os.path.isfile(graph) else None,
        "parameters": {k: obj.parametres.get(k) for k in ("mode", "maxword", "width", "height", "svg")},
        "forms_count": len(actives),
        "selected_count": _count_lines(os.path.join(outdir, "selected.csv")) if outdir else 0,
        "forms_sample": actives[:50],
        "artifacts": _artifacts(outdir),
    }


def parse_proto(obj, matrix_summary: dict) -> dict:
    outdir = obj.parametres.get("pathout")
    table = _read_csv(os.path.join(outdir, "table.csv"), sep="\t") if outdir else []
    return {
        "analysis": "proto",
        "output_dir": outdir,
        "matrix": matrix_summary,
        "parameters": {k: obj.parametres.get(k) for k in ("freqmin", "limfreq", "limrang", "typegraph")},
        "words": [{"word": r[0].strip('"'), "freq": r[1], "mean_rank": r[2]} for r in table if len(r) >= 3][:200],
        "graph": _maybe(os.path.join(outdir, "proto.png")) if outdir else None,
        "artifacts": _artifacts(outdir),
    }


def parse_tgen(obj, corpus_summary: dict, kind: str, report: dict) -> dict:
    outdir = obj.parametres.get("pathout")
    files = (
        {"specificities": "tgenspec.csv", "effectives": "tgeneff.csv"}
        if kind == "spec"
        else {
            "chi2_groups": "tgenchi2.csv",
            "chi2_lemmas": "tgenlemchi2.csv",
            "effectives_groups": "tgeneff.csv",
            "effectives_lemmas": "tgenlemeff.csv",
        }
    )
    tables = {
        label: _read_csv(os.path.join(outdir, name), sep="\t")[:100] for label, name in files.items()
    }
    return {
        "analysis": f"tgen_{kind}",
        "output_dir": outdir,
        "corpus": corpus_summary,
        "tgen": report,
        "tables": tables,
        "artifacts": _artifacts(outdir),
    }
