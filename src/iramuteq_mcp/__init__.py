"""IRaMuTeQ MCP server — headless substitute interface to IRaMuTeQ 0.8.

The server is a thin facade over the IRaMuTeQ sources (GPL-2, Pierre
Ratinaud): corpus parsing, lemmatisation, ALCESTE classification, similarity
graphs etc. are performed by IRaMuTeQ's own Python + R code, imported at
runtime from ``$IRAMUTEQ_PATH`` (default ``~/Tools/iramuteq-0.8alpha7-sources``).
Nothing is reimplemented; outputs are the files IRaMuTeQ writes, parsed by
the source's own readers.

Long R runs (ALCESTE, similarity, specificities) are executed as background
jobs: the run tool returns a job id immediately and the agent polls with
job_status / job_result / job_cancel.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Annotated

from fastmcp import FastMCP

from .bridge import bootstrap, jobs, matrixops, results, textops

mcp = FastMCP("iramuteq")

R_PACKAGES = [
    "ca",
    "gee",
    "ape",
    "igraph",
    "proxy",
    "wordcloud",
    "irlba",
    "textometry",
    "sna",
    "network",
    "intergraph",
    "rgl",
    "cluster",
    "MASS",
    "Matrix",
]


def _job_manager() -> jobs.JobManager:
    ctx = bootstrap.get()
    if not hasattr(ctx, "_job_manager"):
        ctx._job_manager = jobs.JobManager(os.path.join(ctx.UserConfigPath, "mcp-jobs"))
    return ctx._job_manager


def _submit(kind: str, params: dict, runner) -> dict:
    job = _job_manager().submit(kind, params, runner)
    return job.summary()


# ---------------------------------------------------------------------- status


@mcp.tool
async def status() -> dict:
    """Health check: IRaMuTeQ sources, config, R executable and R packages."""
    ctx = await asyncio.to_thread(bootstrap.get)
    missing = []
    try:
        proc = subprocess.run(
            [
                ctx.RPath,
                "--slave",
                "-e",
                "cat(paste(c(%s), collapse=','))"
                % ",".join('"%s"' % p for p in R_PACKAGES),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        installed = set(proc.stdout.strip().split(",")) if proc.returncode == 0 else set()
        missing = [p for p in R_PACKAGES if p not in installed]
    except Exception as exc:  # noqa: BLE001
        missing = [f"cannot check R packages: {exc}"]
    r_version = ""
    try:
        r_version = subprocess.run(
            [ctx.RPath, "--version"], capture_output=True, text=True, timeout=10
        ).stdout.splitlines()[0]
    except Exception:
        pass
    return {
        "iramuteq_path": ctx.AppliPath,
        "user_config": ctx.UserConfigPath,
        "r_path": ctx.RPath,
        "r_version": r_version,
        "r_packages_missing": missing,
        "history": ctx.history.filein,
        "ready": not missing,
    }


# ---------------------------------------------------------------------- corpus


@mcp.tool
async def corpus_validate(
    path: Annotated[str, "Absolute path to the ALCESTE-formatted .txt corpus"],
    encoding: Annotated[str, "File encoding"] = "utf8",
) -> dict:
    """Validate ALCESTE corpus formatting (**** headers, -* themes, *variables)."""
    return await asyncio.to_thread(textops.validate_corpus, path, encoding)


@mcp.tool
async def corpus_build(
    path: Annotated[str, "Absolute path to the ALCESTE-formatted .txt corpus"],
    lang: Annotated[str, "Corpus language for Lexique 3 (french, english, ...)"] = "french",
    encoding: Annotated[str, "File encoding"] = "utf8",
    ucemethod: Annotated[int, "UCE segmentation: 0=chars, 1=words, 2=paragraphs"] = 1,
    ucesize: Annotated[int, "Target UCE size (chars or words depending on ucemethod)"] = 40,
) -> dict:
    """Build (and cache) the IRaMuTeQ corpus: parse, clean, lemmatise, segment UCEs."""
    options = {"ucemethod": ucemethod, "ucesize": ucesize}

    def run():
        _, summary = textops.build_corpus(path, lang=lang, encoding=encoding, options=options)
        return summary

    return await asyncio.to_thread(run)


@mcp.tool
async def corpus_stats(
    path: Annotated[str, "Absolute path to the corpus (built on first use)"],
    lang: Annotated[str, "Corpus language"] = "french",
    encoding: Annotated[str, "File encoding"] = "utf8",
    lem: Annotated[bool, "Lemmatise (verbs to infinitive, nouns to masc. sing.)"] = True,
) -> dict:
    """Textual statistics: frequencies, active/supplementary forms, hapax, abstract."""
    return await asyncio.to_thread(textops.run_stats, path, lang, encoding, lem)


@mcp.tool
async def corpus_concordance(
    path: Annotated[str, "Absolute path to the corpus"],
    lem: Annotated[str, "Lemma or form to search (see corpus_stats for the list)"],
    limit: Annotated[int, "Maximum number of hits"] = 50,
    width: Annotated[int, "Context characters on each side"] = 60,
    lang: Annotated[str, "Corpus language"] = "french",
    encoding: Annotated[str, "File encoding"] = "utf8",
) -> dict:
    """Keyword-in-context concordance for a lemma, with UCI/variables of each hit."""
    def run():
        corpus, _ = textops.build_corpus(path, lang=lang, encoding=encoding)
        return textops.concordance(corpus, lem, limit=limit, width=width)

    return await asyncio.to_thread(run)


# --------------------------------------------------------------- text analyses


def _alceste_job(path, lang, encoding, lem, output_dir, options):
    def runner(job):
        job.phase = "corpus + CHD + profiles/AFC (R)"
        res = textops.run_alceste(
            path, lang=lang, encoding=encoding, lem=lem, output_dir=output_dir, options=options
        )
        job.phase = "done"
        return res

    return _submit(
        "corpus_alceste",
        {"path": path, "lang": lang, "encoding": encoding, "lem": lem, "options": options or {}},
        runner,
    )


@mcp.tool
async def corpus_alceste(
    path: Annotated[str, "Absolute path to the corpus"],
    classif_mode: Annotated[int, "0=double on UC, 1=simple on UCE, 2=simple on UCI"] = 1,
    nbcl_p1: Annotated[int, "Terminal classes of classification phase 1"] = 10,
    mincl: Annotated[int, "Minimum UCEs per class (0 = automatic)"] = 0,
    max_actives: Annotated[int, "Maximum number of active forms analysed (memory!)"] = 20000,
    tailleuc1: Annotated[int, "Target size of UC table 1 (active occurrences)"] = 12,
    tailleuc2: Annotated[int, "Target size of UC table 2 (active occurrences)"] = 14,
    svdmethod: Annotated[str, "SVD method: svdR, irlba or svdlibc"] = "irlba",
    lem: Annotated[bool, "Lemmatise"] = True,
    lang: Annotated[str, "Corpus language"] = "french",
    encoding: Annotated[str, "File encoding"] = "utf8",
    output_dir: Annotated[str | None, "Optional explicit output directory"] = None,
) -> dict:
    """Start an ALCESTE/Reinert classification as a background job (returns job_id)."""
    options = {
        "classif_mode": classif_mode,
        "nbcl_p1": nbcl_p1,
        "mincl": mincl,
        "max_actives": max_actives,
        "tailleuc1": tailleuc1,
        "tailleuc2": tailleuc2,
        "svdmethod": svdmethod,
        "mode.patate": 0,
    }
    return _alceste_job(path, lang, encoding, lem, output_dir, options)


@mcp.tool
async def corpus_lexico(
    path: Annotated[str, "Absolute path to the corpus"],
    modalities: Annotated[list[str] | None, "variable_modality entries to compare; must share ONE variable (empty = first variable's modalities)"] = None,
    mineff: Annotated[int, "Minimum frequency of analysed forms"] = 10,
    indice: Annotated[str, "Specificity index: hypergeo or chi2"] = "hypergeo",
    lem: Annotated[bool, "Lemmatise"] = True,
    lang: Annotated[str, "Corpus language"] = "french",
    encoding: Annotated[str, "File encoding"] = "utf8",
    output_dir: Annotated[str | None, "Optional explicit output directory"] = None,
) -> dict:
    """Start a 'Comme Lexico' specificities analysis as a background job (returns job_id)."""
    options = {"mineff": mineff, "indice": indice}

    def runner(job):
        job.phase = "specificities + AFC (R)"
        return textops.run_lexico(
            path,
            modalities=modalities,
            lang=lang,
            encoding=encoding,
            lem=lem,
            output_dir=output_dir,
            options=options,
        )

    return _submit(
        "corpus_lexico",
        {"path": path, "modalities": modalities, "options": options},
        runner,
    )


@mcp.tool
async def corpus_similitude(
    path: Annotated[str, "Absolute path to the corpus"],
    forms: Annotated[list[str] | None, "Lemma subset to plot (empty = all active forms)"] = None,
    coeff: Annotated[int, "Similarity index (0=cooccurrence, 3=Jaccard, 12=Dice, ...)"] = 0,
    layout: Annotated[int, "Graph layout (0=random, 1=circle, 2=Fruchterman-Reingold, ...)"] = 2,
    seuil: Annotated[float, "Edge threshold"] = 1,
    arbremax: Annotated[bool, "Reduce the graph to a maximum spanning tree"] = True,
    lem: Annotated[bool, "Lemmatise"] = True,
    lang: Annotated[str, "Corpus language"] = "french",
    encoding: Annotated[str, "File encoding"] = "utf8",
    output_dir: Annotated[str | None, "Optional explicit output directory"] = None,
) -> dict:
    """Start a similarity-graph analysis between active forms as a background job."""
    options = {"coeff": coeff, "layout": layout, "seuil": seuil, "arbremax": int(arbremax)}

    def runner(job):
        job.phase = "similarity matrix + graph (R)"
        selected = None
        if forms:
            corpus, _ = textops.build_corpus(path, lang=lang, encoding=encoding)
            textops._prep_lemmas(corpus)
            from functions import DoConf

            keys = DoConf(bootstrap.get().ConfigPath["key"]).getoptions()
            actives, _ = corpus.make_actives_nb(20000, 1)
            pos = {lem_: i for i, lem_ in enumerate(actives)}
            selected = sorted(pos[f] for f in forms if f in pos)
        return textops.run_similitude(
            path,
            selected=selected,
            lang=lang,
            encoding=encoding,
            lem=lem,
            output_dir=output_dir,
            options=options,
        )

    return _submit(
        "corpus_similitude",
        {"path": path, "forms": forms, "options": options},
        runner,
    )


# ---------------------------------------------------------------------- matrix


@mcp.tool
async def matrix_open(
    path: Annotated[str, "Absolute path to the .csv/.xls/.ods table"],
    filetype: Annotated[str | None, "csv, xls or ods (default: from extension)"] = None,
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
    sheetnb: Annotated[int, "Sheet number for xls/ods"] = 1,
    firstrowiscolnames: Annotated[bool, "First row holds column names"] = True,
    firstcolisrownames: Annotated[bool, "First column holds row names"] = True,
) -> dict:
    """Load an individuals x variables table as an IRaMuTeQ matrix (cached)."""

    def run():
        _, summary = matrixops.open_matrix(
            path,
            filetype=filetype,
            encoding=encoding,
            colsep=colsep,
            sheetnb=sheetnb,
            firstrowiscolnames=firstrowiscolnames,
            firstcolisrownames=firstcolisrownames,
        )
        return summary

    return await asyncio.to_thread(run)


@mcp.tool
async def matrix_frequencies(
    path: Annotated[str, "Absolute path to the table"],
    columns: Annotated[list[int], "0-based column indices to tabulate"],
    include_na: Annotated[bool, "Count NA values"] = False,
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
) -> dict:
    """Frequency table + barplot per selected column (IRaMuTeQ Frequences)."""
    return await asyncio.to_thread(
        matrixops.run_frequencies,
        path,
        columns,
        include_na,
        True,
        encoding=encoding,
        colsep=colsep,
    )


@mcp.tool
async def matrix_chi2(
    path: Annotated[str, "Absolute path to the table"],
    columns_a: Annotated[list[int], "0-based column indices (first variable set)"],
    columns_b: Annotated[list[int], "0-based column indices (second variable set)"],
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
) -> dict:
    """Chi-square independence tests between two column sets (IRaMuTeQ ChiSquare)."""
    return await asyncio.to_thread(
        matrixops.run_chi2,
        path,
        columns_a,
        columns_b,
        None,
        encoding=encoding,
        colsep=colsep,
    )


@mcp.tool
async def matrix_reinert(
    path: Annotated[str, "Absolute path to the table"],
    active_columns: Annotated[list[int] | None, "Active column indices (omit if formatted)"] = None,
    supplementary_columns: Annotated[list[int] | None, "Illustrative column indices"] = None,
    formatted: Annotated[bool, "ALCESTE-formatted table (*modalities are supplementary)"] = False,
    nbcl_p1: Annotated[int, "Terminal classes of phase 1"] = 10,
    mincl: Annotated[int, "Minimum rows per class"] = 2,
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
    output_dir: Annotated[str | None, "Optional explicit output directory"] = None,
) -> dict:
    """Start a Reinert classification on the matrix as a background job (returns job_id)."""

    def runner(job):
        job.phase = "CHD + profiles/AFC (R)"
        return matrixops.run_matrix_reinert(
            path,
            active_columns=active_columns,
            supplementary_columns=supplementary_columns,
            formatted=formatted,
            nbcl_p1=nbcl_p1,
            mincl=mincl,
            output_dir=output_dir,
            encoding=encoding,
            colsep=colsep,
        )

    return _submit(
        "matrix_reinert",
        {
            "path": path,
            "active_columns": active_columns,
            "formatted": formatted,
            "nbcl_p1": nbcl_p1,
            "mincl": mincl,
        },
        runner,
    )


@mcp.tool
async def matrix_similitude(
    path: Annotated[str, "Absolute path to the table"],
    columns: Annotated[list[int], "0-based column indices to include"],
    words: Annotated[list[str] | None, "Optional subset of modalities to plot"] = None,
    coeff: Annotated[int, "Similarity index (0=cooccurrence, 3=Jaccard, 12=Dice, ...)"] = 0,
    layout: Annotated[int, "Graph layout (0=random, 1=circle, 2=Fruchterman-Reingold, ...)"] = 2,
    seuil: Annotated[float, "Edge threshold"] = 1,
    arbremax: Annotated[bool, "Reduce the graph to a maximum spanning tree"] = True,
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
    output_dir: Annotated[str | None, "Optional explicit output directory"] = None,
) -> dict:
    """Start a similarity-graph analysis on matrix columns as a background job."""
    options = {"coeff": coeff, "layout": layout, "seuil": seuil, "arbremax": int(arbremax)}

    def runner(job):
        job.phase = "similarity matrix + graph (R)"
        return matrixops.run_matrix_similitude(
            path,
            columns=columns,
            words=words,
            options=options,
            output_dir=output_dir,
            encoding=encoding,
            colsep=colsep,
        )

    return _submit("matrix_similitude", {"path": path, "columns": columns, "options": options}, runner)


# ------------------------------------------------------------------------ jobs


@mcp.tool
async def job_status(job_id: Annotated[str, "Job id returned by a *_run tool"]) -> dict:
    """Poll a background analysis job (status, phase, elapsed, output dir)."""
    job = _job_manager().get(job_id)
    if job is None:
        return {"error": f"unknown job {job_id}", "jobs": _job_manager().list()}
    return job.summary()


@mcp.tool
async def job_result(job_id: Annotated[str, "Job id returned by a *_run tool"]) -> dict:
    """Fetch the parsed result of a finished job (or its error)."""
    try:
        return await asyncio.to_thread(_job_manager().result, job_id)
    except KeyError:
        return {"error": f"unknown job {job_id}"}


@mcp.tool
async def job_cancel(job_id: Annotated[str, "Job id returned by a *_run tool"]) -> dict:
    """Cancel a running job (kills its R process)."""
    try:
        return await asyncio.to_thread(_job_manager().cancel, job_id)
    except KeyError:
        return {"error": f"unknown job {job_id}"}


@mcp.tool
async def job_list() -> dict:
    """List all known jobs (including jobs from previous server runs)."""
    return {"jobs": _job_manager().list()}


# ------------------------------------------------------------------- reporting


@mcp.tool
async def analysis_read(
    path: Annotated[str, "Path to an analysis directory or its Analyse.ira"],
) -> dict:
    """Read any finished IRaMuTeQ analysis directory (parameters + parsed outputs)."""
    return await asyncio.to_thread(results.read_analysis, path)


@mcp.tool
async def analysis_list() -> dict:
    """List corpora/matrices and their analyses recorded in the IRaMuTeQ history."""
    return await asyncio.to_thread(results.list_history)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
