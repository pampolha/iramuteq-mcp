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
import threading
from typing import Annotated

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError

from .bridge import bootstrap, jobs, matrixops, results, subcorpus, textextra, textops

mcp = FastMCP("iramuteq")

ActiveKeys = Annotated[
    list[str] | None,
    "Grammatical categories forced to active (1) for this run, e.g. ['nom','ver','adj']",
]
SupplementaryKeys = Annotated[
    list[str] | None,
    "Grammatical categories forced to supplementary (2) for this run, e.g. ['adv','pro_per']",
]
SegmentationMethod = Annotated[
    int | None,
    "UCE segmentation as in corpus_build: 0=chars, 1=words, 2=paragraphs (None = corpus.cfg default)",
]
SegmentationSize = Annotated[
    int | None,
    "Target UCE size (chars or words depending on ucemethod); None = corpus.cfg default",
]

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


_JOB_MANAGER: jobs.JobManager | None = None
_JOB_MANAGER_LOCK = threading.Lock()


def _job_manager() -> jobs.JobManager:
    """The process-wide job registry, created on first use under the IRaMuTeQ user dir."""
    global _JOB_MANAGER
    if _JOB_MANAGER is None:
        with _JOB_MANAGER_LOCK:
            if _JOB_MANAGER is None:
                ctx = bootstrap.get()
                _JOB_MANAGER = jobs.JobManager(os.path.join(ctx.UserConfigPath, "mcp-jobs"))
    return _JOB_MANAGER


def _locked(fn, *args, **kwargs):
    # the IRaMuTeQ sources are only importable after bootstrap (sys.path + wx stub),
    # so initialise it here: bridge functions import those modules at call time and
    # may be the first thing a fresh server process ever runs.
    bootstrap.get()
    with jobs.analysis_lock():
        return fn(*args, **kwargs)


async def _sync(fn, *args, **kwargs):
    """Run one bridge function in a thread while holding the analysis lock."""
    return await asyncio.to_thread(_locked, fn, *args, **kwargs)


async def _submit(kind: str, params: dict, runner) -> dict:
    manager = await asyncio.to_thread(_job_manager)
    return manager.submit(kind, params, runner).summary()


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
                "pkgs <- c(%s); "
                'cat(paste(pkgs[pkgs %%in%% rownames(installed.packages())], collapse=","))'
                % ",".join('"%s"' % p for p in R_PACKAGES),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        installed = {p for p in proc.stdout.strip().split(",") if p} if proc.returncode == 0 else set()
        missing = [p for p in R_PACKAGES if p not in installed]
    except Exception as exc:
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
    """Validate ALCESTE corpus formatting (**** headers, -* themes, *variables).

    Header format: each text starts with a '****' line carrying its variables
    as '*variable_modality' tokens on that same line, e.g. '**** *sexe_h *age_20'
    (underscore, no spaces, no '='). The '*sexe=h' form is rejected because
    IRaMuTeQ parses only '*variable_modality' and silently drops the rest, which
    leaves downstream analyses without variables. Themes are '-*theme' lines
    inside the text.
    """
    return await _sync(textops.validate_corpus, path, encoding)


@mcp.tool
async def corpus_build(
    path: Annotated[str, "Absolute path to the ALCESTE-formatted .txt corpus"],
    lang: Annotated[str, "Corpus language for Lexique 3 (french, english, ...)"],
    encoding: Annotated[str, "File encoding"] = "utf8",
    ucemethod: Annotated[int, "UCE segmentation: 0=chars, 1=words, 2=paragraphs"] = 1,
    ucesize: Annotated[int, "Target UCE size (chars or words depending on ucemethod)"] = 40,
) -> dict:
    """Build (and cache) the IRaMuTeQ corpus: parse, clean, lemmatise, segment UCEs.

    Expects ALCESTE headers: each text starts with a '****' line carrying its
    variables as '*variable_modality' tokens on that same line, e.g.
    '**** *sexe_h *age_20' (underscore, no spaces, no '='), with themes as
    '-*theme' lines inside the text. Run corpus_validate first: headers like
    '*sexe=h' build fine but are ignored, leaving no variables for
    corpus_lexico and the other variable-based analyses.
    """
    options = {"ucemethod": ucemethod, "ucesize": ucesize}

    def run():
        _, summary = textops.build_corpus(path, lang=lang, encoding=encoding, options=options)
        return summary

    return await _sync(run)


@mcp.tool
async def corpus_stats(
    path: Annotated[str, "Absolute path to the corpus (built on first use)"],
    lang: Annotated[str, "Corpus language"],
    encoding: Annotated[str, "File encoding"] = "utf8",
    lem: Annotated[bool, "Lemmatise (verbs to infinitive, nouns to masc. sing.)"] = True,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
) -> dict:
    """Textual statistics: frequencies, active/supplementary forms, hapax, abstract."""
    return await _sync(
        textops.run_stats, path, lang, encoding, lem, active_keys, supplementary_keys, ucemethod, ucesize
    )


@mcp.tool
async def corpus_concordance(
    path: Annotated[str, "Absolute path to the corpus"],
    lem: Annotated[str, "Lemma or form to search (see corpus_stats for the list)"],
    lang: Annotated[str, "Corpus language"],
    limit: Annotated[int, "Maximum number of hits"] = 50,
    width: Annotated[int, "Context characters on each side"] = 60,
    encoding: Annotated[str, "File encoding"] = "utf8",
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
) -> dict:
    """Keyword-in-context concordance for a lemma, with UCI/variables of each hit."""

    def run():
        corpus, _ = textops.build_corpus(
            path, lang=lang, encoding=encoding, options={"ucemethod": ucemethod, "ucesize": ucesize}
        )
        return textops.concordance(corpus, lem, limit=limit, width=width)

    return await _sync(run)


# --------------------------------------------------------------- text analyses


def _alceste_job(
    path,
    lang,
    encoding,
    lem,
    output_dir,
    options,
    active_keys=None,
    supplementary_keys=None,
    ucemethod=None,
    ucesize=None,
):
    def runner(job):
        job.phase = "corpus + CHD + profiles/AFC (R)"
        res = textops.run_alceste(
            path,
            lang=lang,
            encoding=encoding,
            lem=lem,
            output_dir=output_dir,
            options=options,
            active_keys=active_keys,
            supplementary_keys=supplementary_keys,
            ucemethod=ucemethod,
            ucesize=ucesize,
        )
        job.phase = "done"
        return res

    return _submit(
        "corpus_alceste",
        {
            "path": path,
            "lang": lang,
            "encoding": encoding,
            "lem": lem,
            "options": options or {},
            "ucemethod": ucemethod,
            "ucesize": ucesize,
        },
        runner,
    )


@mcp.tool
async def corpus_alceste(
    path: Annotated[str, "Absolute path to the corpus"],
    lang: Annotated[str, "Corpus language"],
    classif_mode: Annotated[int, "0=double on UC, 1=simple on UCE, 2=simple on UCI"] = 1,
    nbcl_p1: Annotated[int, "Terminal classes of classification phase 1"] = 10,
    mincl: Annotated[int, "Minimum UCEs per class (0 = automatic)"] = 0,
    max_actives: Annotated[int, "Maximum number of active forms analysed (memory!)"] = 20000,
    tailleuc1: Annotated[int, "Target size of UC table 1 (active occurrences)"] = 12,
    tailleuc2: Annotated[int, "Target size of UC table 2 (active occurrences)"] = 14,
    svdmethod: Annotated[str, "SVD method: svdR, irlba or svdlibc"] = "irlba",
    lem: Annotated[bool, "Lemmatise"] = True,
    encoding: Annotated[str, "File encoding"] = "utf8",
    output_dir: Annotated[str | None, "Optional explicit output directory"] = None,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
) -> dict:
    """Start an ALCESTE/Reinert classification as a background job (returns job_id)."""
    try:
        textops.validate_max_actives(max_actives)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
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
    return await _alceste_job(
        path, lang, encoding, lem, output_dir, options, active_keys, supplementary_keys, ucemethod, ucesize
    )


@mcp.tool
async def corpus_lexico(
    path: Annotated[str, "Absolute path to the corpus"],
    lang: Annotated[str, "Corpus language"],
    modalities: Annotated[
        list[str] | None,
        "variable_modality entries to compare; must share ONE variable (empty = first variable's modalities)",
    ] = None,
    mineff: Annotated[int, "Minimum frequency of analysed forms"] = 10,
    indice: Annotated[str, "Specificity index: hypergeo or chi2"] = "hypergeo",
    lem: Annotated[bool, "Lemmatise"] = True,
    encoding: Annotated[str, "File encoding"] = "utf8",
    output_dir: Annotated[str | None, "Optional explicit output directory"] = None,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
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
            active_keys=active_keys,
            supplementary_keys=supplementary_keys,
            ucemethod=ucemethod,
            ucesize=ucesize,
        )

    return await _submit(
        "corpus_lexico",
        {"path": path, "modalities": modalities, "options": options, "ucemethod": ucemethod, "ucesize": ucesize},
        runner,
    )


@mcp.tool
async def corpus_similitude(
    path: Annotated[str, "Absolute path to the corpus"],
    lang: Annotated[str, "Corpus language"],
    forms: Annotated[list[str] | None, "Lemma subset to plot (empty = all active forms)"] = None,
    coeff: Annotated[int, "Similarity index (0=cooccurrence, 3=Jaccard, 12=Dice, ...)"] = 0,
    layout: Annotated[int, "Graph layout (0=random, 1=circle, 2=Fruchterman-Reingold, ...)"] = 2,
    seuil: Annotated[float, "Edge threshold"] = 1,
    arbremax: Annotated[bool, "Reduce the graph to a maximum spanning tree"] = True,
    lem: Annotated[bool, "Lemmatise"] = True,
    encoding: Annotated[str, "File encoding"] = "utf8",
    output_dir: Annotated[str | None, "Optional explicit output directory"] = None,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
) -> dict:
    """Start a similarity-graph analysis between active forms as a background job."""
    options = {"coeff": coeff, "layout": layout, "seuil": seuil, "arbremax": int(arbremax)}

    def runner(job):
        job.phase = "similarity matrix + graph (R)"
        return textops.run_similitude(
            path,
            forms=forms,
            lang=lang,
            encoding=encoding,
            lem=lem,
            output_dir=output_dir,
            options=options,
            active_keys=active_keys,
            supplementary_keys=supplementary_keys,
            ucemethod=ucemethod,
            ucesize=ucesize,
        )

    return await _submit(
        "corpus_similitude",
        {"path": path, "forms": forms, "options": options, "ucemethod": ucemethod, "ucesize": ucesize},
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

    return await _sync(run)


@mcp.tool
async def matrix_frequencies(
    path: Annotated[str, "Absolute path to the table"],
    columns: Annotated[list[int], "0-based column indices to tabulate"],
    include_na: Annotated[bool, "Count NA values"] = False,
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
) -> dict:
    """Frequency table + barplot per selected column (IRaMuTeQ Frequences)."""
    return await _sync(
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
    return await _sync(
        matrixops.run_chi2,
        path,
        columns_a,
        columns_b,
        None,
        encoding=encoding,
        colsep=colsep,
    )


@mcp.tool
async def matrix_mcnemar(
    path: Annotated[str, "Absolute path to the table"],
    columns_a: Annotated[list[int], "0-based column indices (first variable set)"],
    columns_b: Annotated[list[int], "0-based column indices (second variable set)"],
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
) -> dict:
    """McNemar tests for paired categorical columns (IRaMuTeQ McNemar)."""
    return await _sync(matrixops.run_mcnemar, path, columns_a, columns_b, None, encoding=encoding, colsep=colsep)


@mcp.tool
async def matrix_freqmulti(
    path: Annotated[str, "Absolute path to the table"],
    columns: Annotated[list[int], "0-based columns whose cells may hold several values"],
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
) -> dict:
    """Frequencies of multi-valued columns (IRaMuTeQ FreqMultiple)."""
    return await _sync(matrixops.run_freqmulti, path, columns, encoding=encoding, colsep=colsep)


@mcp.tool
async def matrix_splitvar(
    path: Annotated[str, "Absolute path to the table"],
    column: Annotated[int, "0-based column whose modalities split the table"],
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
) -> dict:
    """Split the table into one matrix per modality of a column (IRaMuTeQ SplitMatrixFromVar)."""
    return await _sync(matrixops.run_splitvar, path, column, encoding=encoding, colsep=colsep)


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
    if not formatted and not active_columns:
        raise ToolError("active_columns is required (or use formatted=True)")
    if not formatted and supplementary_columns is not None and not supplementary_columns:
        raise ToolError("supplementary_columns is empty; omit it to default to the remaining columns")

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

    return await _submit(
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

    return await _submit("matrix_similitude", {"path": path, "columns": columns, "options": options}, runner)


# ------------------------------------------------------- corpus utilities (A4)


@mcp.tool
async def corpus_keys() -> dict:
    """Current grammatical-category keys (key.cfg): active, supplementary or eliminated."""
    return await _sync(textops.key_summary)


@mcp.tool
async def corpus_split_by_variable(
    path: Annotated[str, "Absolute path to the corpus"],
    variable: Annotated[str, "Variable name without star, e.g. 'sexe'"],
    encoding: Annotated[str, "File encoding"] = "utf8",
) -> dict:
    """Split the corpus into one file per modality of a *variable (written beside the input)."""
    return await _sync(subcorpus.split_by_variable, path, variable, encoding)


@mcp.tool
async def corpus_extract_themes(
    path: Annotated[str, "Absolute path to the corpus"],
    themes: Annotated[list[str], "Themes to keep, e.g. ['travail'] (with or without '-*')"],
    encoding: Annotated[str, "File encoding"] = "utf8",
) -> dict:
    """Keep only the listed -* themes of the corpus, written as one combined file."""
    return await _sync(subcorpus.extract_themes, path, themes, encoding)


@mcp.tool
async def corpus_extract_mods(
    path: Annotated[str, "Absolute path to the corpus"],
    mods: Annotated[list[str], "Modalities to extract, e.g. ['*sexe_h']"],
    onefile: Annotated[bool, "One combined file (True) or one file per modality (False)"] = False,
    encoding: Annotated[str, "File encoding"] = "utf8",
) -> dict:
    """Extract the texts carrying the given *modalities (written beside the input)."""
    return await _sync(subcorpus.extract_mods, path, mods, onefile, encoding)


@mcp.tool
async def corpus_export_class(
    path: Annotated[str, "Absolute path to the corpus the ALCESTE run used"],
    analysis_dir: Annotated[str, "Output directory of a finished corpus_alceste run"],
    class_id: Annotated[int, "ALCESTE class number (1-based)"],
    lang: Annotated[str, "Corpus language (must match the ALCESTE run)"],
    encoding: Annotated[str, "File encoding (must match the ALCESTE run)"] = "utf8",
    lem: Annotated[bool, "Write lemmas instead of forms"] = False,
    uci: Annotated[bool, "Export whole UCIs instead of UCEs"] = False,
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
) -> dict:
    """Export one ALCESTE class as a new corpus file (for follow-up analyses)."""
    return await _sync(
        subcorpus.export_class,
        path,
        analysis_dir,
        class_id,
        lang,
        encoding,
        lem,
        uci,
        ucemethod=ucemethod,
        ucesize=ucesize,
    )


# --------------------------------------------- text analyses, part 2 (B2)


@mcp.tool
async def corpus_labbe(
    path: Annotated[str, "Absolute path to the corpus"],
    lang: Annotated[str, "Corpus language"],
    modalities: Annotated[list[str] | None, "Modalities of ONE variable (empty = first variable)"] = None,
    mineff: Annotated[int, "Minimum frequency of analysed forms"] = 10,
    indice: Annotated[str, "Specificity index: hypergeo or chi2"] = "hypergeo",
    lem: Annotated[bool, "Lemmatise"] = True,
    encoding: Annotated[str, "File encoding"] = "utf8",
    output_dir: Annotated[str | None, "Optional explicit output directory"] = None,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
) -> dict:
    """Start a Labbé lexical-distance analysis between modalities as a background job."""
    if indice not in ("hypergeo", "chi2"):
        raise ToolError("indice must be 'hypergeo' or 'chi2'")
    params = {
        "path": path,
        "modalities": modalities,
        "mineff": mineff,
        "indice": indice,
        "ucemethod": ucemethod,
        "ucesize": ucesize,
    }

    def runner(job):
        job.phase = "Labbé distances + clustering (R)"
        return textextra.run_labbe(
            path,
            modalities=modalities,
            mineff=mineff,
            indice=indice,
            lang=lang,
            encoding=encoding,
            lem=lem,
            output_dir=output_dir,
            active_keys=active_keys,
            supplementary_keys=supplementary_keys,
            ucemethod=ucemethod,
            ucesize=ucesize,
        )

    return await _submit("corpus_labbe", params, runner)


@mcp.tool
async def corpus_wordcloud(
    path: Annotated[str, "Absolute path to the corpus"],
    lang: Annotated[str, "Corpus language"],
    mode: Annotated[int, "0 = active forms, 1 = supplementary forms, 2 = both"] = 0,
    forms: Annotated[
        list[str] | None, "Optional subset of lemmas to draw (must be active/supplementary per mode)"
    ] = None,
    maxword: Annotated[int, "Maximum number of words drawn"] = 100,
    width: Annotated[int, "Image width (px)"] = 800,
    height: Annotated[int, "Image height (px)"] = 800,
    mincex: Annotated[float, "Smallest relative font size"] = 0.3,
    maxcex: Annotated[float, "Largest relative font size"] = 3.0,
    col_text: Annotated[list[int] | None, "Text colour as [R,G,B]"] = None,
    col_bg: Annotated[list[int] | None, "Background colour as [R,G,B]"] = None,
    svg: Annotated[bool, "SVG instead of PNG"] = False,
    encoding: Annotated[str, "File encoding"] = "utf8",
    lem: Annotated[bool, "Lemmatise"] = True,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
) -> dict:
    """Word cloud of the corpus forms (IRaMuTeQ WordCloud)."""
    options = {
        "maxword": maxword,
        "width": width,
        "height": height,
        "mincex": mincex,
        "maxcex": maxcex,
        "col_text": col_text,
        "col_bg": col_bg,
        "svg": svg,
    }
    return await _sync(
        textextra.run_wordcloud,
        path,
        mode,
        forms,
        options,
        lang,
        encoding,
        lem,
        None,
        active_keys,
        supplementary_keys,
        ucemethod,
        ucesize,
    )


@mcp.tool
async def corpus_class_cloud(
    path: Annotated[str, "Absolute path to the corpus the ALCESTE run used"],
    analysis_dir: Annotated[str, "Output directory of a finished corpus_alceste run"],
    class_id: Annotated[int, "ALCESTE class number (1-based)"],
    lang: Annotated[str, "Corpus language (must match the ALCESTE run)"],
    indice: Annotated[int, "Weight: 0 = class frequency, 1 = chi-square with the class"] = 0,
    maxword: Annotated[int, "Maximum number of words drawn"] = 100,
    width: Annotated[int, "Image width (px)"] = 800,
    height: Annotated[int, "Image height (px)"] = 800,
    svg: Annotated[bool, "SVG instead of PNG"] = False,
    encoding: Annotated[str, "File encoding (must match the ALCESTE run)"] = "utf8",
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
) -> dict:
    """Word cloud of one ALCESTE class, sized by its profile (IRaMuTeQ ClusterCloud)."""
    options = {"maxword": maxword, "width": width, "height": height, "svg": svg}
    return await _sync(
        textextra.run_class_cloud,
        path,
        analysis_dir,
        class_id,
        indice,
        options,
        lang,
        encoding,
        ucemethod=ucemethod,
        ucesize=ucesize,
    )


@mcp.tool
async def matrix_proto(
    path: Annotated[str, "Absolute path to the table"],
    words_columns: Annotated[list[int], "0-based columns holding the words"],
    rank_columns: Annotated[list[int], "0-based columns holding the paired ranks (same length)"],
    freqmin: Annotated[int, "Minimum frequency of a word"] = 2,
    limfreq: Annotated[int | None, "Frequency limit for the plot (omit = no limit)"] = None,
    limrang: Annotated[int | None, "Rank limit for the plot (omit = no limit)"] = None,
    typegraph: Annotated[str, "'classical' or 'plan'"] = "classical",
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
) -> dict:
    """Prototypical analysis: words characterised by frequency and mean rank (IRaMuTeQ)."""
    return await _sync(
        matrixops.run_proto,
        path,
        words_columns,
        rank_columns,
        freqmin,
        limfreq,
        limrang,
        typegraph,
        encoding,
        colsep,
    )


@mcp.tool
async def corpus_tgen_spec(
    path: Annotated[str, "Absolute path to the corpus"],
    tgen_path: Annotated[str, "Tab-separated file: one line per group, 'group<TAB>lemma<TAB>lemma...'"],
    lang: Annotated[str, "Corpus language"],
    modalities: Annotated[list[str] | None, "*variable_modality entries (empty = all)"] = None,
    encoding: Annotated[str, "File encoding"] = "utf8",
    lem: Annotated[bool, "Lemmatise"] = True,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
) -> dict:
    """Specificity of Tgen groups (usual segments) across modalities (IRaMuTeQ TgenSpec)."""
    return await _sync(
        textextra.run_tgen_spec,
        path,
        tgen_path,
        modalities,
        lang,
        encoding,
        lem,
        None,
        active_keys,
        supplementary_keys,
        ucemethod,
        ucesize,
    )


@mcp.tool
async def corpus_tgen_prof(
    path: Annotated[str, "Absolute path to the corpus the ALCESTE run used"],
    analysis_dir: Annotated[str, "Output directory of a finished corpus_alceste run"],
    tgen_path: Annotated[str, "Tab-separated Tgen file: 'group<TAB>lemma<TAB>lemma...'"],
    lang: Annotated[str, "Corpus language (must match the ALCESTE run)"],
    encoding: Annotated[str, "File encoding (must match the ALCESTE run)"] = "utf8",
    lem: Annotated[bool, "Lemmatise"] = True,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
    ucemethod: SegmentationMethod = None,
    ucesize: SegmentationSize = None,
) -> dict:
    """Start a Tgen profile over the classes of an ALCESTE run as a background job."""
    params = {
        "path": path,
        "analysis_dir": analysis_dir,
        "tgen_path": tgen_path,
        "ucemethod": ucemethod,
        "ucesize": ucesize,
    }

    def runner(job):
        job.phase = "Tgen profiles (R)"
        return textextra.run_tgen_prof(
            analysis_dir,
            path,
            tgen_path,
            lang=lang,
            encoding=encoding,
            lem=lem,
            active_keys=active_keys,
            supplementary_keys=supplementary_keys,
            ucemethod=ucemethod,
            ucesize=ucesize,
        )

    return await _submit("corpus_tgen_prof", params, runner)


# ------------------------------------------------------------------------ jobs


def _unknown_job(job_id: str) -> ToolError:
    return ToolError(f"unknown job id: {job_id}. Use job_list to see known jobs.")


@mcp.tool
async def job_status(
    job_id: Annotated[str, "Job id returned by a *_run tool"],
    ctx: Context | None = None,
) -> dict:
    """Poll a background analysis job: status, phase, elapsed time, output dir."""
    manager = await asyncio.to_thread(_job_manager)
    job = manager.get(job_id)
    if job is None:
        raise _unknown_job(job_id)
    if ctx is not None and job.status in jobs.ACTIVE:
        # sent only when the client supplied a progress token
        await ctx.report_progress(progress=job.elapsed, message=job.phase or job.status)
    return job.summary()


@mcp.tool
async def job_result(job_id: Annotated[str, "Job id returned by a *_run tool"]) -> dict:
    """Fetch a job: its status, and the parsed result once it is done."""
    res = await asyncio.to_thread(lambda: _job_manager().result(job_id))
    if res is None:
        raise _unknown_job(job_id)
    return res


@mcp.tool
async def job_cancel(job_id: Annotated[str, "Job id returned by a *_run tool"]) -> dict:
    """Cancel a job (best-effort): a running R process is terminated immediately.

    A cancel that lands during a Python phase (corpus build, lemmatisation)
    or between two R scripts only takes effect when that phase ends, so the
    job may still finish and report ``done``.  Finished jobs are returned
    unchanged.
    """
    res = await asyncio.to_thread(lambda: _job_manager().cancel(job_id))
    if res is None:
        raise _unknown_job(job_id)
    return res


@mcp.tool
async def job_list() -> dict:
    """List all known jobs, including those from previous server runs."""
    manager = await asyncio.to_thread(_job_manager)
    return {"jobs": manager.list()}


@mcp.tool
async def job_prune(
    job_ids: Annotated[list[str] | None, "Specific finished jobs to delete (omit for all matching)"] = None,
    statuses: Annotated[
        list[jobs.JobStatus] | None,
        "Statuses to prune: done, error, cancelled, interrupted (default: all finished)",
    ] = None,
    older_than_hours: Annotated[float | None, "Only prune jobs that ended at least this long ago"] = None,
) -> dict:
    """Delete finished job records and their stored results. Running jobs are never pruned."""
    allowed = set(jobs.FINISHED)
    chosen = list(statuses) if statuses else list(jobs.FINISHED)
    bad = sorted(str(s) for s in set(chosen) - allowed)
    if bad:
        raise ToolError(f"unknown statuses {bad}; choose from {sorted(str(s) for s in allowed)}")
    older = None if older_than_hours is None else older_than_hours * 3600
    return await asyncio.to_thread(lambda: _job_manager().prune(job_ids, chosen, older))


# ------------------------------------------------------------------- reporting


@mcp.tool
async def analysis_read(
    path: Annotated[str, "Path to an analysis directory or its Analyse.ira"],
) -> dict:
    """Read any finished IRaMuTeQ analysis directory (parameters + parsed outputs)."""
    return await _sync(results.read_analysis, path)


@mcp.tool
async def analysis_list() -> dict:
    """List corpora/matrices and their analyses recorded in the IRaMuTeQ history."""
    return await _sync(results.list_history)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
