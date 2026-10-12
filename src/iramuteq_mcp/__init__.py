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
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
) -> dict:
    """Textual statistics: frequencies, active/supplementary forms, hapax, abstract."""
    return await asyncio.to_thread(
        textops.run_stats, path, lang, encoding, lem, active_keys, supplementary_keys
    )


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


def _alceste_job(path, lang, encoding, lem, output_dir, options, active_keys=None, supplementary_keys=None):
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
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
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
    return _alceste_job(
        path, lang, encoding, lem, output_dir, options, active_keys, supplementary_keys
    )


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
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
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
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
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
            active_keys=active_keys,
            supplementary_keys=supplementary_keys,
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
async def matrix_mcnemar(
    path: Annotated[str, "Absolute path to the table"],
    columns_a: Annotated[list[int], "0-based column indices (first variable set)"],
    columns_b: Annotated[list[int], "0-based column indices (second variable set)"],
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
) -> dict:
    """McNemar tests for paired categorical columns (IRaMuTeQ McNemar)."""
    return await asyncio.to_thread(
        matrixops.run_mcnemar, path, columns_a, columns_b, None, encoding=encoding, colsep=colsep
    )


@mcp.tool
async def matrix_freqmulti(
    path: Annotated[str, "Absolute path to the table"],
    columns: Annotated[list[int], "0-based columns whose cells may hold several values"],
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
) -> dict:
    """Frequencies of multi-valued columns (IRaMuTeQ FreqMultiple)."""
    return await asyncio.to_thread(
        matrixops.run_freqmulti, path, columns, encoding=encoding, colsep=colsep
    )


@mcp.tool
async def matrix_splitvar(
    path: Annotated[str, "Absolute path to the table"],
    column: Annotated[int, "0-based column whose modalities split the table"],
    encoding: Annotated[str, "File encoding"] = "utf-8",
    colsep: Annotated[str, "Field separator for csv"] = ";",
) -> dict:
    """Split the table into one matrix per modality of a column (IRaMuTeQ SplitMatrixFromVar)."""
    return await asyncio.to_thread(
        matrixops.run_splitvar, path, column, encoding=encoding, colsep=colsep
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


# ------------------------------------------------------- corpus utilities (A4)


@mcp.tool
async def corpus_keys() -> dict:
    """Current grammatical-category keys (key.cfg): active, supplementary or eliminated."""
    return await asyncio.to_thread(textops.key_summary)


@mcp.tool
async def corpus_split_by_variable(
    path: Annotated[str, "Absolute path to the corpus"],
    variable: Annotated[str, "Variable name without star, e.g. 'sexe'"],
    encoding: Annotated[str, "File encoding"] = "utf8",
) -> dict:
    """Split the corpus into one file per modality of a *variable (written beside the input)."""
    return await asyncio.to_thread(subcorpus.split_by_variable, path, variable, encoding)


@mcp.tool
async def corpus_extract_themes(
    path: Annotated[str, "Absolute path to the corpus"],
    themes: Annotated[list[str], "Themes to keep, e.g. ['travail'] (with or without '-*')"],
    encoding: Annotated[str, "File encoding"] = "utf8",
) -> dict:
    """Keep only the listed -* themes of the corpus, written as one combined file."""
    return await asyncio.to_thread(subcorpus.extract_themes, path, themes, encoding)


@mcp.tool
async def corpus_extract_mods(
    path: Annotated[str, "Absolute path to the corpus"],
    mods: Annotated[list[str], "Modalities to extract, e.g. ['*sexe_h']"],
    onefile: Annotated[bool, "One combined file (True) or one file per modality (False)"] = False,
    encoding: Annotated[str, "File encoding"] = "utf8",
) -> dict:
    """Extract the texts carrying the given *modalities (written beside the input)."""
    return await asyncio.to_thread(subcorpus.extract_mods, path, mods, onefile, encoding)


@mcp.tool
async def corpus_export_class(
    path: Annotated[str, "Absolute path to the corpus the ALCESTE run used"],
    analysis_dir: Annotated[str, "Output directory of a finished corpus_alceste run"],
    class_id: Annotated[int, "ALCESTE class number (1-based)"],
    lang: Annotated[str, "Corpus language (must match the ALCESTE run)"] = "french",
    encoding: Annotated[str, "File encoding (must match the ALCESTE run)"] = "utf8",
    lem: Annotated[bool, "Write lemmas instead of forms"] = False,
    uci: Annotated[bool, "Export whole UCIs instead of UCEs"] = False,
) -> dict:
    """Export one ALCESTE class as a new corpus file (for follow-up analyses)."""
    return await asyncio.to_thread(
        subcorpus.export_class, path, analysis_dir, class_id, lang, encoding, lem, uci
    )


# --------------------------------------------- text analyses, part 2 (B2)


@mcp.tool
async def corpus_labbe(
    path: Annotated[str, "Absolute path to the corpus"],
    modalities: Annotated[list[str] | None, "Modalities of ONE variable (empty = first variable)"] = None,
    mineff: Annotated[int, "Minimum frequency of analysed forms"] = 10,
    indice: Annotated[str, "Specificity index: hypergeo or chi2"] = "hypergeo",
    lem: Annotated[bool, "Lemmatise"] = True,
    lang: Annotated[str, "Corpus language"] = "french",
    encoding: Annotated[str, "File encoding"] = "utf8",
    output_dir: Annotated[str | None, "Optional explicit output directory"] = None,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
) -> dict:
    """Start a Labbé lexical-distance analysis between modalities as a background job."""
    params = {"path": path, "modalities": modalities, "mineff": mineff, "indice": indice}

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
        )

    return _submit("corpus_labbe", params, runner)


@mcp.tool
async def corpus_wordcloud(
    path: Annotated[str, "Absolute path to the corpus"],
    mode: Annotated[int, "0 = active forms, 1 = supplementary forms, 2 = both"] = 0,
    forms: Annotated[list[str] | None, "Optional subset of lemmas to draw (must be active/supplementary per mode)"] = None,
    maxword: Annotated[int, "Maximum number of words drawn"] = 100,
    width: Annotated[int, "Image width (px)"] = 800,
    height: Annotated[int, "Image height (px)"] = 800,
    mincex: Annotated[float, "Smallest relative font size"] = 0.3,
    maxcex: Annotated[float, "Largest relative font size"] = 3.0,
    col_text: Annotated[list[int] | None, "Text colour as [R,G,B]"] = None,
    col_bg: Annotated[list[int] | None, "Background colour as [R,G,B]"] = None,
    svg: Annotated[bool, "SVG instead of PNG"] = False,
    lang: Annotated[str, "Corpus language"] = "french",
    encoding: Annotated[str, "File encoding"] = "utf8",
    lem: Annotated[bool, "Lemmatise"] = True,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
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
    return await asyncio.to_thread(
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
    )


@mcp.tool
async def corpus_class_cloud(
    path: Annotated[str, "Absolute path to the corpus the ALCESTE run used"],
    analysis_dir: Annotated[str, "Output directory of a finished corpus_alceste run"],
    class_id: Annotated[int, "ALCESTE class number (1-based)"],
    indice: Annotated[int, "Weight: 0 = class frequency, 1 = chi-square with the class"] = 0,
    maxword: Annotated[int, "Maximum number of words drawn"] = 100,
    width: Annotated[int, "Image width (px)"] = 800,
    height: Annotated[int, "Image height (px)"] = 800,
    svg: Annotated[bool, "SVG instead of PNG"] = False,
    lang: Annotated[str, "Corpus language (must match the ALCESTE run)"] = "french",
    encoding: Annotated[str, "File encoding (must match the ALCESTE run)"] = "utf8",
) -> dict:
    """Word cloud of one ALCESTE class, sized by its profile (IRaMuTeQ ClusterCloud)."""
    options = {"maxword": maxword, "width": width, "height": height, "svg": svg}
    return await asyncio.to_thread(
        textextra.run_class_cloud,
        path,
        analysis_dir,
        class_id,
        indice,
        options,
        lang,
        encoding,
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
    return await asyncio.to_thread(
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
    modalities: Annotated[list[str] | None, "*variable_modality entries (empty = all)"] = None,
    lang: Annotated[str, "Corpus language"] = "french",
    encoding: Annotated[str, "File encoding"] = "utf8",
    lem: Annotated[bool, "Lemmatise"] = True,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
) -> dict:
    """Specificity of Tgen groups (usual segments) across modalities (IRaMuTeQ TgenSpec)."""
    return await asyncio.to_thread(
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
    )


@mcp.tool
async def corpus_tgen_prof(
    path: Annotated[str, "Absolute path to the corpus the ALCESTE run used"],
    analysis_dir: Annotated[str, "Output directory of a finished corpus_alceste run"],
    tgen_path: Annotated[str, "Tab-separated Tgen file: 'group<TAB>lemma<TAB>lemma...'"],
    lang: Annotated[str, "Corpus language (must match the ALCESTE run)"] = "french",
    encoding: Annotated[str, "File encoding (must match the ALCESTE run)"] = "utf8",
    lem: Annotated[bool, "Lemmatise"] = True,
    active_keys: ActiveKeys = None,
    supplementary_keys: SupplementaryKeys = None,
) -> dict:
    """Start a Tgen profile over the classes of an ALCESTE run as a background job."""
    params = {"path": path, "analysis_dir": analysis_dir, "tgen_path": tgen_path}

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
        )

    return _submit("corpus_tgen_prof", params, runner)


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
