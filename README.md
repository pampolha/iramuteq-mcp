# iramuteq-mcp

An [MCP](https://modelcontextprotocol.io) server for [IRaMuTeQ](http://www.iramuteq.org) —
the French R-based text-mining and corpus analysis suite (Pierre Ratinaud) —
built with [FastMCP](https://gofastmcp.com).

This server imports the IRaMuTeQ 0.8 sources as a library,
drives the Python entry points the wxPython GUI calls,
runs the generated R scripts, and parses the same output files with the source's own readers.
The GUI layer is neutralised.

## Requirements

- Python ≥ 3.13 (`uv sync`)
- R with the IRaMuTeQ packages (`ca`, `gee`, `ape`, `igraph`, `proxy`,
  `wordcloud`, `irlba`, `textometry`, `sna`, `network`, `intergraph`, `rgl`,
  `cluster`, `MASS`, `Matrix`)

## Tools

Long R runs (ALCESTE, Labbé, Tgen profiles, similarity) return a `job_id`
immediately; poll with `job_status` / `job_result`. Other tools return the
parsed result directly. `job_cancel` is best-effort: it terminates a running
R process immediately, but a cancel landing during a Python phase (corpus
build, lemmatisation) or between two R scripts only takes effect when that
phase ends, so the job may still finish and report `done`.

**Status & corpus**

| Tool | Description |
|---|---|
| `status` | IRaMuTeQ sources, config, R executable/version, missing R packages |
| `corpus_validate` | Validate ALCESTE formatting (`****` headers, `-*` themes, `*variables`) |
| `corpus_build` | Parse, clean, lemmatise and segment a `.txt` corpus (cached) |
| `corpus_keys` | Current grammatical-category keys (`key.cfg`): active / supplementary / eliminated |
| `corpus_stats` | Frequencies, active/supplementary forms, hapax, abstract |
| `corpus_concordance` | KWIC concordance for a lemma, with UCI variables |

**Text analyses**

| Tool | Description |
|---|---|
| `corpus_alceste` *(job)* | Reinert/ALCESTE descending hierarchical classification |
| `corpus_lexico` *(job)* | Specificities of forms/types by one `*variable`'s modalities |
| `corpus_labbe` *(job)* | Labbé lexical distance between the modalities of one variable |
| `corpus_similitude` *(job)* | Similarity graph between active forms (27 indices, MST, layouts) |
| `corpus_wordcloud` | Word cloud of active / supplementary / both forms, optional subset |
| `corpus_class_cloud` | Word cloud of one ALCESTE class, weighted by frequency or chi² |
| `corpus_tgen_spec` | Specificity of Tgen groups (usual segments) across modalities |
| `corpus_tgen_prof` *(job)* | Tgen group profiles over the classes of an ALCESTE run |

**Sub-corpora & classes**

| Tool | Description |
|---|---|
| `corpus_split_by_variable` | One file per modality of a `*variable` (beside the input) |
| `corpus_extract_themes` | Keep only the listed `-*` themes (beside the input) |
| `corpus_extract_mods` | Extract texts carrying given `*modalities` (beside the input) |
| `corpus_export_class` | Export one ALCESTE class as a new corpus file |

**Matrix analyses** (`.csv`/`.xls`/`.ods`, individuals × variables)

| Tool | Description |
|---|---|
| `matrix_open` | Load a table (cached), report columns/rows |
| `matrix_frequencies` | Frequency table + barplot per column |
| `matrix_freqmulti` | Frequencies of multi-valued columns |
| `matrix_chi2` | Chi-square independence tests between two column sets |
| `matrix_mcnemar` | McNemar tests for paired categorical columns |
| `matrix_splitvar` | One matrix per modality of a column |
| `matrix_proto` | Prototypical analysis: words characterised by frequency and mean rank |
| `matrix_reinert` *(job)* | Reinert classification on rows |
| `matrix_similitude` *(job)* | Similarity graph on modalities |

**Jobs & results**

| Tool | Description |
|---|---|
| `job_status` / `job_result` / `job_cancel` / `job_list` | Poll, fetch, cancel background analyses |
| `analysis_read` | Parse any finished analysis directory (`Analyse.ira` + outputs) |
| `analysis_list` | Corpora/matrices and analyses recorded in the IRaMuTeQ history |

### Common options

- **Analysis keys**: every corpus analysis accepts `active_keys` /
  `supplementary_keys` (e.g. `["nom","ver","adj"]`, `["adv","pro_per"]`). They
  override `key.cfg` for that run only.
- **Segmentation**: `corpus_build`'s `ucemethod` / `ucesize` are accepted by
  every tool that reads a corpus (`corpus_stats`, `corpus_alceste`,
  `corpus_concordance`, ...). Pass the values you built with, otherwise the
  tool resolves `corpus.cfg`'s defaults, misses the cache and rebuilds the
  corpus into a second `<corpus>_corpus_N` directory.
- **Modalities** (`corpus_lexico`, `corpus_labbe`): must belong to one
  `*variable`; if omitted, the first variable's modalities are used. Tgen
  analyses accept modalities from any variables.
- **Tgen file**: tab-separated, one group per line:
  `group<TAB>lemma<TAB>lemma…`. Lemmas absent from the corpus are dropped and
  reported in the result.
- **Where files go**: analyses write their usual `<corpus>_<type>_<N>` (or
  `<matrix>_<type>_<N>`) directories; the sub-corpus tools write beside their
  input file, as the GUI does. Results list the artifact paths.

## Known issues

- Lexico and Labbé modalities must belong to a **single** `*variable` (e.g.
  `*sexe_h` vs `*sexe_f`); the source rejects mixed variables.
- `corpus_extract_themes` writes its output without a file extension — that
  is the source's naming.
- `matrix_reinert` needs ≥ 3 phase-1 classes (`nbcl_p1 >= 3`); `nbcl_p1 = 2`
  hits a dimension-drop bug in `chdquest.R::makepoids`.
- A similarity run on modalities that never co-occur produces an empty graph
  and crashes in `simi.R`'s maximum-spanning-tree step — give it columns with
  co-occurring modalities.
- Very small matrices (< ~30 rows) can break CHD's internal splits; use real
  data sizes.
- Analyses are serialised (one at a time): they share corpus sqlite handles
  and R is memory-hungry. `max_actives` drives table size — lower it if R runs
  out of memory.

## Development

```bash
uv sync --group dev                 # install the project plus pytest, ruff, mypy
uv run pytest                       # full suite
uv run pytest -m "not requires_r"   # unit tests only (job lifecycle, bounds, validation)
uv run pytest -m "not slow"         # skip the mid-run cancellation test
uv run ruff check src tests         # lint
uv run ruff format src tests        # format
uv run mypy src/iramuteq_mcp        # type check
uv run iramuteq-mcp                 # start the MCP server (stdio)
```

Tests that need R and the IRaMuTeQ packages are marked `requires_r`. When R or
a package is missing they are **skipped** with the reason. The
fixtures are generated or copied into temporary directories.

Job records are kept until they are cancelled or pruned explicitly
(`job_prune`).

## License

GPL-2.0 (see `LICENSE`). This project drives and derives from IRaMuTeQ
(`Copyright (c) 2008-2024 Pierre Ratinaud`, GNU GPL v2).
