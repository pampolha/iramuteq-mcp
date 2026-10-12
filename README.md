# iramuteq-mcp

An [MCP](https://modelcontextprotocol.io) server for [IRaMuTeQ](http://www.iramuteq.org) —
the French R-based text-mining and corpus analysis suite (Pierre Ratinaud) —
built with [FastMCP](https://gofastmcp.com).

This server is a **substitute interface to the program itself**: it does not
reimplement any analysis. It imports the IRaMuTeQ 0.8 sources as a library,
drives the same Python entry points the wxPython GUI calls, runs the same
generated R scripts, and parses the same output files with the source's own
readers. The GUI layer is neutralised (wx is stubbed, dialogs are bypassed by
injecting the options they would have collected).

## Requirements

- Python ≥ 3.13 (`uv sync`)
- R with the IRaMuTeQ packages (`ca`, `gee`, `ape`, `igraph`, `proxy`,
  `wordcloud`, `irlba`, `textometry`, `sna`, `network`, `intergraph`, `rgl`,
  `cluster`, `MASS`, `Matrix`) — the `status` tool reports what is missing
- The IRaMuTeQ 0.8 source tree, pointed at by `IRAMUTEQ_PATH`
  (default `/home/pampolha/Tools/iramuteq-0.8alpha7-sources`)

User config is seeded in `~/.iramuteq-<version>/` (shared with the desktop
app if installed). Analyses write their usual `<corpus>_<type>_<N>` directories
beside the input file (or to an explicit `output_dir`).

## How it maps onto the IRaMuTeQ sources

| Layer | Source module | Bridge code |
|---|---|---|
| Corpus parse/clean/lemmatise/segment | `corpus.py::BuildFromAlceste`, `Corpus` | `bridge/textops.py` |
| ALCESTE classification | `textreinert.py::Reinert` → `PrintRScript.RchdTxt/ReinertTxtProf` → `Rscripts/CHD.R`, `chdtxt.R` | `bridge/textops.py` |
| Textual statistics | `textstat.py::Stat` | `bridge/textops.py` |
| Specificities ("Comme Lexico") | `textaslexico.py::Lexico` | `bridge/textops.py` |
| Similarity graphs (text) | `textsimi.py::SimiTxt` → `Rscripts/simi.R` | `bridge/textops.py` |
| Matrix analyses | `tableau.py::Tableau`, `tabfrequence/tabchi2/tabchdalc/tabsimi` | `bridge/matrixops.py` |
| Output parsing | `functions.ReadProfileAsDico`, `Corpus.make_ucecl_from_R`, `DoConf` | `bridge/results.py` |
| R execution | `functions.exec_rcode` (`R --slave --vanilla -f <wrapper>`) | tracked by `bridge/patches.py` |
| Headless glue | — | `bridge/wxstub.py`, `bridge/bootstrap.py`, `bridge/patches.py`, `bridge/jobs.py` |

## Tools

**Status & corpus**

| Tool | Description |
|---|---|
| `status` | IRaMuTeQ sources, config, R executable/version, missing R packages |
| `corpus_validate` | Validate ALCESTE formatting (`****` headers, `-*` themes, `*variables`) |
| `corpus_build` | Parse, clean, lemmatise and segment a `.txt` corpus (cached) |
| `corpus_stats` | Frequencies, active/supplementary forms, hapax, abstract |
| `corpus_concordance` | KWIC concordance for a lemma, with UCI variables |

**Text analyses** (long R runs return a `job_id`; poll `job_*`)

| Tool | Description |
|---|---|
| `corpus_alceste` | Reinert/ALCESTE descending hierarchical classification |
| `corpus_lexico` | Specificities of forms/types by `variable_modality` (one variable) |
| `corpus_similitude` | Similarity graph between active forms (27 indices, MST, layouts) |

**Matrix analyses** (`.csv`/`.xls`/`.ods`, individuals × variables)

| Tool | Description |
|---|---|
| `matrix_open` | Load a table (cached), report columns/rows |
| `matrix_frequencies` | Frequency table + barplot per column |
| `matrix_chi2` | Chi-square independence tests between two column sets |
| `matrix_reinert` | Reinert classification on rows (job) |
| `matrix_similitude` | Similarity graph on modalities (job) |

**Jobs & results**

| Tool | Description |
|---|---|
| `job_status` / `job_result` / `job_cancel` / `job_list` | Poll, fetch, cancel background analyses |
| `analysis_read` | Parse any finished analysis directory (`Analyse.ira` + outputs) |
| `analysis_list` | Corpora/matrices and analyses recorded in the IRaMuTeQ history |

Every result carries the parsed tables (classes, profiles, chi², …) and the
absolute paths of the artifacts IRaMuTeQ produced (PNG graphs, CSVs,
`RData.RData`, `Analyse.ira`) so the client can open what the GUI would show.

## Known quirks (faithful to the sources)

- Lexico modalities must belong to a **single** `*variable` (e.g. `*sexe_h`
  vs `*sexe_f`); the source rejects mixed variables.
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
uv sync
uv run python tests/smoke.py      # bootstrap -> corpus -> stats -> alceste
uv run python tests/test_flows.py # job API + all analyses end-to-end
uv run python tests/test_client.py# FastMCP tool registration + job cancel
uv run iramuteq-mcp               # start the MCP server (stdio)
```

## License

GPL-2.0 (see `LICENSE`). This project drives and derives from IRaMuTeQ
(`Copyright (c) 2008-2024 Pierre Ratinaud`, GNU GPL v2); bundling the sources
with this server is permitted under those terms, with attribution and source
availability. The bundled Lexique 3 dictionaries are permissively licensed.
