# iramuteq-mcp

An [MCP](https://modelcontextprotocol.io) server for [IRaMuTeQ](http://www.iramuteq.org) —
the French R-based text-mining and corpus analysis suite — built with
[FastMCP](https://gofastmcp.com).

IRaMuTeQ 0.8 is a wxPython desktop application. This server exposes its corpus
format and R analysis layer **headlessly**, so any MCP client (Claude Desktop,
OpenWork, VS Code, …) can analyse IRaMuTeQ corpora without the GUI.

## Tools

| Tool | Description |
|---|---|
| `corpus_info` | Parse a corpus: documents (UCIs), paragraphs, tokens, forms, hapax, `****` header variables |
| `word_frequencies` | Most frequent forms with percentages |
| `concordance` | Keyword-in-context search (word or regex) with document context |
| `variable_distribution` | Distribution of a `*variable` across documents |
| `make_subcorpus` | Extract documents matching a variable value into a new corpus file |
| `check_r_packages` | Check the R packages IRaMuTeQ needs (`ca`, `ape`, `igraph`, …) |
| `list_r_scripts` | List IRaMuTeQ's bundled R analysis scripts |
| `run_r_script` | Run a bundled R script via `Rscript` and return stdout/stderr |

## Corpus format

```
**** *year 2020 *author Doe
-* para
Le chat dort sur le canapé.
**** *year 2021 *author Roy
La mer est calme ce matin.
```

- `**** …` starts a document (UCI) and carries its variables
- `-* …` marks a paragraph
- other non-empty lines are text

## Setup

Requirements:

- Python ≥ 3.13 ([uv](https://docs.astral.sh/uv/) recommended)
- R (≥ 4.x) with the packages listed in `check_r_packages`
- IRaMuTeQ 0.8 sources (default path: `/home/pampolha/Tools/iramuteq-0.8alpha7-sources`,
  override with the `IRAMUTEQ_PATH` environment variable)

```sh
uv sync
uv run iramuteq-mcp        # stdio transport
```

### Client configuration

```json
{
  "mcpServers": {
    "iramuteq": {
      "command": "uv",
      "args": ["--project", "/path/to/iramuteq-mcp", "run", "iramuteq-mcp"],
      "env": { "IRAMUTEQ_PATH": "/path/to/iramuteq-0.8alpha7-sources" }
    }
  }
}
```

## License

GPL-2.0-or-later, matching IRaMuTeQ. IRaMuTeQ is © Pierre Ratinaud and
contributors (Python 3 port by Laurent Mérat).