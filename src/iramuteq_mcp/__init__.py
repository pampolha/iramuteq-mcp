# -*- coding: utf-8 -*-
"""IRaMuTeQ MCP server — headless access to IRaMuTeQ corpus parsing and R analyses.

IRaMuTeQ (http://www.iramuteq.org) is a French/R text-mining suite (GPL-2+).
Its 0.8 sources ship a wxPython GUI; this server exposes the corpus format
and the R analysis layer without any GUI dependency.

Corpus format (see corpus.py::BuildCorpus.read_corpus):
  - a line starting with ``****`` starts a new UCI (document) and carries its
    metadata, e.g. ``**** *year 2020`` or ``**** **author Doe``
  - a line starting with ``-*`` marks a paragraph
  - any other non-empty line is text content
"""

from __future__ import annotations

import os
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP

IRAMUTEQ_PATH = Path(
    os.environ.get("IRAMUTEQ_PATH", "/home/pampolha/Tools/iramuteq-0.8alpha7-sources")
)
RSCRIPT = os.environ.get("IRAMUTEQ_RSCRIPT", "Rscript")

mcp = FastMCP("iramuteq")

# --------------------------------------------------------------------------
# Corpus parsing (mirrors corpus.py::read_corpus / testetoile)
# --------------------------------------------------------------------------

WORD_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ']+", re.UNICODE)


class Uci:
    """A document (Unité de Contexte Initiale)."""

    def __init__(self, ident: int, header: str):
        self.ident = ident
        self.header = header
        self.variables: dict[str, str] = {}
        self.paras: list[str] = []
        self.lines: list[str] = []

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _parse_header(header: str) -> dict[str, str]:
    """``**** *year 2020 **author Doe`` -> {'year': '2020', 'author': 'Doe'}``."""
    variables: dict[str, str] = {}
    tokens = header.split()[1:]  # drop leading ****
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("*"):
            name = tokens[i].lstrip("*")
            value = ""
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("*"):
                value = tokens[i + 1]
                i += 1
            variables[name] = value
        i += 1
    return variables


def _detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:4096]
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def parse_corpus(path: str, encoding: str | None = None) -> list[Uci]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"corpus not found: {path}")
    enc = encoding or _detect_encoding(p)
    ucis: list[Uci] = []
    current: Uci | None = None
    for line in p.read_text(encoding=enc).splitlines():
        line = line.rstrip("\r\n")
        if line.startswith("****"):
            current = Uci(len(ucis), line)
            current.variables = _parse_header(line)
            ucis.append(current)
        elif current is None:
            if line.strip():
                raise ValueError(
                    "text before the first '****' document marker — "
                    "is this an IRaMuTeQ corpus?"
                )
        elif line.startswith("-*"):
            current.paras.append(line.split()[0])
        elif line.strip():
            current.lines.append(line)
    if not ucis:
        raise ValueError("no document found (missing '****' markers)")
    return ucis


def _tokens(text: str) -> list[str]:
    """Light cleaning close to IRaMuTeQ defaults: lowercase, keep letters/digits."""
    text = text.replace("’", "'").replace("œ", "oe")
    return [t for t in WORD_RE.findall(text.lower()) if not t.isdigit()]


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@mcp.tool
def corpus_info(
    path: Annotated[str, "Absolute path to an IRaMuTeQ corpus text file"],
    encoding: Annotated[str | None, "Force file encoding (default: auto-detect)"] = None,
) -> dict:
    """Parse an IRaMuTeQ corpus and return its structure summary.

    Returns number of documents (UCIs), paragraphs, text lines, tokens,
    distinct forms, hapax, and the variables found in `****` headers.
    """
    ucis = parse_corpus(path, encoding)
    all_text = "\n".join(u.text for u in ucis)
    tokens = _tokens(all_text)
    freq = Counter(tokens)
    hapax = sum(1 for f in freq if freq[f] == 1)
    variables: dict[str, list[str]] = {}
    for u in ucis:
        for k, v in u.variables.items():
            variables.setdefault(k, [])
            if v and v not in variables[k]:
                variables[k].append(v)
    return {
        "path": path,
        "documents": len(ucis),
        "paragraphs": sum(len(u.paras) for u in ucis),
        "text_lines": sum(1 for u in ucis for _ in u.lines),
        "tokens": len(tokens),
        "forms": len(freq),
        "hapax": hapax,
        "variables": variables,
        "document_headers": [u.header for u in ucis[:20]],
    }


@mcp.tool
def word_frequencies(
    path: Annotated[str, "Absolute path to an IRaMuTeQ corpus text file"],
    top: Annotated[int, "Number of most frequent forms to return"] = 50,
    min_len: Annotated[int, "Minimum token length"] = 1,
    encoding: str | None = None,
) -> list[dict]:
    """Return the most frequent forms of a corpus (descending frequency)."""
    ucis = parse_corpus(path, encoding)
    freq = Counter(
        t for t in _tokens("\n".join(u.text for u in ucis)) if len(t) >= min_len
    )
    return [
        {"form": form, "frequency": n, "percent": round(100 * n / max(1, sum(freq.values())), 2)}
        for form, n in freq.most_common(top)
    ]


@mcp.tool
def concordance(
    path: Annotated[str, "Absolute path to an IRaMuTeQ corpus text file"],
    query: Annotated[str, "Word or regular expression to search"],
    width: Annotated[int, "Characters of context on each side"] = 60,
    max_hits: Annotated[int, "Maximum matches to return"] = 50,
    case_sensitive: bool = False,
    encoding: str | None = None,
) -> list[dict]:
    """Keyword-in-context search over the corpus.

    Each hit contains the left context, the match, the right context and the
    document header it was found in.
    """
    ucis = parse_corpus(path, encoding)
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query if any(c in query for c in r"\.*^$[]|()+?{}") else re.escape(query), flags)
    hits = []
    for u in ucis:
        for m in pattern.finditer(u.text):
            l = max(0, m.start() - width)
            r = min(len(u.text), m.end() + width)
            hits.append(
                {
                    "document": u.header,
                    "left": u.text[l : m.start()].replace("\n", " "),
                    "match": m.group(0),
                    "right": u.text[m.end() : r].replace("\n", " "),
                }
            )
            if len(hits) >= max_hits:
                return hits
    return hits


@mcp.tool
def variable_distribution(
    path: Annotated[str, "Absolute path to an IRaMuTeQ corpus text file"],
    variable: Annotated[str, "Variable name as written in the header (without *)"],
    encoding: str | None = None,
) -> dict:
    """Distribution of a `*variable` across documents, with per-value token counts."""
    ucis = parse_corpus(path, encoding)
    dist: dict[str, dict] = {}
    for u in ucis:
        value = u.variables.get(variable)
        if value is None:
            continue
        d = dist.setdefault(value, {"documents": 0, "tokens": 0})
        d["documents"] += 1
        d["tokens"] += len(_tokens(u.text))
    return {"variable": variable, "distribution": dist}


@mcp.tool
def make_subcorpus(
    path: Annotated[str, "Absolute path to an IRaMuTeQ corpus text file"],
    variable: Annotated[str, "Header variable to filter on"],
    value: Annotated[str, "Value the variable must equal"],
    output_path: Annotated[str, "Where to write the sub-corpus text file"],
    encoding: str | None = None,
) -> dict:
    """Extract a sub-corpus of the documents whose `variable` equals `value`.

    Writes a valid IRaMuTeq corpus file (headers preserved) and returns the
    number of documents written.
    """
    ucis = parse_corpus(path, encoding)
    selected = [u for u in ucis if u.variables.get(variable) == value]
    if not selected:
        raise ValueError(f"no document with {variable}={value!r}")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for u in selected:
            f.write(u.header + "\n")
            for line in u.lines:
                f.write(line + "\n")
    return {"output_path": str(out), "documents": len(selected)}


# --------------------------------------------------------------------------
# R layer
# --------------------------------------------------------------------------

R_PACKAGES = [
    "ca", "gee", "ape", "igraph", "proxy", "wordcloud", "irlba",
    "textometry", "sna", "network", "intergraph", "rgl",
]


@mcp.tool
def check_r_packages() -> list[dict]:
    """Check which R packages required by IRaMuTeq are installed."""
    pkgs = ",".join(f"'{p}'" for p in R_PACKAGES)
    proc = subprocess.run(
        [RSCRIPT, "-e", f"for (p in c({pkgs})) cat(p, requireNamespace(p, quietly=TRUE), sep='\\n')"],
        capture_output=True, text=True, timeout=120,
    )
    out = []
    lines = [l for l in proc.stdout.splitlines() if l]
    for name, status in zip(lines[0::2], lines[1::2]):
        out.append({"package": name, "installed": status == "TRUE"})
    return out


@mcp.tool
def list_r_scripts() -> list[str]:
    """List the R analysis scripts bundled with IRaMuTeq (Rscripts directory)."""
    rdir = IRAMUTEQ_PATH / "Rscripts"
    if not rdir.is_dir():
        raise FileNotFoundError(f"Rscripts not found under {IRAMUTEQ_PATH}")
    return sorted(p.name for p in rdir.glob("*.R"))


@mcp.tool
def run_r_script(
    script: Annotated[str, "File name of a script in IRaMuTeQ's Rscripts directory, e.g. 'CHD.R'"],
    args: Annotated[list[str], "Command-line arguments passed to the script"] = [],
    timeout: Annotated[int, "Timeout in seconds"] = 300,
) -> dict:
    """Run one of IRaMuTeQ's bundled R scripts with Rscript.

    Returns exit code, stdout and stderr. The script runs with the corpus
    tables it expects as arguments — use this for advanced/custom analyses.
    """
    script_path = IRAMUTEQ_PATH / "Rscripts" / script
    if not script_path.is_file():
        available = ", ".join(list_r_scripts())
        raise FileNotFoundError(f"unknown script {script!r}. Available: {available}")
    proc = subprocess.run(
        [RSCRIPT, str(script_path), *args],
        capture_output=True, text=True, timeout=timeout,
    )
    return {
        "script": script,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-20000:],
    }


def main() -> None:
    mcp.run()  # stdio transport by default


if __name__ == "__main__":
    main()