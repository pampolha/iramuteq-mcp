"""Sub-corpus operations: IRaMuTeQ's file splitters and class export.

The splitters in ``tools.py`` write their outputs beside the input file (as
the GUI does); we report exactly which files were created by diffing the
directory before and after the run.
"""

from __future__ import annotations

import hashlib
import os

from . import textops


def _listing(directory: str) -> dict[str, str]:
    """Snapshot: file name -> content digest (timestamps are too coarse to rely on)."""
    if not os.path.isdir(directory):
        return {}
    out = {}
    for n in os.listdir(directory):
        p = os.path.join(directory, n)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                out[n] = hashlib.sha1(f.read()).hexdigest()
    return out


def _created(directory: str, before: dict[str, str]) -> list[str]:
    """Files that are new or whose content changed since the snapshot."""
    after = _listing(directory)
    changed = [n for n, digest in after.items() if before.get(n) != digest]
    return sorted(os.path.join(directory, n) for n in changed)


def _header_tokens(src: str) -> set[str]:
    """All starred tokens found on ``****`` header lines."""
    tokens: set[str] = set()
    with open(src, encoding="utf8", errors="replace") as f:
        for line in f:
            if line.startswith("**** "):
                tokens.update(t for t in line.split()[1:] if t.startswith("*"))
    return tokens


def _input(path: str) -> str:
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"corpus not found: {p}")
    return p


def split_by_variable(path: str, variable: str, encoding: str = "utf8") -> dict:
    """Split ``****`` texts by a ``*variable`` into ``<modality>.txt`` files."""
    from tools import SplitFromVar

    src = _input(path)
    bare = variable.lstrip("*")
    if not bare or "_" in bare:
        raise ValueError("variable must be a bare name, e.g. 'sexe' (not 'sexe_h')")
    # the source matches the starred variable name as written in the headers
    var = "*" + bare
    base = os.path.dirname(src)
    before = _listing(base)
    SplitFromVar({"filein": src, "var": var, "encodein": encoding, "encodeout": "utf8"})
    # the splitter rewrites <var>_<mod>.txt for each modality present
    expected = {
        os.path.join(base, t.lstrip("*") + ".txt") for t in _header_tokens(src) if t.startswith("*" + bare + "_")
    }
    files = sorted(set(_created(base, before)) | {p for p in expected if os.path.isfile(p)})
    return {"input": src, "variable": bare, "created": files, "count": len(files)}


def extract_themes(path: str, themes: list[str], encoding: str = "utf8") -> dict:
    """Keep only the ``-*theme`` sections listed; written as one combined file."""
    from tools import SplitFromThem

    src = _input(path)
    if not themes:
        raise ValueError("give at least one theme, e.g. ['travail']")
    them = [t if t.startswith("-*") else "-*" + t for t in themes]
    base = os.path.dirname(src)
    before = _listing(base)
    SplitFromThem({"filein": src, "them": them, "encodein": encoding, "encodeout": "utf8"})
    out = os.path.join(base, "_".join(t.replace("-*", "") for t in them))
    if not os.path.isfile(out):
        raise RuntimeError(f"no text found for themes {themes}")
    return {"input": src, "themes": them, "created": _created(base, before), "output": out}


def extract_mods(
    path: str,
    mods: list[str],
    onefile: bool = False,
    encoding: str = "utf8",
) -> dict:
    """Extract texts carrying the given ``*modalities``.

    ``onefile=True`` writes a single file; otherwise one file per modality.
    """
    from tools import ExtractMods

    src = _input(path)
    if not mods:
        raise ValueError("give at least one modality, e.g. ['*sexe_h']")
    starred = [m if m.startswith("*") else "*" + m for m in mods]
    base = os.path.dirname(src)
    before = _listing(base)
    ExtractMods(
        {
            "filein": src,
            "mods": starred,
            "onefile": onefile,
            "encodein": encoding,
            "encodeout": "utf8",
        }
    )
    if onefile:
        expected = {os.path.join(base, "_".join(m.lstrip("*") for m in starred) + ".txt")}
    else:
        expected = {os.path.join(base, t.lstrip("*") + ".txt") for t in _header_tokens(src) if t in starred}
    files = sorted(set(_created(base, before)) | {p for p in expected if os.path.isfile(p)})
    return {"input": src, "mods": starred, "onefile": onefile, "created": files}


def export_class(
    corpus_path: str,
    analysis_dir: str,
    class_id: int,
    lang: str = "french",
    encoding: str = "utf8",
    lem: bool = False,
    uci: bool = False,
    out_path: str | None = None,
    ucemethod: int | None = None,
    ucesize: int | None = None,
) -> dict:
    """Export one ALCESTE class (from ``analysis_dir/uce.csv``) as a new corpus file.

    The corpus must be the one the classification was run on (same file,
    language, encoding and segmentation), so that UCE identifiers line up.
    """
    corpus, _ = textops.get_corpus(
        corpus_path, lang=lang, encoding=encoding, options={"ucemethod": ucemethod, "ucesize": ucesize}
    )
    uce_file = os.path.join(os.path.abspath(analysis_dir), "uce.csv")
    if not os.path.isfile(uce_file):
        raise FileNotFoundError(f"no uce.csv in {analysis_dir}; run corpus_alceste first")
    corpus.make_ucecl_from_R(uce_file)
    nclasses = len(corpus.lc)
    if not 1 <= class_id <= nclasses:
        raise ValueError(f"class_id must be between 1 and {nclasses}")
    members = corpus.lc[class_id - 1]
    if not members:
        raise ValueError(f"class {class_id} is empty")
    out = out_path or os.path.join(os.path.abspath(analysis_dir), f"classe_{class_id}.txt")
    corpus.export_classe(out, class_id, lem=lem, uci=uci)
    return {
        "class_id": class_id,
        "segments": len(members),
        "classes": nclasses,
        "output": out,
    }
