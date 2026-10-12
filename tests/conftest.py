"""Pytest configuration: fixtures and the requires_r skip rule."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from helpers import FIXTURES, write_matrix

_R_STATUS: dict | None = None


def _r_status() -> dict:
    """Probe R and its packages once per session (through the real status tool)."""
    global _R_STATUS
    if _R_STATUS is None:
        from helpers import run

        from iramuteq_mcp import status

        try:
            _R_STATUS = run(status())
        except Exception as exc:  # R missing or sources not found
            _R_STATUS = {"ready": False, "error": str(exc)}
    return _R_STATUS


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_r") is None:
        return
    st = _r_status()
    if not st.get("ready"):
        reason = st.get("error") or f"R packages missing: {st.get('r_packages_missing')}"
        pytest.skip(f"requires R + IRaMuTeQ packages ({reason})")


@pytest.fixture(scope="session")
def corpus_path(tmp_path_factory) -> Path:
    """The small ALCESTE fixture, copied so analyses never write into the repo."""
    dest = tmp_path_factory.mktemp("corpus") / "mini_corpus.txt"
    shutil.copy(FIXTURES / "mini_corpus.txt", dest)
    return dest


@pytest.fixture(scope="session")
def tgen_path(tmp_path_factory) -> Path:
    dest = tmp_path_factory.mktemp("tgen") / "mini_tgen.txt"
    shutil.copy(FIXTURES / "mini_tgen.txt", dest)
    return dest


@pytest.fixture(scope="session")
def matrix_path(tmp_path_factory) -> Path:
    return write_matrix(tmp_path_factory.mktemp("matrix") / "mini_matrix.csv")


def pytest_configure(config):
    sys.path.insert(0, str(Path(__file__).parent))
