"""Import hook that makes the IRaMuTeQ analysis modules headless-safe.

The source files are never edited: a :data:`sys.meta_path` finder loads the
real IRaMuTeQ module and patches its namespace right after its code runs,
before the import statement that requested it returns.

* ``functions.progressbar`` -> no-op progress dialog (some analyses call it
  unconditionally: textstat.py:39, textaslexico.py:39, textlabbe.py:84).
* ``functions.check_Rresult`` -> raises :class:`RScriptError` with R's
  stderr instead of opening the ``BugReport`` wx dialog.
* ``functions.exec_rcode`` -> wrapped so the job manager can track (and
  cancel) the spawned ``R`` process.
* ``corpus.sqlite3`` -> shim allowing corpus handles to cross threads.
* ``openanalyse.OpenAnalyse`` -> no-op instead of the GUI result viewer.

Because ``functions`` and ``openanalyse`` are patched before any other module
can run ``from functions import progressbar``, every such binding receives
the patched object; nothing has to be rebound afterwards.  This only holds
when the hook is installed before the first IRaMuTeQ import, which
:func:`install` enforces.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import types
from collections.abc import Callable
from typing import Any

log = logging.getLogger("iramuteq_mcp")

_local = threading.local()

_ORIGINALS: dict[str, object] = {}
_PATCHERS: dict[str, Callable[[types.ModuleType], None]] = {}


class RScriptError(RuntimeError):
    """Raised when a generated R script fails."""

    def __init__(self, message: str, returncode: int | None = None, output: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.output = output


class StubProgress:
    """Stand-in for wx.ProgressDialog used by the analyses."""

    def Update(self, *args, **kwargs):
        return (True, False)

    def Pulse(self, *args, **kwargs):
        return (True, False)

    def Center(self, *args, **kwargs):
        pass

    def Destroy(self):
        pass


def _noop_openanalyse(*args, **kwargs):
    """Skip the GUI result-viewer; the MCP reads the output files instead."""


def _stub_progressbar(self, maxi):
    """Headless replacement for ``functions.progressbar``."""
    return StubProgress()


def _check_rresult(parent, pid):
    """Headless replacement for ``functions.check_Rresult``: raise instead of opening a dialog."""
    if isinstance(pid, subprocess.Popen):
        if pid.returncode != 0:
            try:
                out, err = pid.communicate(timeout=10)
            except Exception:
                out, err = "", ""
            output = "\n".join(str(x) for x in (out, err) if x)
            raise RScriptError(
                "R script failed (returncode=%s)" % pid.returncode,
                returncode=pid.returncode,
                output=output,
            )
        return True
    if isinstance(pid, int):
        if pid != 0:
            raise RScriptError("R script failed (returncode=%s)" % pid, returncode=pid)
        return True
    return True


def _make_exec_rcode(orig):
    """Wrap ``functions.exec_rcode`` so background R processes reach this thread's listener."""

    def exec_rcode(rpath, rcode, wait=True, graph=False):
        result = orig(rpath, rcode, wait=wait, graph=graph)
        if not wait:
            listener = getattr(_local, "listener", None)
            if listener is not None and hasattr(result, "poll"):
                listener(result)
        return result

    return exec_rcode


def set_process_listener(callback) -> None:
    """Register ``callback(popen)`` for R processes spawned in this thread."""
    _local.listener = callback


class _SqliteShim:
    """sqlite3 proxy allowing corpus handles to cross the job worker threads."""

    def __init__(self, real):
        self._real = real

    def connect(self, *args, **kwargs):
        kwargs.setdefault("check_same_thread", False)
        return self._real.connect(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _patches_module(name: str):
    """Register the decorated function as the patcher run when IRaMuTeQ module ``name`` loads."""

    def register(fn: Callable[[types.ModuleType], None]):
        _PATCHERS[name] = fn
        return fn

    return register


@_patches_module("functions")
def _patch_functions(mod: Any) -> None:
    """Swap the GUI-bound helpers of ``functions`` for their headless versions."""
    _ORIGINALS.update(progressbar=mod.progressbar, check_Rresult=mod.check_Rresult, exec_rcode=mod.exec_rcode)
    mod.progressbar = _stub_progressbar
    mod.check_Rresult = _check_rresult
    mod.exec_rcode = _make_exec_rcode(_ORIGINALS["exec_rcode"])


@_patches_module("corpus")
def _patch_corpus(mod: Any) -> None:
    """Let corpus sqlite handles be used from threads other than the one that opened them."""
    mod.sqlite3 = _SqliteShim(sqlite3)


@_patches_module("openanalyse")
def _patch_openanalyse(mod: Any) -> None:
    """Replace the GUI result viewer before any analysis module imports it."""
    _ORIGINALS["OpenAnalyse"] = mod.OpenAnalyse
    mod.OpenAnalyse = _noop_openanalyse


class _PatchingLoader(importlib.abc.Loader):
    """Runs the real loader, then the module's patcher, before the import returns."""

    def __init__(self, real: importlib.abc.Loader, patcher: Callable[[types.ModuleType], None]):
        self._real = real
        self._patcher = patcher

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        self._patcher(module)

    def __getattr__(self, name):
        return getattr(self._real, name)


class _PatchingFinder(importlib.abc.MetaPathFinder):
    """Resolves the patched module names inside the IRaMuTeQ source tree only."""

    def __init__(self, root: str):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        patcher = _PATCHERS.get(fullname)
        if patcher is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, [self.root])
        if spec is None or spec.loader is None:
            raise ImportError(f"IRaMuTeQ module {fullname!r} not found in {self.root}", name=fullname)
        spec.loader = _PatchingLoader(spec.loader, patcher)
        return spec


def is_installed() -> bool:
    """Whether the patching finder is registered on ``sys.meta_path``."""
    return any(isinstance(f, _PatchingFinder) for f in sys.meta_path)


def install(root: str) -> None:
    """Register the patching finder for the IRaMuTeQ tree at ``root``.

    Raises :class:`RuntimeError` when a patched module was imported earlier:
    modules that copied its unpatched helpers would otherwise keep them silently.
    """
    if is_installed():
        return
    early = sorted(name for name in _PATCHERS if name in sys.modules)
    if early:
        raise RuntimeError(f"IRaMuTeQ modules {early} were imported before the patch hook was installed")
    sys.meta_path.insert(0, _PatchingFinder(os.path.abspath(root)))
