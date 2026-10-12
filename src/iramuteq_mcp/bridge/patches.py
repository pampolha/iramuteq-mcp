"""Runtime monkeypatches that make IRaMuTeQ analysis modules headless-safe.

Three patches, all done by attribute replacement on the source modules (the
source files themselves are never edited):

1. ``functions.progressbar`` -> no-op progress dialog (some analyses call it
   unconditionally: textstat.py:39, textaslexico.py:39, textlabbe.py:84).
2. ``functions.check_Rresult`` -> raises :class:`RScriptError` with R's
   stderr instead of opening the ``BugReport`` wx dialog.
3. ``functions.exec_rcode`` -> wrapped so the job manager can track (and
   cancel) the spawned ``R`` process.

Modules that did ``from functions import progressbar`` bind the name at
import time, so after importing the analysis modules call :func:`sweep` to
replace any remaining reference to the original objects.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading

log = logging.getLogger("iramuteq_mcp")

_local = threading.local()

_ORIGINALS: dict[str, object] = {}
_installed = False


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
    return StubProgress()


def _check_rresult(parent, pid):
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


def install() -> None:
    global _installed
    if _installed:
        return
    import functions

    _ORIGINALS["progressbar"] = functions.progressbar
    _ORIGINALS["check_Rresult"] = functions.check_Rresult
    _ORIGINALS["exec_rcode"] = functions.exec_rcode

    functions.progressbar = _stub_progressbar
    functions.check_Rresult = _check_rresult
    functions.exec_rcode = _make_exec_rcode(_ORIGINALS["exec_rcode"])

    sweep()
    _installed = True


class _SqliteShim:
    """sqlite3 proxy allowing corpus handles to cross the job worker threads."""

    def __init__(self, real):
        self._real = real

    def connect(self, *args, **kwargs):
        kwargs.setdefault("check_same_thread", False)
        return self._real.connect(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def sweep() -> None:
    """Replace stale `from functions import ...` bindings with the patched ones."""
    import functions

    pairs = [
        (_ORIGINALS.get("progressbar"), functions.progressbar),
        (_ORIGINALS.get("check_Rresult"), functions.check_Rresult),
        (_ORIGINALS.get("exec_rcode"), functions.exec_rcode),
    ]
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        for name, val in list(vars(mod).items()):
            for old, new in pairs:
                if old is not None and val is old and new is not val:
                    setattr(mod, name, new)
                    break
        # corpus keeps sqlite handles across threads (build thread != job thread)
        if mod.__name__ == "corpus" and not isinstance(vars(mod).get("sqlite3"), _SqliteShim):
            import sqlite3

            mod.sqlite3 = _SqliteShim(sqlite3)
        # AnalyseMatrix's success path calls OpenAnalyse (GUI result viewer)
        oa = sys.modules.get("openanalyse")
        if oa is not None and mod.__name__ not in ("openanalyse", "iramuteq_mcp.bridge.patches"):
            if vars(mod).get("OpenAnalyse") is oa.OpenAnalyse:
                setattr(mod, "OpenAnalyse", _noop_openanalyse)
