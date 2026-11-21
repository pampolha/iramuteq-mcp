"""The import hook that patches the IRaMuTeQ sources: coverage, ordering and failure modes."""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
import threading
import traceback

import pytest

from iramuteq_mcp.bridge import bootstrap, patches

SOURCES = os.path.abspath(os.environ.get("IRAMUTEQ_PATH", bootstrap.DEFAULT_IRAMUTEQ_PATH))

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(SOURCES, "iramuteq.py")), reason=f"IRaMuTeQ sources not found at {SOURCES}"
)

ANALYSIS_MODULES = [
    "functions",
    "corpus",
    "openanalyse",
    "analysetxt",
    "analysematrix",
    "textstat",
    "textreinert",
    "textaslexico",
    "textlabbe",
    "textsimi",
    "textwordcloud",
    "textcheckcorpus",
    "tableau",
    "tabfrequence",
    "tabchi2",
    "tabchi2mcnemar",
    "tabchdalc",
    "tabsimi",
    "tabsplitvar",
    "tabverges",
    "tools",
    "guifunct",
]

PATCHED_NAMES = ("progressbar", "check_Rresult", "exec_rcode", "OpenAnalyse")


def _run_fresh(code: str) -> dict:
    """Run ``code`` in a new interpreter (clean import state) and return the JSON it prints last."""
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, "IRAMUTEQ_PATH": SOURCES},
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_no_module_keeps_an_unpatched_helper_after_importing_the_whole_tree():
    report = _run_fresh(
        f"""
        import importlib, json, os, sys
        from iramuteq_mcp.bridge import bootstrap, patches

        root = bootstrap.get().AppliPath
        required_failed = {{}}
        for name in {ANALYSIS_MODULES!r}:
            try:
                importlib.import_module(name)
            except BaseException as exc:
                required_failed[name] = repr(exc)
        optional_failed = {{}}
        for fn in sorted(os.listdir(root)):
            name = fn[:-3]
            if not fn.endswith(".py") or name == "iramuteq" or name in sys.modules:
                continue
            try:
                importlib.import_module(name)
            except BaseException as exc:
                optional_failed[name] = repr(exc)

        functions = sys.modules["functions"]
        expected = {{
            "progressbar": functions.progressbar,
            "check_Rresult": functions.check_Rresult,
            "exec_rcode": functions.exec_rcode,
            "OpenAnalyse": patches._noop_openanalyse,
        }}
        originals = {{id(v): k for k, v in patches._ORIGINALS.items()}}
        leaks, wrong, checked = [], [], []
        for mname, mod in list(sys.modules.items()):
            if mod is None or mname == "iramuteq_mcp.bridge.patches":
                continue
            try:
                items = list(vars(mod).items())
            except TypeError:
                continue
            for attr, val in items:
                if id(val) in originals:
                    leaks.append(f"{{mname}}.{{attr}} is the original {{originals[id(val)]}}")
            in_tree = os.path.abspath(getattr(mod, "__file__", "") or "").startswith(root + os.sep)
            if in_tree:
                for attr in {PATCHED_NAMES!r}:
                    if attr in vars(mod):
                        checked.append(f"{{mname}}.{{attr}}")
                        if vars(mod)[attr] is not expected[attr]:
                            wrong.append(f"{{mname}}.{{attr}}")
        print(json.dumps({{
            "required_failed": required_failed,
            "optional_failed": optional_failed,
            "originals": sorted(patches._ORIGINALS),
            "leaks": leaks,
            "wrong": wrong,
            "checked": checked,
            "functions_file": functions.__file__,
            "root": root,
            "corpus_sqlite_shim": isinstance(sys.modules["corpus"].sqlite3, patches._SqliteShim),
            "openanalyse_patched": sys.modules["openanalyse"].OpenAnalyse is patches._noop_openanalyse,
            "finders": sum(isinstance(f, patches._PatchingFinder) for f in sys.meta_path),
        }}))
        """
    )
    assert report["required_failed"] == {}
    assert report["originals"] == ["OpenAnalyse", "check_Rresult", "exec_rcode", "progressbar"]
    assert report["leaks"] == [], "modules still hold unpatched IRaMuTeQ helpers"
    assert report["wrong"] == [], "modules bound something other than the patched helper"
    assert report["functions_file"].startswith(report["root"] + os.sep)
    assert report["corpus_sqlite_shim"] is True
    assert report["openanalyse_patched"] is True
    assert report["finders"] == 1
    must_be_checked = {
        "textstat.progressbar",
        "textaslexico.progressbar",
        "textlabbe.progressbar",
        "corpus.progressbar",
        "analysetxt.OpenAnalyse",
        "analysematrix.OpenAnalyse",
    }
    assert must_be_checked <= set(report["checked"]), "the scan did not see the bindings it exists to check"


def test_bootstrap_refuses_when_a_patched_module_was_imported_first():
    report = _run_fresh(
        """
        import json, os, sys
        from iramuteq_mcp.bridge import bootstrap, patches, wxstub

        root = os.environ["IRAMUTEQ_PATH"]
        wxstub.install()
        sys.path.insert(0, root)
        import functions
        try:
            bootstrap.init(root)
            outcome = "no error"
        except RuntimeError as exc:
            outcome = str(exc)
        print(json.dumps({"outcome": outcome, "installed": patches.is_installed()}))
        """
    )
    assert "imported before the patch hook was installed" in report["outcome"]
    assert "functions" in report["outcome"]
    assert report["installed"] is False


def test_hook_only_resolves_its_modules_inside_the_source_tree(tmp_path):
    finder = patches._PatchingFinder(str(tmp_path))
    assert finder.find_spec("json") is None
    assert finder.find_spec("textreinert") is None
    with pytest.raises(ImportError, match="not found"):
        finder.find_spec("functions")
    (tmp_path / "functions.py").write_text("progressbar = check_Rresult = exec_rcode = None\n")
    importlib.invalidate_caches()
    spec = finder.find_spec("functions")
    assert spec.origin == str(tmp_path / "functions.py")
    assert isinstance(spec.loader, patches._PatchingLoader)


def test_install_is_idempotent():
    root = bootstrap.get().AppliPath
    patches.install(root)
    patches.install(root)
    assert sum(isinstance(f, patches._PatchingFinder) for f in sys.meta_path) == 1


def test_patched_modules_keep_their_source_for_tracebacks():
    bootstrap.get()
    import functions

    loader = functions.__spec__.loader
    assert isinstance(loader, patches._PatchingLoader)
    assert "def ReadDicoAsDico" in loader.get_source("functions")
    assert loader.get_filename("functions") == functions.__file__
    assert inspect.getsource(functions.ReadDicoAsDico).startswith("def ReadDicoAsDico")
    with pytest.raises(FileNotFoundError) as info:
        functions.ReadDicoAsDico("/nonexistent/dictionary.txt")
    rendered = "".join(traceback.format_exception(info.value))
    assert "open(dicopath" in rendered, "traceback lost the IRaMuTeQ source line"


def test_corpus_sqlite_handles_cross_threads():
    bootstrap.get()
    import corpus

    conn = corpus.sqlite3.connect(":memory:")
    errors = []

    def use():
        try:
            conn.execute("select 1").fetchone()
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=use)
    worker.start()
    worker.join()
    assert errors == []


class _FakePopen(subprocess.Popen):
    def __init__(self, returncode, out="", err=""):
        self.returncode = returncode
        self._out, self._err = out, err

    def communicate(self, input=None, timeout=None):
        return self._out, self._err

    def poll(self):
        return self.returncode

    def __del__(self):
        pass


def test_check_rresult_raises_with_r_output():
    assert patches._check_rresult(None, _FakePopen(0)) is True
    assert patches._check_rresult(None, 0) is True
    with pytest.raises(patches.RScriptError) as info:
        patches._check_rresult(None, _FakePopen(1, "", "Error: boom"))
    assert info.value.returncode == 1
    assert "Error: boom" in info.value.output
    with pytest.raises(patches.RScriptError):
        patches._check_rresult(None, 2)


def test_exec_rcode_wrapper_reports_background_processes_only():
    proc = _FakePopen(None)
    wrapped = patches._make_exec_rcode(lambda rpath, rcode, wait=True, graph=False: proc if not wait else 0)
    seen = []
    patches.set_process_listener(seen.append)
    try:
        assert wrapped("R", "script.R", wait=True) == 0
        assert seen == []
        assert wrapped("R", "script.R", wait=False) is proc
        assert seen == [proc]
    finally:
        patches.set_process_listener(None)


@pytest.mark.requires_r
def test_failing_r_script_is_tracked_and_raises_through_the_real_helpers(tmp_path):
    ctx = bootstrap.get()
    import functions

    script = tmp_path / "fail.R"
    script.write_text('stop("headless-boom")\n')
    seen = []
    patches.set_process_listener(seen.append)
    try:
        pid = functions.exec_rcode(ctx.RPath, str(script), wait=False)
        pid.wait(timeout=60)
        with pytest.raises(patches.RScriptError) as info:
            functions.check_Rresult(None, pid)
    finally:
        patches.set_process_listener(None)
    assert seen == [pid]
    assert info.value.returncode != 0
    assert "headless-boom" in info.value.output
