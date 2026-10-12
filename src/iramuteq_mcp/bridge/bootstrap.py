"""Bootstrap the IRaMuTeQ source tree for headless use.

Reproduces what ``iramuteq.py`` does at import / frame construction time
(path discovery, user config seeding, R discovery, history), without any
GUI, and exposes the resulting ``ira``-like context object the analysis
classes expect (the same contract exercised by ``autres/iracmd.py``).

Nothing in the IRaMuTeQ source tree is copied or modified: the tree is
imported through ``sys.path``.
"""

from __future__ import annotations

import configparser  # stdlib must win over the vendored configparser.py
import json
import logging
import os
import sys
import tempfile

from . import wxstub

log = logging.getLogger("iramuteq_mcp")

DEFAULT_IRAMUTEQ_PATH = "/home/pampolha/Tools/iramuteq-0.8alpha7-sources"

_ctx: "IraContext | None" = None


class NoOpTree:
    """Stand-in for the GUI analysis tree (ira.tree.AddAnalyse / AddMatAnalyse)."""

    def AddAnalyse(self, *args, **kwargs):
        pass

    def AddMatAnalyse(self, *args, **kwargs):
        pass

    def OnItemAppend(self, *args, **kwargs):
        pass


class IraContext:
    """The ``ira``/parent object expected by the IRaMuTeQ analysis classes.

    Attribute surface follows ``iramuteq.py``'s ``IraFrame`` and
    ``autres/iracmd.py::CmdLine``.
    """

    def __init__(self, appli_path: str):
        self.AppliPath = appli_path
        self.ImagePath = os.path.join(appli_path, "images")

        # configuration generale (factory files live in the source tree)
        from chemins import (
            ConstructConfigPath,
            ConstructDicoPath,
            ConstructGlobalPath,
            PathOut,
            RscriptsPath as RSCRIPT_FILES,
        )

        global_paths = ConstructGlobalPath(appli_path)
        self.ConfigGlob = configparser.ConfigParser()
        self.ConfigGlob.read(global_paths["global"])

        user_home = os.getenv("HOME") or os.path.expanduser("~")
        self.UserConfigPath = os.path.abspath(
            os.path.join(user_home, ".iramuteq-%s" % self.ConfigGlob.get("DEFAULT", "version_nb"))
        )
        self.ConfigPath = ConstructConfigPath(self.UserConfigPath)
        self.DictPath = ConstructDicoPath(self.UserConfigPath)

        # user config directory + seed any missing config/dictionaries
        from checkinstall import CreateIraDirectory

        CreateIraDirectory(self.UserConfigPath, appli_path)
        self._seed_configs(ConstructConfigPath(appli_path, user=False))
        self._seed_dicos(ConstructDicoPath(appli_path))

        # R script locations: PathOut-based map exactly like IraFrame.__init__
        self.RscriptsPath = PathOut(dirout=os.path.join(appli_path, "Rscripts"))
        self.RscriptsPath.basefiles(RSCRIPT_FILES)

        self.pref = configparser.RawConfigParser()
        self.pref.read(self.ConfigPath["preferences"])
        self.PathPath = configparser.ConfigParser()
        self.PathPath.read(self.ConfigPath["path"])
        self.syscoding = "utf8"
        self.SysEncoding = "utf8"
        self.encode = "utf8"
        self.TEMPDIR = tempfile.mkdtemp(prefix="iramuteq-mcp-")

        self.RPath = self._discover_r()
        self.history = self._open_history()
        self.lexique = {}
        self.expressions = {}
        self._lexicon_langs: set[str] = set()
        self.tree = NoOpTree()
        self.filename = ""
        self.input_path = ""

    # ------------------------------------------------------------------ setup

    def _seed_configs(self, factory: dict) -> None:
        import shutil

        for key, dest in self.ConfigPath.items():
            if key in ("global", "history"):
                continue
            if not os.path.exists(dest) and os.path.exists(factory[key]):
                shutil.copyfile(factory[key], dest)

    def _seed_dicos(self, factory: dict) -> None:
        import shutil

        for key, dest in self.DictPath.items():
            if not os.path.exists(dest) and os.path.exists(factory.get(key, "")):
                shutil.copyfile(factory[key], dest)

    def _discover_r(self) -> str:
        from checkinstall import CheckRPath, FindRPathNix

        path_path = configparser.ConfigParser()
        path_path.read(self.ConfigPath["path"])
        if CheckRPath(path_path):
            return path_path.get("PATHS", "rpath")
        best = FindRPathNix()
        if best:
            path_path.set("PATHS", "rpath", best)
            with open(self.ConfigPath["path"], "w", encoding="utf8") as f:
                path_path.write(f)
            return best
        raise RuntimeError(
            "R executable not found; install R or set [PATHS] rpath in %s"
            % self.ConfigPath["path"]
        )

    def _open_history(self):
        from functions import History

        histfile = os.path.join(self.UserConfigPath, "history.db")
        if not os.path.exists(histfile):
            with open(histfile, "w", encoding="utf8") as f:
                json.dump({"history": [], "matrix": []}, f)
        return History(histfile)

    # ------------------------------------------------------------------ lexicon

    def load_lexicon(self, lang: str = "french") -> None:
        """Load Lexique 3 + expression dictionary for ``lang`` (cached)."""
        if lang in self._lexicon_langs:
            return
        from functions import ReadDicoAsDico, ReadLexique

        ReadLexique(self, lang=lang)
        exp_path = self.DictPath.get(f"{lang}_exp")
        self.expressions = ReadDicoAsDico(exp_path) if exp_path and os.path.exists(exp_path) else {}
        self._lexicon_langs.add(lang)


def init(iramuteq_path: str | None = None) -> IraContext:
    """Idempotent bootstrap: wx stub, sys.path, config, R, history."""
    global _ctx
    if _ctx is not None:
        return _ctx

    wxstub.install()

    root = os.path.abspath(
        iramuteq_path or os.environ.get("IRAMUTEQ_PATH", DEFAULT_IRAMUTEQ_PATH)
    )
    if not os.path.isfile(os.path.join(root, "iramuteq.py")):
        raise RuntimeError(f"IRaMuTeQ sources not found at {root} (set IRAMUTEQ_PATH)")

    # langue._init() and friends derive AppliPath from sys.argv[0]
    sys.argv[0] = os.path.join(root, "iramuteq.py")
    if root not in sys.path:
        sys.path.insert(0, root)

    _ctx = IraContext(root)

    from . import patches

    patches.install()
    log.info("iramuteq bootstrap ok: appli=%s user=%s R=%s", root, _ctx.UserConfigPath, _ctx.RPath)
    return _ctx


def get() -> IraContext:
    return init()
