"""Permissive wxPython stand-in for headless use of the IRaMuTeQ sources.

Every IRaMuTeQ module imports wx at module level (often to subclass
``wx.Dialog`` or combine style constants), but the analysis logic never
touches GUI code when run with ``dlg=False``/``dlg=None``.  This module
installs a :mod:`importlib` meta-path finder that fabricates ``wx`` and any
``wx.*`` submodule on demand.

Fabricated attributes are real classes (so they can be subclassed and used
in ``isinstance``) whose instances accept any call and return dummies:

* ``class MyDialog(wx.Dialog)`` works;
* ``style = wx.DEFAULT | wx.BORDER`` works (``type.__or__`` on 3.10+);
* ``wx.Size(400, 150)``, ``PyEmbeddedImage(data).GetBitmap()`` work;
* ``"gtk2" in wx.PlatformInfo`` works (metaclass iteration).

``install()`` must run before any IRaMuTeQ module is imported.  It is a
no-op if the stub is already active.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types

__all__ = ["install", "is_installed"]

_STUB_MARK = "__iramuteq_wxstub__"


class _DummyMeta(type):
    """Metaclass making generated classes behave like rich wx constants."""

    def __getattr__(cls, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        sub = _make_class(name)
        setattr(cls, name, sub)
        return sub

    def __iter__(cls):
        return iter(())

    def __or__(cls, other):
        return _make_class(f"{cls.__name__}_or_{getattr(other, '__name__', other)}")()

    def __ror__(cls, other):
        return _make_class(f"or_{cls.__name__}")()


class _Dummy(metaclass=_DummyMeta):
    """Base of every generated class; instances are callables that swallow all."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Dummy()

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        val = _Dummy()
        setattr(self, name, val)
        return val

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return True

    def __repr__(self):
        return f"<wx-stub {type(self).__name__}>"


def _make_class(name: str) -> type:
    return _DummyMeta(name, (_Dummy,), {})


class _StubModule(types.ModuleType):
    """Module that generates a dummy class for any unknown attribute."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        val = _make_class(name)
        setattr(self, name, val)
        return val


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = _StubModule(spec.name)
        mod.__path__ = []  # mark as package so submodules resolve
        mod.__iramuteq_wxstub__ = True
        return mod

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "wx" or fullname.startswith("wx."):
            return importlib.machinery.ModuleSpec(fullname, _Loader(), is_package=True)
        return None


def is_installed() -> bool:
    wx = sys.modules.get("wx")
    return bool(wx is not None and getattr(wx, _STUB_MARK, False))


def install() -> None:
    """Install the stub finder and evict any already-imported wx modules."""
    if is_installed():
        return
    for name in list(sys.modules):
        if name == "wx" or name.startswith("wx."):
            del sys.modules[name]
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    wx = importlib.import_module("wx")
    setattr(wx, _STUB_MARK, True)
