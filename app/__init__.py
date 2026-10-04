"""don-dj: a local two-deck DJ program for beginners."""
from __future__ import annotations

import os
from pathlib import Path


def _prefer_pyside_plugins() -> None:
    """Use PySide6's own plugins.

    Anaconda ships a qt.conf next to python.exe that points every Qt app at
    the Qt 5 plugins in Library/plugins. PySide6 is Qt 6, so those plugins
    are rejected and the window dies with "no Qt platform plugin could be
    initialized". Point Qt at this install before any QApplication is created.
    """
    import PySide6
    from PySide6.QtCore import QCoreApplication

    plugins = Path(PySide6.__file__).resolve().parent / "plugins"
    os.environ["QT_PLUGIN_PATH"] = os.fspath(plugins)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.fspath(plugins / "platforms")
    if QCoreApplication.instance() is None:
        QCoreApplication.setLibraryPaths([os.fspath(plugins)])


_prefer_pyside_plugins()
