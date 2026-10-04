"""voxograph.ui - public API for the split wxPython GUI package.

Re-exports the names the former ``ui.py`` module exposed at module level so
``voxograph.ui:main``, ``from voxograph.ui import MainWindow``, ``EVENTS``,
``CoreThread`` and ``_extract_bridge_json`` keep working unchanged.
"""
from voxograph.ui.events import (
    BOOK_DETAILS_WRAP_WIDTH,
    EVENTS,
    border,
)
from voxograph.ui.bridge import _extract_bridge_json
from voxograph.ui.window import MainWindow
from voxograph.ui.core_thread import CoreThread
from voxograph.ui.app import main

__all__ = [
    'BOOK_DETAILS_WRAP_WIDTH',
    'EVENTS',
    'MainWindow',
    'CoreThread',
    '_extract_bridge_json',
    'border',
    'main',
]
