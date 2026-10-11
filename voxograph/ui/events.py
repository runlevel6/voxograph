"""voxograph.ui.events - wx event table and shared layout constants."""
from wx.lib.newevent import NewEvent

from voxograph.core.constants import BOOK_DETAILS_WRAP_WIDTH, border


EVENTS = {
    'CORE_STARTED': NewEvent(),
    'CORE_PROGRESS': NewEvent(),
    'CORE_CHAPTER_STARTED': NewEvent(),
    'CORE_CHAPTER_FINISHED': NewEvent(),
    'CORE_AI_REWRITE': NewEvent(),
    'CORE_AI_RETRY_EXHAUSTED': NewEvent(),
    'CORE_ERROR': NewEvent(),
    'CORE_FINISHED': NewEvent()
}
