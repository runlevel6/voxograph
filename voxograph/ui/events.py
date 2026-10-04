"""voxograph.ui.events - wx event table and shared layout constants."""
from wx.lib.newevent import NewEvent


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

border = 5

# Long titles/authors wrap at this width so the Book Details box (and with it
# the whole right column) stays compact instead of growing to the text length.
BOOK_DETAILS_WRAP_WIDTH = 260
