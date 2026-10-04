"""voxograph.ui.window - the composed MainWindow frame."""
from voxograph.ui.menu import MenuMixin
from voxograph.ui.layout import LayoutMixin
from voxograph.ui.params import ParamsMixin
from voxograph.ui.ref_audio import RefAudioMixin
from voxograph.ui.voice_samples import VoiceSampleMixin
from voxograph.ui.settings import SettingsMixin
from voxograph.ui.ebook import EbookMixin
from voxograph.ui.preview import PreviewMixin
from voxograph.ui.synthesis import SynthesisMixin
from voxograph.ui.core_events import CoreEventsMixin
from voxograph.ui.events import EVENTS
from pathlib import Path
import wx
from voxograph.core import (
    load_settings,
)


class MainWindow(
    MenuMixin,
    LayoutMixin,
    ParamsMixin,
    RefAudioMixin,
    VoiceSampleMixin,
    SettingsMixin,
    EbookMixin,
    PreviewMixin,
    SynthesisMixin,
    CoreEventsMixin,
    wx.Frame,
):
    def __init__(self, parent, title):
        screen_width, screen_h = wx.GetDisplaySize()
        self.window_width = int(screen_width * 0.6)
        super().__init__(parent, title=title, size=(self.window_width, self.window_width * 3 // 4))
        self.chapters_panel = None
        self.preview_threads = []
        self.selected_chapter = None
        self.selected_book = None
        self.synthesis_in_progress = False
        self.stop_event = None

        self.Bind(EVENTS['CORE_STARTED'][1], self.on_core_started)
        self.Bind(EVENTS['CORE_CHAPTER_STARTED'][1], self.on_core_chapter_started)
        self.Bind(EVENTS['CORE_CHAPTER_FINISHED'][1], self.on_core_chapter_finished)
        self.Bind(EVENTS['CORE_PROGRESS'][1], self.on_core_progress)
        self.Bind(EVENTS['CORE_AI_REWRITE'][1], self.on_core_ai_rewrite)
        self.Bind(EVENTS['CORE_AI_RETRY_EXHAUSTED'][1], self.on_core_ai_retry_exhausted)
        self.Bind(EVENTS['CORE_ERROR'][1], self.on_core_error)
        self.Bind(EVENTS['CORE_FINISHED'][1], self.on_core_finished)

        self.settings = load_settings()
        self.last_open_dir = self.settings.get('last_open_dir', str(Path.home()))

        self.create_menu()
        self.create_layout()
        self.Centre()
        self.Show(True)
