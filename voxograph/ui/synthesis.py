"""voxograph.ui.synthesis - MainWindow SynthesisMixin methods."""
from voxograph.ui.core_thread import CoreThread
import platform
import subprocess
import threading
import wx


class SynthesisMixin:
    def on_start(self, event):
        self.synthesis_in_progress = True
        file_path = self.selected_file_path
        voice = self.get_selected_voice()
        speed = self.get_selected_speed()
        selected_chapters = [c for c in self.document_chapters if c.is_selected]

        if not selected_chapters:
            wx.MessageBox("No chapters selected. Please select at least one chapter.",
                          "No Chapters Selected", wx.OK | wx.ICON_WARNING)
            self.synthesis_in_progress = False
            return

        if not file_path:
            wx.MessageBox("No EPUB file loaded. Please open an EPUB file first.",
                          "No File Loaded", wx.OK | wx.ICON_WARNING)
            self.synthesis_in_progress = False
            return

        self.start_button.Disable()
        self.cancel_button.Show()
        self._relayout_right_panel()
        self.params_panel.Disable()
        self.table.EnableCheckBoxes(False)

        self.stop_event = threading.Event()
        engine = self.get_selected_tts_engine()
        print('Starting Audiobook Synthesis', dict(
            file_path=file_path, voice=voice, speed=speed, engine=engine,
            model=self.get_chatterbox_model(),
            exaggeration=self.get_chatterbox_exaggeration(),
            cfg_weight=self.get_chatterbox_cfg_weight(),
            turbo_temperature=self.get_chatterbox_turbo_temperature(),
            turbo_top_p=self.get_chatterbox_turbo_top_p(),
            turbo_top_k=self.get_chatterbox_turbo_top_k(),
            turbo_repetition_penalty=self.get_chatterbox_turbo_repetition_penalty()))
        self.core_thread = CoreThread(
            params=dict(file_path=file_path, voice=voice, pick_manually=False, speed=speed,
                        output_folder=self.output_folder_text_ctrl.GetValue(),
                        selected_chapters=selected_chapters,
                        tts_engine=engine),
            stop_event=self.stop_event)
        self.core_thread.start()
    def on_cancel(self, event):
        self.cancel_current_synthesis()
    def cancel_current_synthesis(self):
        """Stop the current run: fire stop_event and flag the UI as stopped.

        Shared by the "⛔ Cancel Synthesis" button and the Cancel button on the
        AI-service-unavailable popup (on_core_ai_retry_exhausted).
        """
        if self.stop_event:
            self.stop_event.set()
        self.cancel_button.Disable()
        self.cancel_button.SetLabel("⏳ Stopping…")
        self.synthesis_in_progress = False
    def set_table_chapter_status(self, chapter_index, status):
        self.table.SetItem(chapter_index, 3, status)
    def open_folder_with_explorer(self, folder_path):
        try:
            if platform.system() == 'Windows':
                subprocess.Popen(['explorer', folder_path])
            elif platform.system() == 'Linux':
                subprocess.Popen(['xdg-open', folder_path])
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', folder_path])
        except Exception as e:
            print(e)
