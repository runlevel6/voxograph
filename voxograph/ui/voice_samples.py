"""voxograph.ui.voice_samples - MainWindow VoiceSampleMixin methods."""
import threading
import wx
from voxograph.core import (
    generate_voice_sample,
    generate_voice_samples,
    missing_voice_samples,
)
from voxograph.voices import voices


class VoiceSampleMixin:
    def on_build_voice_sample(self, event=None, resume=False):
        """Render the selected Kokoro voice into a Chatterbox preset clip."""
        button = event.GetEventObject() if event is not None else getattr(self, 'build_sample_button', None)
        voice = self.get_selected_voice()

        def finish(ok, message=''):
            def do_finish():
                if button:
                    button.SetLabel("🎙 Build Sample")
                    button.Enable()
                self._sync_ref_audio_text()
                self._sync_ref_audio_hint()
                if not ok:
                    wx.MessageBox(message, "Voice Sample Error", wx.OK | wx.ICON_ERROR)
                elif resume:
                    self._resume_preview()
            wx.CallAfter(do_finish)

        if button:
            button.SetLabel("⏳")
            button.Disable()

        def build():
            try:
                path = generate_voice_sample(voice, overwrite=True)
                finish(bool(path), '' if path else f"Kokoro produced no audio for {voice}.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                finish(False, str(e))

        # fix #8: background thread so the GUI stays responsive while Kokoro runs.
        thread = threading.Thread(target=build, daemon=True)
        thread.start()
        self.preview_threads.append(thread)
        self.preview_threads = [t for t in self.preview_threads if t.is_alive()]
    def on_build_all_samples(self, event=None):
        """Render every Kokoro voice in the dropdown into a preset clip."""
        all_voices = [v for lang_voices in voices.values() for v in lang_voices]
        pending = missing_voice_samples(all_voices)
        if not pending:
            wx.MessageBox(f"All {len(all_voices)} voice presets are already built.",
                          "Voice Samples", wx.OK | wx.ICON_INFORMATION)
            return
        if wx.MessageBox(f"Build {len(pending)} voice sample(s)? Kokoro renders each clip on the "
                         f"selected device, so this can take a few minutes.",
                         "Build Voice Samples", wx.YES_NO | wx.ICON_QUESTION) != wx.YES:
            return

        stop_event = threading.Event()
        dialog = wx.ProgressDialog("Building voice samples", "Starting…",
                                   maximum=len(pending), parent=self,
                                   style=wx.PD_APP_MODAL | wx.PD_CAN_ABORT | wx.PD_ELAPSED_TIME)
        state = {'done': False, 'alive': True, 'error': ''}

        def report(voice, path, done, total):
            # Guarded: the user may cancel, after which the dialog is destroyed
            # while this worker is still finishing the current voice.
            if state['alive']:
                wx.CallAfter(dialog.Update, done, f"{voice} ({done}/{total})")

        def build_all():
            try:
                generate_voice_samples(pending, stop_event=stop_event, progress=report)
            except Exception as e:
                import traceback
                traceback.print_exc()
                state['error'] = str(e)
            finally:
                state['done'] = True
                if state['alive']:
                    wx.CallAfter(dialog.Update, len(pending), "Done")
                    wx.CallAfter(dialog.EndModal, wx.ID_OK)

        threading.Thread(target=build_all, daemon=True).start()
        try:
            while not state['done']:
                # Nested event loop: renders the CallAfter updates and lets the
                # user press Cancel, which returns without state['done'] set.
                dialog.ShowModal()
                if not state['done']:
                    stop_event.set()
                    print('Voice sample build cancelled by user.')
                    break
        finally:
            state['alive'] = False
            dialog.Destroy()
        if state['error']:
            wx.MessageBox(f"Error building voice samples: {state['error']}",
                          "Voice Sample Error", wx.OK | wx.ICON_ERROR)
        self._sync_ref_audio_text()
        self._sync_ref_audio_hint()
    def prompt_for_ref_audio(self, missing_path=''):
        """UI-thread handler: resolve a missing clone source and resume the preview."""
        if self.get_preset_source() == 'preset':
            voice = self.get_selected_voice()
            message = (f"No Chatterbox voice preset has been built for {voice} yet.\n\n"
                       f"Build one now with Kokoro? It takes a few seconds and is then "
                       f"reused every time this voice is selected.")
        elif missing_path:
            message = (f"The saved Chatterbox reference audio no longer exists:\n\n{missing_path}\n\n"
                       "Pick a replacement WAV now?")
        else:
            message = ("Chatterbox clones a voice from a reference audio clip, so one is required.\n\n"
                       "Pick a WAV file now?")
        dialog = wx.MessageDialog(self, message, "Missing Reference Audio",
                                  wx.YES_NO | wx.ICON_QUESTION)
        try:
            choice = dialog.ShowModal()
        finally:
            dialog.Destroy()
        if choice != wx.ID_YES:
            return
        if self.get_preset_source() == 'preset':
            self.on_build_voice_sample(resume=True)
        else:
            self.open_ref_audio_dialog()
            if self.get_ref_audio():
                self._resume_preview()
    def _resume_preview(self):
        """Re-run the preview that was interrupted by the reference audio prompt."""
        if getattr(self, 'preview_button', None) and self.selected_chapter:
            self.on_preview_chapter(None)
