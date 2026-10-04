"""voxograph.ui.preview - MainWindow PreviewMixin methods."""
from tempfile import NamedTemporaryFile
from kokoro import KPipeline
from voxograph.ui.bridge import _extract_bridge_json
import json
import numpy as np
import os
import soundfile
import subprocess
import threading
import wx
from voxograph.core import (
    preview_excerpt,
)


class PreviewMixin:
    def on_preview_chapter(self, event=None):
        if not self.selected_chapter:
            wx.MessageBox("No chapter selected for preview.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        engine = self.get_selected_tts_engine()
        button = event.GetEventObject() if event is not None else getattr(self, 'preview_button', None)
        if button is None:
            return
        ai_enabled = self.ai_enabled_checkbox.GetValue()
        api_key = self.ai_api_key_text_ctrl.GetValue().strip() if ai_enabled else ''
        model = self.ai_model_dropdown.GetValue()

        if ai_enabled and not api_key:
            wx.MessageBox("AI phonetic check is enabled but the Gemini API key is missing. "
                          "Disable AI or enter a key in the settings panel.",
                          "Missing API Key", wx.OK | wx.ICON_INFORMATION)
            return

        # preview only needs a short sample, but never a blind slice mid-word:
        # preview_excerpt() returns whole sentences up to the chunk ceiling.
        text = preview_excerpt(self.selected_chapter.extracted_text)
        if not text.strip():
            wx.MessageBox("Selected chapter has no text for preview.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        self._preview_seq = getattr(self, '_preview_seq', 0) + 1
        preview_seq = self._preview_seq

        def reset_button():
            def do_reset():
                if preview_seq != getattr(self, '_preview_seq', 0):
                    return
                button.SetLabel("🔊 Preview")
                button.Enable()
            wx.CallAfter(do_reset)

        button.SetLabel("⏳")
        button.Disable()

        def generate_preview():
            import voxograph.core as core
            try:
                preview_text = text
                if ai_enabled:
                    corrected = core.correct_phonetics_ai(
                        text=preview_text, api_key=api_key, model=model,
                        tts_engine=engine,
                        chatterbox_model=self.get_chatterbox_model())
                    if corrected and corrected.strip() and corrected != preview_text:
                        preview_text = corrected

                with NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name

                if engine == 'chatterbox':
                    ref_audio = self.get_ref_audio()
                    if not ref_audio:
                        # Nothing to clone from yet: prompt_for_ref_audio builds
                        # the voice preset (or asks for a WAV), then resumes.
                        wx.CallAfter(self.prompt_for_ref_audio)
                        return
                    device = self.get_chatterbox_device()
                    payload = json.dumps(core.chatterbox_bridge_request(
                        text=preview_text,
                        output_path=tmp_path,
                        device=device,
                        ref_audio=ref_audio,
                        model=self.get_chatterbox_model(),
                        exaggeration=self.get_chatterbox_exaggeration(),
                        cfg_weight=self.get_chatterbox_cfg_weight(),
                        turbo_temperature=self.get_chatterbox_turbo_temperature(),
                        turbo_top_p=self.get_chatterbox_turbo_top_p(),
                        turbo_top_k=self.get_chatterbox_turbo_top_k(),
                        turbo_repetition_penalty=self.get_chatterbox_turbo_repetition_penalty(),
                    ))
                    proc = subprocess.run(
                        [str(core.CHATTERBOX_BRIDGE_PYTHON), str(core.CHATTERBOX_BRIDGE_SCRIPT)],
                        input=payload, capture_output=True, text=True
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(f"Chatterbox bridge failed: {proc.stderr}")
                    result = _extract_bridge_json(proc.stdout)
                    if not result.get('success'):
                        raise RuntimeError(result.get('error', 'Unknown error'))
                else:
                    pipeline = KPipeline(lang_code=core.lang_code_from_voice(self.get_selected_voice()))
                    audio_segments = core.gen_audio_segments(
                        pipeline, preview_text, voice=self.get_selected_voice(), speed=self.get_selected_speed())

                    if not audio_segments:
                        wx.CallAfter(wx.MessageBox, "Could not generate audio for preview.", "Error", wx.OK | wx.ICON_ERROR)
                        return

                    final_audio = np.concatenate(audio_segments)
                    soundfile.write(tmp_path, final_audio, core.sample_rate)

                subprocess.run(['ffplay', '-autoexit', '-nodisp', tmp_path])
                os.remove(tmp_path)

            except Exception as e:
                wx.CallAfter(wx.MessageBox, f"Error during preview: {e}", "Preview Error", wx.OK | wx.ICON_ERROR)
                import traceback
                traceback.print_exc()
            finally:
                reset_button()

        # fix #8: don't join on the UI thread — use daemon threads and just
        # let previous previews finish in the background.  The finally block
        # above re-enables the button when each thread ends naturally.
        thread = threading.Thread(target=generate_preview, daemon=True)
        thread.start()
        self.preview_threads.append(thread)
        # Prune dead threads from the list so it doesn't grow forever
        self.preview_threads = [t for t in self.preview_threads if t.is_alive()]
    def on_check_phonetic_ai(self, event):
        if not self.selected_chapter:
            wx.MessageBox("No chapter selected for AI check.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        if not self.ai_enabled_checkbox.GetValue():
            wx.MessageBox("AI phonetic check is not enabled. Please enable it in the settings panel.",
                          "AI Not Enabled", wx.OK | wx.ICON_INFORMATION)
            return

        api_key = self.ai_api_key_text_ctrl.GetValue().strip()
        if not api_key:
            wx.MessageBox("Gemini API key is missing. Please enter it in the settings panel.",
                          "Missing API Key", wx.OK | wx.ICON_INFORMATION)
            return

        text = self.selected_chapter.extracted_text.strip()
        if not text:
            wx.MessageBox("Selected chapter has no text for AI analysis.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        model = self.ai_model_dropdown.GetValue()
        tts_engine = self.get_selected_tts_engine()

        button = event.GetEventObject()
        button.SetLabel("⏳")
        button.Disable()

        def run_ai_check():
            import voxograph.core as core
            try:
                result = core.check_phonetic_transcription_ai(
                    text=text,
                    api_key=api_key,
                    model=model,
                    tts_engine=tts_engine,
                    chatterbox_model=self.get_chatterbox_model()
                )
                wx.CallAfter(self.show_ai_result_dialog, result)
            except Exception as e:
                wx.CallAfter(wx.MessageBox, f"Error during AI check: {e}", "AI Error", wx.OK | wx.ICON_ERROR)
                import traceback
                traceback.print_exc()
            finally:
                wx.CallAfter(button.SetLabel, "🤖 Check with AI")
                wx.CallAfter(button.Enable)

        thread = threading.Thread(target=run_ai_check, daemon=True)
        thread.start()
    def show_ai_result_dialog(self, result_text):
        dialog = wx.Dialog(self, title="AI Phonetic Transcription Analysis", size=(700, 500))
        sizer = wx.BoxSizer(wx.VERTICAL)

        text_ctrl = wx.TextCtrl(dialog, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP)
        text_ctrl.SetValue(result_text)

        close_button = wx.Button(dialog, label="Close")
        close_button.Bind(wx.EVT_BUTTON, lambda event: dialog.EndModal(wx.ID_OK))

        sizer.Add(text_ctrl, 1, wx.ALL | wx.EXPAND, 10)
        sizer.Add(close_button, 0, wx.ALL | wx.CENTER, 10)

        dialog.SetSizer(sizer)
        dialog.ShowModal()
        dialog.Destroy()
