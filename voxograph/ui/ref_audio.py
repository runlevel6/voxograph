"""voxograph.ui.ref_audio - MainWindow RefAudioMixin methods."""
import os
import wx
from voxograph.core import (
    resolve_chatterbox_ref_audio,
    voice_sample_exists,
    voice_sample_info,
)


class RefAudioMixin:
    def _update_engine_ui(self):
        chatterbox = bool(self.chatterbox_radio.GetValue())
        turbo = chatterbox and self.get_chatterbox_model() == 'turbo'
        # The voice dropdown and compute device stay visible for both engines:
        # the dropdown picks the Kokoro voice *and* the Chatterbox voice preset,
        # and the device drives Kokoro sample building plus Chatterbox itself.
        self.compute_label.Show()
        self.compute_radio_panel.Show()
        self.voice_label.Show()
        self.voice_dropdown.Show()

        # The model selector only applies to the Chatterbox engine.
        self.model_label.Show(chatterbox)
        self.model_dropdown.Show(chatterbox)

        for widget in (self.voice_source_label, self.voice_source_radio_panel,
                       self.ref_audio_label, self.ref_audio_text_input):
            widget.Show(chatterbox)

        # Chatterbox has no speed control: the Kokoro speed row is swapped for
        # the selected model's own knobs in its place. Turbo ignores
        # exaggeration/CFG weight, so it shows its sampling knobs instead of
        # showing dead controls.
        self.speed_label.Show(not chatterbox)
        self.speed_text_input.Show(not chatterbox)
        for widget in (self.exaggeration_label, self.exaggeration_spin,
                       self.cfg_weight_label, self.cfg_weight_spin):
            widget.Show(chatterbox and not turbo)
        for widget in (self.turbo_temperature_label, self.turbo_temperature_spin,
                       self.turbo_top_p_label, self.turbo_top_p_spin,
                       self.turbo_top_k_label, self.turbo_top_k_spin,
                       self.turbo_repetition_penalty_label,
                       self.turbo_repetition_penalty_spin):
            widget.Show(turbo)

        # Preset mode builds samples; custom mode manages a picked WAV.
        custom = chatterbox and self.get_preset_source() == 'custom'
        self.ref_audio_button.Show(custom)
        self.clear_ref_audio_button.Show(custom)
        self.build_sample_button.Show(chatterbox and not custom)
        self.build_all_button.Show(chatterbox and not custom)
        self._sync_ref_audio_hint()
        self.params_panel.Layout()
        right_panel = getattr(self, 'right_panel', None)
        if hasattr(right_panel, 'SetupScrolling'):
            right_panel.SetupScrolling(scroll_x=False, scroll_y=True)
    def _on_tts_engine_changed(self, engine):
        self._update_engine_ui()
        self.save_current_settings()
    def on_voice_source_changed(self, source):
        self._update_engine_ui()
        self._sync_ref_audio_text()
        self.save_current_settings()
    def get_preset_source(self):
        """'preset' (clone the selected Kokoro voice) or 'custom' (own WAV)."""
        if getattr(self, 'custom_radio', None) and self.custom_radio.GetValue():
            return 'custom'
        return 'preset'
    def open_ref_audio_dialog(self, event=None):
        """Show the picker and return the chosen path, or '' if cancelled."""
        with wx.FileDialog(self, "Open Reference Audio", wildcard="WAV files (*.wav)|*.wav|All files (*.*)|*.*",
                           defaultDir=self.settings.get('last_open_dir') or os.path.expanduser('~'),
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return ''
            path = dialog.GetPath()
            self.selected_ref_audio = path
            self.ref_audio_text_input.SetValue(path)
            self._sync_ref_audio_hint()
            self.save_current_settings()
            return path
    def on_select_ref_audio(self, event):
        self.selected_ref_audio = self.ref_audio_text_input.GetValue()
        self.save_current_settings()
    def clear_ref_audio(self, event=None):
        """Forget the configured Chatterbox reference audio."""
        self.selected_ref_audio = ''
        if getattr(self, 'ref_audio_text_input', None):
            self.ref_audio_text_input.SetValue('')
        self._sync_ref_audio_hint()
        self.save_current_settings()
    def _sync_ref_audio_text(self):
        """Show whichever source is actually in effect in the read-only field."""
        ctrl = getattr(self, 'ref_audio_text_input', None)
        if ctrl is None:
            return
        path = self.get_ref_audio()
        stored = getattr(self, 'selected_ref_audio', '')
        if path:
            ctrl.SetValue(path)
        elif self.get_preset_source() == 'preset':
            ctrl.SetValue('')          # no sample built yet for this voice
        else:
            ctrl.SetValue(stored if os.path.exists(stored) else '')
    def _sync_ref_audio_hint(self):
        """Report where the clone source comes from, or what is still missing."""
        hint = getattr(self, 'ref_audio_hint', None)
        if hint is None:
            return
        chatterbox = bool(getattr(self, 'chatterbox_radio', None) and self.chatterbox_radio.GetValue())
        hint.Show(chatterbox)
        if not chatterbox:
            return

        if self.get_preset_source() == 'custom':
            stored = getattr(self, 'selected_ref_audio', '') or self.settings.get('chatterbox_ref_audio', '')
            if not stored:
                hint.SetLabel("⚠ Pick a WAV to clone a voice from.")
                hint.SetForegroundColour(wx.Colour(200, 90, 0))
            elif not os.path.exists(stored):
                hint.SetLabel("⚠ Configured file is gone — select a new one.")
                hint.SetForegroundColour(wx.Colour(200, 90, 0))
            else:
                hint.SetLabel(f"✓ Cloning: {os.path.basename(stored)}")
                hint.SetForegroundColour(wx.Colour(0, 120, 0))
            return

        voice = self.get_selected_voice()
        if voice_sample_exists(voice):
            info = voice_sample_info(voice)
            duration = info.get('duration_sec')
            hint.SetLabel(f"✓ Preset: {voice}" + (f" ({duration}s)" if duration else ''))
            hint.SetForegroundColour(wx.Colour(0, 120, 0))
        else:
            hint.SetLabel(f"⚠ No sample for {voice} — press 🎙 Build Sample.")
            hint.SetForegroundColour(wx.Colour(200, 90, 0))
    def get_ref_audio(self):
        """Resolve the WAV Chatterbox should clone (voice preset or custom clip)."""
        return resolve_chatterbox_ref_audio(
            self.get_selected_voice(),
            source=self.get_preset_source(),
            custom_path=getattr(self, 'selected_ref_audio', '') or self.settings.get('chatterbox_ref_audio', ''),
        )
    def get_selected_tts_engine(self):
        return 'chatterbox' if self.chatterbox_radio.GetValue() else 'kokoro'
    def get_chatterbox_device(self):
        return 'cuda' if self.cuda_radio.GetValue() else 'cpu'
    def get_chatterbox_exaggeration(self):
        return float(self.exaggeration_spin.GetValue())
    def get_chatterbox_cfg_weight(self):
        return float(self.cfg_weight_spin.GetValue())
    def get_chatterbox_turbo_temperature(self):
        return float(self.turbo_temperature_spin.GetValue())
    def get_chatterbox_turbo_top_p(self):
        return float(self.turbo_top_p_spin.GetValue())
    def get_chatterbox_turbo_top_k(self):
        return int(self.turbo_top_k_spin.GetValue())
    def get_chatterbox_turbo_repetition_penalty(self):
        return float(self.turbo_repetition_penalty_spin.GetValue())
    def get_chatterbox_model(self):
        """'turbo' or 'multilingual' (default) per the Model dropdown."""
        dropdown = getattr(self, 'model_dropdown', None)
        if dropdown is not None:
            return 'turbo' if dropdown.GetValue().startswith('Turbo') else 'multilingual'
        return self.settings.get('chatterbox_model', 'multilingual')
    def on_select_chatterbox_model(self, event):
        """Switching model swaps which style rows are in effect, then persists."""
        self._update_engine_ui()
        self.save_current_settings()
