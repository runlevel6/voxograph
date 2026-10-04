"""voxograph.ui.params - MainWindow ParamsMixin methods."""
from voxograph.ui.events import border
import os
import torch.cuda
import wx
from voxograph.core import (
    DEFAULT_VOICE,
)
from voxograph.core.constants import (
    CHATTERBOX_TURBO_DEFAULT_TEMPERATURE, CHATTERBOX_TURBO_DEFAULT_TOP_P,
    CHATTERBOX_TURBO_DEFAULT_TOP_K, CHATTERBOX_TURBO_DEFAULT_REPETITION_PENALTY,
    CHATTERBOX_TURBO_TEMPERATURE_RANGE, CHATTERBOX_TURBO_TOP_P_RANGE,
    CHATTERBOX_TURBO_TOP_K_RANGE, CHATTERBOX_TURBO_REPETITION_PENALTY_RANGE,
)
from voxograph.voices import flags, voices


class ParamsMixin:
    def create_params_panel(self):
        panel_box = wx.Panel(self.right_panel, style=wx.SUNKEN_BORDER)
        panel_box_sizer = wx.StaticBoxSizer(wx.VERTICAL, panel_box, "Audiobook Parameters")
        panel_box.SetSizer(panel_box_sizer)

        panel = self.params_panel = wx.Panel(panel_box)
        panel_box_sizer.Add(panel, 1, wx.ALL | wx.EXPAND, 5)
        self.right_sizer.Add(panel_box, 1, wx.ALL | wx.EXPAND, 5)
        sizer = wx.GridBagSizer(10, 10)
        panel.SetSizer(sizer)

        tts_engine_label = wx.StaticText(panel, label="TTS Engine:")
        tts_engine_radio_panel = wx.Panel(panel)
        kokoro_radio = wx.RadioButton(tts_engine_radio_panel, label="Kokoro", style=wx.RB_GROUP)
        chatterbox_radio = wx.RadioButton(tts_engine_radio_panel, label="Chatterbox")
        initial_engine = self.settings.get('tts_engine', 'kokoro')
        if initial_engine == 'chatterbox':
            chatterbox_radio.SetValue(True)
        else:
            kokoro_radio.SetValue(True)
        sizer.Add(tts_engine_label, pos=(0, 0), flag=wx.ALL, border=border)
        tts_engine_radio_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        tts_engine_radio_panel.SetSizer(tts_engine_radio_panel_sizer)
        tts_engine_radio_panel_sizer.Add(kokoro_radio, 0, wx.ALL, 5)
        tts_engine_radio_panel_sizer.Add(chatterbox_radio, 0, wx.ALL, 5)
        kokoro_radio.Bind(wx.EVT_RADIOBUTTON, lambda event: self._on_tts_engine_changed('kokoro'))
        chatterbox_radio.Bind(wx.EVT_RADIOBUTTON, lambda event: self._on_tts_engine_changed('chatterbox'))
        self.kokoro_radio = kokoro_radio
        self.chatterbox_radio = chatterbox_radio

        # Chatterbox ships as a family: Multilingual V3 (default, highest
        # quality) and Turbo (smaller/faster, English-only, native
        # paralinguistic tags). Both clone the selected voice from a reference
        # WAV, so they share all the Chatterbox rows below.
        model_label = wx.StaticText(panel, label="Model:")
        model_dropdown = wx.ComboBox(
            panel, choices=['Multilingual V3', 'Turbo'],
            value=('Turbo' if self.settings.get('chatterbox_model', 'multilingual') == 'turbo'
                   else 'Multilingual V3'),
            style=wx.CB_READONLY)
        model_dropdown.SetToolTip(
            "Which Chatterbox model to synthesize with:\n"
            "Multilingual V3 — 500M, highest quality, English-locked here, "
            "shaped by Exaggeration and CFG Weight.\n"
            "Turbo — 350M, faster and lighter, English-only, supports "
            "paralinguistic tags like [laugh] and [cough]; it ignores "
            "Exaggeration/CFG Weight and is shaped by its sampling knobs instead.")
        model_dropdown.Bind(wx.EVT_COMBOBOX, self.on_select_chatterbox_model)
        model_row = wx.BoxSizer(wx.HORIZONTAL)
        model_row.Add(model_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        model_row.Add(model_dropdown, 0, wx.ALIGN_CENTER_VERTICAL)
        engine_column = wx.BoxSizer(wx.VERTICAL)
        engine_column.Add(tts_engine_radio_panel, 0, wx.EXPAND)
        engine_column.Add(model_row, 0, wx.LEFT | wx.BOTTOM, 5)
        sizer.Add(engine_column, pos=(0, 1), flag=wx.ALL, border=border)
        self.model_label = model_label
        self.model_dropdown = model_dropdown

        compute_label = wx.StaticText(panel, label="Compute Device:")
        compute_radio_panel = wx.Panel(panel)
        cpu_radio = wx.RadioButton(compute_radio_panel, label="CPU", style=wx.RB_GROUP)
        cuda_radio = wx.RadioButton(compute_radio_panel, label="CUDA")
        if torch.cuda.is_available():
            cuda_radio.SetValue(True)
        else:
            cpu_radio.SetValue(True)
        sizer.Add(compute_label, pos=(1, 0), flag=wx.ALL, border=border)
        sizer.Add(compute_radio_panel, pos=(1, 1), flag=wx.ALL, border=border)
        compute_radio_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        compute_radio_panel.SetSizer(compute_radio_panel_sizer)
        compute_radio_panel_sizer.Add(cpu_radio, 0, wx.ALL, 5)
        compute_radio_panel_sizer.Add(cuda_radio, 0, wx.ALL, 5)
        cpu_radio.Bind(wx.EVT_RADIOBUTTON, lambda event: torch.set_default_device('cpu'))
        cuda_radio.Bind(wx.EVT_RADIOBUTTON, lambda event: torch.set_default_device('cuda'))
        self.cpu_radio = cpu_radio
        self.cuda_radio = cuda_radio
        self.compute_label = compute_label
        self.compute_radio_panel = compute_radio_panel

        flag_and_voice_list = []
        for code, l in voices.items():
            for v in l:
                flag_and_voice_list.append(f'{flags[code]} {v}')

        voice_label = wx.StaticText(panel, label="Voice:")
        default_voice_from_settings = self.settings.get('voice', DEFAULT_VOICE)
        initial_voice_display = next(
            (fv for fv in flag_and_voice_list if fv.endswith(default_voice_from_settings)),
            flag_and_voice_list[0] if flag_and_voice_list else ''
        )
        self.selected_voice = initial_voice_display
        voice_dropdown = wx.ComboBox(panel, choices=flag_and_voice_list, value=initial_voice_display)
        voice_dropdown.Bind(wx.EVT_COMBOBOX, self.on_select_voice)
        sizer.Add(voice_label, pos=(2, 0), flag=wx.ALL, border=border)
        sizer.Add(voice_dropdown, pos=(2, 1), flag=wx.ALL, border=border)
        self.voice_label = voice_label
        self.voice_dropdown = voice_dropdown

        voice_source_label = wx.StaticText(panel, label="Voice Source:")
        voice_source_radio_panel = wx.Panel(panel)
        preset_radio = wx.RadioButton(voice_source_radio_panel, label="🎤 Voice Preset", style=wx.RB_GROUP)
        custom_radio = wx.RadioButton(voice_source_radio_panel, label="📁 Custom WAV")
        if self.settings.get('chatterbox_voice_source', 'preset') == 'custom':
            custom_radio.SetValue(True)
        else:
            preset_radio.SetValue(True)
        preset_radio.Bind(wx.EVT_RADIOBUTTON, lambda event: self.on_voice_source_changed('preset'))
        custom_radio.Bind(wx.EVT_RADIOBUTTON, lambda event: self.on_voice_source_changed('custom'))
        voice_source_radio_panel.SetToolTip(
            "Voice Preset: Chatterbox clones the selected Kokoro voice from a sample clip. "
            "Chatterbox Multilingual V3 is English-locked, so US/UK voices make the best presets.\n"
            "Custom WAV: clone any recording you supply.")
        sizer.Add(voice_source_label, pos=(3, 0), flag=wx.ALL, border=border)
        sizer.Add(voice_source_radio_panel, pos=(3, 1), flag=wx.ALL, border=border)
        voice_source_radio_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        voice_source_radio_panel.SetSizer(voice_source_radio_panel_sizer)
        voice_source_radio_panel_sizer.Add(preset_radio, 0, wx.ALL, 5)
        voice_source_radio_panel_sizer.Add(custom_radio, 0, wx.ALL, 5)
        self.voice_source_label = voice_source_label
        self.voice_source_radio_panel = voice_source_radio_panel
        self.preset_radio = preset_radio
        self.custom_radio = custom_radio

        stored_ref_audio = self.settings.get('chatterbox_ref_audio', '')
        self.selected_ref_audio = stored_ref_audio
        ref_audio_label = wx.StaticText(panel, label="Reference Audio:")
        ref_audio_text_input = wx.TextCtrl(panel, value=stored_ref_audio, style=wx.TE_READONLY)
        ref_audio_text_input.Bind(wx.EVT_TEXT, self.on_select_ref_audio)
        ref_audio_button = wx.Button(panel, label="📂 Select")
        ref_audio_button.Bind(wx.EVT_BUTTON, self.open_ref_audio_dialog)
        clear_ref_audio_button = wx.Button(panel, label="✕ Clear")
        clear_ref_audio_button.Bind(wx.EVT_BUTTON, self.clear_ref_audio)
        build_sample_button = wx.Button(panel, label="🎙 Build Sample")
        build_sample_button.Bind(wx.EVT_BUTTON, self.on_build_voice_sample)
        self.build_sample_button = build_sample_button
        build_all_button = wx.Button(panel, label="Build All")
        build_all_button.Bind(wx.EVT_BUTTON, self.on_build_all_samples)
        self.build_all_button = build_all_button
        ref_button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ref_button_sizer.Add(ref_audio_button, 0, wx.RIGHT | wx.ALL, 5)
        ref_button_sizer.Add(clear_ref_audio_button, 0, wx.RIGHT | wx.ALL, 5)
        ref_button_sizer.Add(build_sample_button, 0, wx.RIGHT | wx.ALL, 5)
        ref_button_sizer.Add(build_all_button, 0, wx.ALL, 5)
        ref_hint = wx.StaticText(panel, label="")
        sizer.Add(ref_audio_label, pos=(4, 0), flag=wx.ALL, border=border)
        sizer.Add(ref_audio_text_input, pos=(4, 1), flag=wx.ALL | wx.EXPAND, border=border)
        sizer.Add(ref_button_sizer, pos=(5, 1), flag=wx.ALL, border=border)
        sizer.Add(ref_hint, pos=(5, 2), flag=wx.ALL, border=border)
        self.ref_audio_label = ref_audio_label
        self.ref_audio_text_input = ref_audio_text_input
        self.ref_audio_button = ref_audio_button
        self.clear_ref_audio_button = clear_ref_audio_button
        self.ref_audio_hint = ref_hint
        self._sync_ref_audio_hint()

        # Kokoro-only: playback rate of the generated audio.
        speed_label = wx.StaticText(panel, label="Speed:")
        speed_text_input = wx.TextCtrl(panel, value=str(self.settings.get('speed', 1.0)))
        self.selected_speed = float(speed_text_input.GetValue())
        speed_text_input.Bind(wx.EVT_TEXT, self.on_select_speed)
        self.speed_label = speed_label
        self.speed_text_input = speed_text_input
        sizer.Add(speed_label, pos=(6, 0), flag=wx.ALL, border=border)
        sizer.Add(speed_text_input, pos=(6, 1), flag=wx.ALL, border=border)

        # Chatterbox-only: Chatterbox has no speed control, so these two style
        # knobs replace it (row 7/8) and are shown in its place.
        exaggeration_label = wx.StaticText(panel, label="Exaggeration:")
        exaggeration_spin = wx.SpinCtrlDouble(
            panel, min=0.0, max=1.0, inc=0.05,
            value=str(float(self.settings.get('chatterbox_exaggeration', 0.5))))
        exaggeration_spin.SetDigits(2)
        exaggeration_spin.SetToolTip(
            "How expressive/dramatic the speech is (Chatterbox only).\n"
            "Default 0.5. Use 0.7+ for dramatic or excited narration — higher "
            "values also speed the pacing up.")
        exaggeration_spin.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_select_exaggeration)
        self.exaggeration_label = exaggeration_label
        self.exaggeration_spin = exaggeration_spin
        sizer.Add(exaggeration_label, pos=(7, 0), flag=wx.ALL, border=border)
        sizer.Add(exaggeration_spin, pos=(7, 1), flag=wx.ALL, border=border)

        cfg_weight_label = wx.StaticText(panel, label="CFG Weight:")
        cfg_weight_spin = wx.SpinCtrlDouble(
            panel, min=0.0, max=1.0, inc=0.05,
            value=str(float(self.settings.get('chatterbox_cfg_weight', 0.5))))
        cfg_weight_spin.SetDigits(2)
        cfg_weight_spin.SetToolTip(
            "How closely the output follows the style and pacing of the "
            "reference clip (Chatterbox only).\n"
            "Default 0.5. Use 0.3 when the reference speaker talks too fast — it "
            "slows down and clarifies the pacing. Use 0 to stop accent bleeding "
            "in cross-lingual voice transfers.")
        cfg_weight_spin.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_select_cfg_weight)
        self.cfg_weight_label = cfg_weight_label
        self.cfg_weight_spin = cfg_weight_spin
        sizer.Add(cfg_weight_label, pos=(8, 0), flag=wx.ALL, border=border)
        sizer.Add(cfg_weight_spin, pos=(8, 1), flag=wx.ALL, border=border)

        # Chatterbox Turbo ignores exaggeration/CFG weight, so it gets the
        # sampling knobs it actually honors instead (rows 9-12). Each model keeps
        # its own values in the config, so switching models preserves both sets.
        turbo_temperature_label = wx.StaticText(panel, label="Temperature:")
        turbo_temperature_spin = wx.SpinCtrlDouble(
            panel, min=CHATTERBOX_TURBO_TEMPERATURE_RANGE[0],
            max=CHATTERBOX_TURBO_TEMPERATURE_RANGE[1], inc=0.05,
            value=str(float(self.settings.get('chatterbox_turbo_temperature',
                                              CHATTERBOX_TURBO_DEFAULT_TEMPERATURE))))
        turbo_temperature_spin.SetDigits(2)
        turbo_temperature_spin.SetToolTip(
            "Sampling temperature for Chatterbox Turbo only.\n"
            f"Default {CHATTERBOX_TURBO_DEFAULT_TEMPERATURE}. Lower is more "
            "consistent and repeatable; higher is more varied but sloppier.")
        turbo_temperature_spin.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_select_turbo_temperature)
        self.turbo_temperature_label = turbo_temperature_label
        self.turbo_temperature_spin = turbo_temperature_spin
        sizer.Add(turbo_temperature_label, pos=(9, 0), flag=wx.ALL, border=border)
        sizer.Add(turbo_temperature_spin, pos=(9, 1), flag=wx.ALL, border=border)

        turbo_top_p_label = wx.StaticText(panel, label="Top P:")
        turbo_top_p_spin = wx.SpinCtrlDouble(
            panel, min=CHATTERBOX_TURBO_TOP_P_RANGE[0],
            max=CHATTERBOX_TURBO_TOP_P_RANGE[1], inc=0.01,
            value=str(float(self.settings.get('chatterbox_turbo_top_p',
                                              CHATTERBOX_TURBO_DEFAULT_TOP_P))))
        turbo_top_p_spin.SetDigits(2)
        turbo_top_p_spin.SetToolTip(
            "Nucleus sampling cutoff for Chatterbox Turbo only.\n"
            f"Default {CHATTERBOX_TURBO_DEFAULT_TOP_P}. Lower narrows the "
            "candidate set and makes the delivery more predictable.")
        turbo_top_p_spin.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_select_turbo_top_p)
        self.turbo_top_p_label = turbo_top_p_label
        self.turbo_top_p_spin = turbo_top_p_spin
        sizer.Add(turbo_top_p_label, pos=(10, 0), flag=wx.ALL, border=border)
        sizer.Add(turbo_top_p_spin, pos=(10, 1), flag=wx.ALL, border=border)

        turbo_top_k_label = wx.StaticText(panel, label="Top K:")
        turbo_top_k_spin = wx.SpinCtrl(
            panel, min=CHATTERBOX_TURBO_TOP_K_RANGE[0],
            max=CHATTERBOX_TURBO_TOP_K_RANGE[1],
            value=str(int(self.settings.get('chatterbox_turbo_top_k',
                                          CHATTERBOX_TURBO_DEFAULT_TOP_K))))
        turbo_top_k_spin.SetToolTip(
            "Top-k sampling cutoff for Chatterbox Turbo only.\n"
            f"Default {CHATTERBOX_TURBO_DEFAULT_TOP_K}. Lower keeps only the most "
            "likely tokens; 0 disables top-k filtering entirely.")
        turbo_top_k_spin.Bind(wx.EVT_SPINCTRL, self.on_select_turbo_top_k)
        self.turbo_top_k_label = turbo_top_k_label
        self.turbo_top_k_spin = turbo_top_k_spin
        sizer.Add(turbo_top_k_label, pos=(11, 0), flag=wx.ALL, border=border)
        sizer.Add(turbo_top_k_spin, pos=(11, 1), flag=wx.ALL, border=border)

        turbo_repetition_penalty_label = wx.StaticText(panel, label="Repetition Penalty:")
        turbo_repetition_penalty_spin = wx.SpinCtrlDouble(
            panel, min=CHATTERBOX_TURBO_REPETITION_PENALTY_RANGE[0],
            max=CHATTERBOX_TURBO_REPETITION_PENALTY_RANGE[1], inc=0.05,
            value=str(float(self.settings.get('chatterbox_turbo_repetition_penalty',
                                              CHATTERBOX_TURBO_DEFAULT_REPETITION_PENALTY))))
        turbo_repetition_penalty_spin.SetDigits(2)
        turbo_repetition_penalty_spin.SetToolTip(
            "How strongly Chatterbox Turbo penalizes repeating a token it "
            "already used.\n"
            f"Default {CHATTERBOX_TURBO_DEFAULT_REPETITION_PENALTY}. Raise it to "
            "break up loops on long narration; 1.0 disables the penalty.")
        turbo_repetition_penalty_spin.Bind(
            wx.EVT_SPINCTRLDOUBLE, self.on_select_turbo_repetition_penalty)
        self.turbo_repetition_penalty_label = turbo_repetition_penalty_label
        self.turbo_repetition_penalty_spin = turbo_repetition_penalty_spin
        sizer.Add(turbo_repetition_penalty_label, pos=(12, 0), flag=wx.ALL, border=border)
        sizer.Add(turbo_repetition_penalty_spin, pos=(12, 1), flag=wx.ALL, border=border)

        ai_enabled_label = wx.StaticText(panel, label="AI Phonetic Check:")
        ai_enabled_checkbox = wx.CheckBox(panel, label="Enabled")
        ai_enabled_checkbox.SetValue(self.settings.get('gemini_enabled', False))
        ai_enabled_checkbox.Bind(wx.EVT_CHECKBOX, self.on_ai_enabled_changed)
        self.ai_enabled_checkbox = ai_enabled_checkbox
        sizer.Add(ai_enabled_label, pos=(13, 0), flag=wx.ALL, border=border)
        sizer.Add(ai_enabled_checkbox, pos=(13, 1), flag=wx.ALL, border=border)

        api_key_label = wx.StaticText(panel, label="Gemini API Key:")
        api_key_text_input = wx.TextCtrl(panel, value=self.settings.get('gemini_api_key', ''), style=wx.TE_PASSWORD)
        api_key_text_input.Bind(wx.EVT_TEXT, self.on_ai_api_key_changed)
        self.ai_api_key_text_ctrl = api_key_text_input
        sizer.Add(api_key_label, pos=(14, 0), flag=wx.ALL, border=border)
        sizer.Add(api_key_text_input, pos=(14, 1), flag=wx.ALL | wx.EXPAND, border=border)

        ai_model_label = wx.StaticText(panel, label="Gemini Model:")
        ai_model_choices = ['gemini-3.1-flash-lite', 'gemini-3.5-flash', 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-flash-lite-latest']
        ai_model_dropdown = wx.ComboBox(panel, choices=ai_model_choices, value=self.settings.get('gemini_model', 'gemini-3.1-flash-lite'))
        ai_model_dropdown.Bind(wx.EVT_COMBOBOX, self.on_ai_model_changed)
        self.ai_model_dropdown = ai_model_dropdown
        sizer.Add(ai_model_label, pos=(15, 0), flag=wx.ALL, border=border)
        sizer.Add(ai_model_dropdown, pos=(15, 1), flag=wx.ALL | wx.EXPAND, border=border)

        output_folder_label = wx.StaticText(panel, label="Output Folder:")
        initial_output_folder = self.settings.get('output_folder', os.path.abspath('.'))
        self.output_folder_text_ctrl = wx.TextCtrl(panel, value=initial_output_folder)
        self.output_folder_text_ctrl.SetEditable(False)
        output_folder_button = wx.Button(panel, label="📂 Select")
        output_folder_button.Bind(wx.EVT_BUTTON, self.open_output_folder_dialog)
        sizer.Add(output_folder_label, pos=(16, 0), flag=wx.ALL, border=border)
        sizer.Add(self.output_folder_text_ctrl, pos=(16, 1), flag=wx.ALL | wx.EXPAND, border=border)
        sizer.Add(output_folder_button, pos=(17, 1), flag=wx.ALL, border=border)

        self._update_engine_ui()
        self._sync_ref_audio_text()
    def open_output_folder_dialog(self, event):
        with wx.DirDialog(self, "Choose a directory:", style=wx.DD_DEFAULT_STYLE) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            output_folder = dialog.GetPath()
            print(f"Selected output folder: {output_folder}")
            self.output_folder_text_ctrl.SetValue(output_folder)
            self.save_current_settings()
    def on_select_voice(self, event):
        self.selected_voice = event.GetString()
        # The voice drives the Chatterbox preset, so refresh what it clones from.
        self._sync_ref_audio_text()
        self._sync_ref_audio_hint()
        self.save_current_settings()
    def on_select_speed(self, event):
        try:
            speed = float(event.GetString())
            if speed > 0:
                print('Selected speed', speed)
                self.selected_speed = speed
                self.save_current_settings()
            else:
                print("Speed must be a positive number.")
        except ValueError:
            print("Invalid speed value. Please enter a number.")
    def on_select_exaggeration(self, event):
        """Chatterbox emotion exaggeration (0-1; higher = more expressive)."""
        self.save_current_settings()
    def on_select_cfg_weight(self, event):
        """Chatterbox CFG weight (0-1; lower = looser pacing, 0 = no accent bleed)."""
        self.save_current_settings()
    def on_select_turbo_temperature(self, event):
        """Chatterbox Turbo sampling temperature (0-2; higher = more varied)."""
        self.save_current_settings()
    def on_select_turbo_top_p(self, event):
        """Chatterbox Turbo nucleus sampling cutoff (0-1; lower = tighter)."""
        self.save_current_settings()
    def on_select_turbo_top_k(self, event):
        """Chatterbox Turbo top-k cutoff (0 = no top-k filtering)."""
        self.save_current_settings()
    def on_select_turbo_repetition_penalty(self, event):
        """Chatterbox Turbo repetition penalty (1-2; higher = fewer loops)."""
        self.save_current_settings()
    def on_ai_enabled_changed(self, event):
        self.save_current_settings()
    def on_ai_api_key_changed(self, event):
        self.save_current_settings()
    def on_ai_model_changed(self, event):
        self.save_current_settings()
