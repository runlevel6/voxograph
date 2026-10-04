#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# A simple wxWidgets UI for voxograph

import torch.cuda
import numpy as np
import soundfile
import threading
import platform
import subprocess
import io
import os
import json
import wx
from wx.lib.newevent import NewEvent
from wx.lib.scrolledpanel import ScrolledPanel
from PIL import Image
from tempfile import NamedTemporaryFile
from pathlib import Path

# fix #9: import settings helpers from core — single source of truth
from voxograph.core import (
    load_settings, save_settings, DEFAULT_VOICE, check_phonetic_transcription_ai,
    correct_phonetics_ai, voice_sample_exists, voice_sample_info,
    generate_voice_sample, generate_voice_samples, missing_voice_samples,
    resolve_chatterbox_ref_audio,
)
from voxograph.voices import voices, flags

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


def _extract_bridge_json(stdout):
    """Return the JSON object emitted by the Chatterbox bridge.

    The bridge's stdout can carry stray prints from third-party libraries
    (e.g. perth's "loaded PerthNet" line), so scan the lines and return the
    last one that parses as a JSON object rather than assuming clean output.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError(
        f"Chatterbox bridge produced no JSON result. stdout={stdout!r}"
    )


class MainWindow(wx.Frame):
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

    def create_menu(self):
        menubar = wx.MenuBar()
        file_menu = wx.Menu()

        open_item = wx.MenuItem(file_menu, wx.ID_OPEN, "&Open\tCtrl+O")
        file_menu.Append(open_item)
        self.Bind(wx.EVT_MENU, self.on_open, open_item)

        exit_item = wx.MenuItem(file_menu, wx.ID_EXIT, "&Exit\tCtrl+Q")
        file_menu.Append(exit_item)
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)

        menubar.Append(file_menu, "&File")
        self.SetMenuBar(menubar)

    def on_core_started(self, event):
        print('CORE_STARTED')
        self.progress_bar_label.Show()
        self.progress_bar.Show()
        self.progress_bar.SetValue(0)
        self.progress_bar.Layout()
        self.eta_label.Show()
        self.params_panel.Layout()
        self.synth_panel.Layout()

    def on_core_chapter_started(self, event):
        self.set_table_chapter_status(event.chapter_index, "⏳ In Progress")

    def on_core_chapter_finished(self, event):
        self.set_table_chapter_status(event.chapter_index, "✅ Done")   

    def on_core_progress(self, event):
        self.progress_bar.SetValue(event.stats.progress)
        self.progress_bar_label.SetLabel(f"Synthesis Progress: {event.stats.progress}%")
        self.eta_label.SetLabel(f"Estimated Time Remaining: {event.stats.eta}")
        self.synth_panel.Layout()

    def on_core_ai_rewrite(self, event):
        ci = getattr(event, 'chapter_index', None)
        ct = getattr(event, 'chapter_total', None)
        idx = getattr(event, 'chunk_index', 0)
        tot = getattr(event, 'chunk_total', 0)
        if ci is not None and tot > 0:
            self.progress_bar_label.SetLabel(
                f"AI rewriting chapter {ci + 1}/{ct}: chunk {idx}/{tot}")
        elif ci is not None:
            self.progress_bar_label.SetLabel(f"AI rewriting chapter {ci + 1}/{ct}…")
        else:
            self.progress_bar_label.SetLabel("AI rewriting…")
        self.synth_panel.Layout()

    def on_core_ai_retry_exhausted(self, event):
        msg = getattr(event, 'message', 'AI service unavailable after multiple retries.')
        answer = wx.MessageBox(
            f"The Gemini AI service is currently unavailable or experiencing high demand.\n\n"
            f"All retry attempts have been exhausted.\n\n"
            f"Details: {msg}\n\n"
            f"Click OK to continue synthesis without AI phonetic correction, "
            f"or Cancel to stop the synthesis.",
            "AI Service Unavailable",
            wx.OK | wx.CANCEL | wx.ICON_WARNING
        )
        if answer == wx.CANCEL and self.synthesis_in_progress:
            self.cancel_current_synthesis()

    def on_core_error(self, event):
        msg = getattr(event, 'message', 'Unknown error.')
        # A fatal core error can return before CORE_FINISHED, so restore the
        # controls here to avoid leaving the UI stuck in "synthesis running".
        self.synthesis_in_progress = False
        self.cancel_button.Hide()
        self.start_button.Enable()
        self.synth_panel.Layout()
        wx.MessageBox(msg, "Synthesis Error", wx.OK | wx.ICON_ERROR)

    def on_core_finished(self, event):
        self.synthesis_in_progress = False
        self.cancel_button.Hide()
        self.start_button.Enable()
        self.synth_panel.Layout()
        self.save_current_settings()
        self.open_folder_with_explorer(self.output_folder_text_ctrl.GetValue())

    def create_layout(self):
        top_panel = wx.Panel(self)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_panel.SetSizer(top_sizer)

        open_epub_button = wx.Button(top_panel, label="📁 Open EPUB")
        open_epub_button.Bind(wx.EVT_BUTTON, self.on_open)
        top_sizer.Add(open_epub_button, 0, wx.ALL, 5)

        help_button = wx.Button(top_panel, label="ℹ️ About")
        help_button.Bind(wx.EVT_BUTTON, lambda event: self.about_dialog())
        top_sizer.Add(help_button, 0, wx.ALL, 5)

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)

        self.splitter = wx.Panel(self)
        self.splitter_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.splitter.SetSizer(self.splitter_sizer)

        self.main_sizer.Add(top_panel, 0, wx.ALL | wx.EXPAND, 5)
        self.main_sizer.Add(self.splitter, 1, wx.EXPAND)

    def create_layout_for_ebook(self, splitter):
        splitter_left = wx.Panel(splitter, -1)
        splitter_right = wx.Panel(self.splitter)
        self.splitter_left, self.splitter_right = splitter_left, splitter_right
        self.splitter_sizer.Add(splitter_left, 1, wx.ALL | wx.EXPAND, 5)
        self.splitter_sizer.Add(splitter_right, 2, wx.ALL | wx.EXPAND, 5)

        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        splitter_left.SetSizer(self.left_sizer)

        self.center_panel = wx.Panel(splitter_right)
        self.center_sizer = wx.BoxSizer(wx.VERTICAL)
        self.center_panel.SetSizer(self.center_sizer)
        self.text_area = wx.TextCtrl(self.center_panel, style=wx.TE_MULTILINE, size=(int(self.window_width * 0.4), -1))
        font = wx.Font(14, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.text_area.SetFont(font)
        self.text_area.Bind(wx.EVT_TEXT, lambda event: setattr(self.selected_chapter, 'extracted_text', self.text_area.GetValue()))

        self.chapter_label = wx.StaticText(
            self.center_panel, label=f'Edit / Preview content for section "{self.selected_chapter.short_name}":')
        preview_button = self.preview_button = wx.Button(self.center_panel, label="🔊 Preview")
        preview_button.Bind(wx.EVT_BUTTON, self.on_preview_chapter)

        check_ai_button = wx.Button(self.center_panel, label="🤖 Check with AI")
        check_ai_button.Bind(wx.EVT_BUTTON, self.on_check_phonetic_ai)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(preview_button, 0, wx.ALL, 5)
        button_sizer.Add(check_ai_button, 0, wx.ALL, 5)

        self.center_sizer.Add(self.chapter_label, 0, wx.ALL, 5)
        self.center_sizer.Add(button_sizer, 0, wx.ALL, 5)
        self.center_sizer.Add(self.text_area, 1, wx.ALL | wx.EXPAND, 5)

        splitter_right_sizer = wx.BoxSizer(wx.HORIZONTAL)
        splitter_right.SetSizer(splitter_right_sizer)

        self.create_right_panel(splitter_right)
        splitter_right_sizer.Add(self.center_panel, 1, wx.ALL | wx.EXPAND, 5)
        splitter_right_sizer.Add(self.right_panel, 1, wx.ALL | wx.EXPAND, 5)

    def about_dialog(self):
        msg = ("A simple tool to generate audiobooks from EPUB files using Kokoro-82M models\n"
               "Distributed under the MIT License.\n\n"
               "Originally by Claudio Santini 2025 — https://claudio.uk\n"
               "Fork by Vlad Reshetov 2025\n")
        wx.MessageBox(msg, "Voxograph")

    def create_right_panel(self, splitter_right):
        # Scrolled: the parameters + synthesis panels can be taller than the
        # window (especially with the Chatterbox rows), and the start button
        # must always be reachable.
        self.right_panel = ScrolledPanel(splitter_right, style=wx.TAB_TRAVERSAL)
        self.right_panel.SetScrollRate(10, 10)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_panel.SetSizer(self.right_sizer)

        self.book_info_panel_box = wx.Panel(self.right_panel, style=wx.SUNKEN_BORDER)
        book_info_panel_box_sizer = wx.StaticBoxSizer(wx.VERTICAL, self.book_info_panel_box, "Book Details")
        self.book_info_panel_box.SetSizer(book_info_panel_box_sizer)
        self.right_sizer.Add(self.book_info_panel_box, 1, wx.ALL | wx.EXPAND, 5)

        self.book_info_panel = wx.Panel(self.book_info_panel_box, style=wx.BORDER_NONE)
        self.book_info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.book_info_panel.SetSizer(self.book_info_sizer)
        book_info_panel_box_sizer.Add(self.book_info_panel, 1, wx.ALL | wx.EXPAND, 5)

        self.cover_bitmap = wx.StaticBitmap(self.book_info_panel, -1)
        self.book_info_sizer.Add(self.cover_bitmap, 0, wx.ALL, 5)
        self.cover_bitmap.Refresh()
        self.book_info_panel.Refresh()
        self.book_info_panel.Layout()
        self.cover_bitmap.Layout()

        self.create_book_details_panel()
        self.create_params_panel()
        self.create_synthesis_panel()
        self.right_panel.SetupScrolling(scroll_x=False, scroll_y=True)

    def create_book_details_panel(self):
        book_details_panel = wx.Panel(self.book_info_panel)
        book_details_sizer = wx.GridBagSizer(10, 10)
        book_details_panel.SetSizer(book_details_sizer)
        self.book_info_sizer.Add(book_details_panel, 1, wx.ALL | wx.EXPAND, 5)

        title_label = wx.StaticText(book_details_panel, label="Title:")
        title_text = wx.StaticText(book_details_panel, label=getattr(self, 'selected_book_title', 'N/A'))
        book_details_sizer.Add(title_label, pos=(0, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(title_text, pos=(0, 1), flag=wx.ALL, border=5)

        author_label = wx.StaticText(book_details_panel, label="Author:")
        author_text = wx.StaticText(book_details_panel, label=getattr(self, 'selected_book_author', 'N/A'))
        book_details_sizer.Add(author_label, pos=(1, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(author_text, pos=(1, 1), flag=wx.ALL, border=5)

        length_label = wx.StaticText(book_details_panel, label="Total Length:")
        total_len = sum(len(c.extracted_text) for c in self.document_chapters) if hasattr(self, 'document_chapters') else 0
        length_text = wx.StaticText(book_details_panel, label=f'{total_len:,} characters')
        book_details_sizer.Add(length_label, pos=(2, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(length_text, pos=(2, 1), flag=wx.ALL, border=5)

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
        sizer.Add(tts_engine_radio_panel, pos=(0, 1), flag=wx.ALL, border=border)
        tts_engine_radio_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        tts_engine_radio_panel.SetSizer(tts_engine_radio_panel_sizer)
        tts_engine_radio_panel_sizer.Add(kokoro_radio, 0, wx.ALL, 5)
        tts_engine_radio_panel_sizer.Add(chatterbox_radio, 0, wx.ALL, 5)
        kokoro_radio.Bind(wx.EVT_RADIOBUTTON, lambda event: self._on_tts_engine_changed('kokoro'))
        chatterbox_radio.Bind(wx.EVT_RADIOBUTTON, lambda event: self._on_tts_engine_changed('chatterbox'))
        self.kokoro_radio = kokoro_radio
        self.chatterbox_radio = chatterbox_radio

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

        speed_label = wx.StaticText(panel, label="Speed:")
        speed_text_input = wx.TextCtrl(panel, value=str(self.settings.get('speed', 1.0)))
        self.selected_speed = float(speed_text_input.GetValue())
        speed_text_input.Bind(wx.EVT_TEXT, self.on_select_speed)
        sizer.Add(speed_label, pos=(6, 0), flag=wx.ALL, border=border)
        sizer.Add(speed_text_input, pos=(6, 1), flag=wx.ALL, border=border)

        ai_enabled_label = wx.StaticText(panel, label="AI Phonetic Check:")
        ai_enabled_checkbox = wx.CheckBox(panel, label="Enabled")
        ai_enabled_checkbox.SetValue(self.settings.get('gemini_enabled', False))
        ai_enabled_checkbox.Bind(wx.EVT_CHECKBOX, self.on_ai_enabled_changed)
        self.ai_enabled_checkbox = ai_enabled_checkbox
        sizer.Add(ai_enabled_label, pos=(7, 0), flag=wx.ALL, border=border)
        sizer.Add(ai_enabled_checkbox, pos=(7, 1), flag=wx.ALL, border=border)

        api_key_label = wx.StaticText(panel, label="Gemini API Key:")
        api_key_text_input = wx.TextCtrl(panel, value=self.settings.get('gemini_api_key', ''), style=wx.TE_PASSWORD)
        api_key_text_input.Bind(wx.EVT_TEXT, self.on_ai_api_key_changed)
        self.ai_api_key_text_ctrl = api_key_text_input
        sizer.Add(api_key_label, pos=(8, 0), flag=wx.ALL, border=border)
        sizer.Add(api_key_text_input, pos=(8, 1), flag=wx.ALL | wx.EXPAND, border=border)

        ai_model_label = wx.StaticText(panel, label="Gemini Model:")
        ai_model_choices = ['gemini-3.1-flash-lite', 'gemini-3.5-flash', 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-flash-lite-latest']
        ai_model_dropdown = wx.ComboBox(panel, choices=ai_model_choices, value=self.settings.get('gemini_model', 'gemini-3.1-flash-lite'))
        ai_model_dropdown.Bind(wx.EVT_COMBOBOX, self.on_ai_model_changed)
        self.ai_model_dropdown = ai_model_dropdown
        sizer.Add(ai_model_label, pos=(9, 0), flag=wx.ALL, border=border)
        sizer.Add(ai_model_dropdown, pos=(9, 1), flag=wx.ALL | wx.EXPAND, border=border)

        output_folder_label = wx.StaticText(panel, label="Output Folder:")
        initial_output_folder = self.settings.get('output_folder', os.path.abspath('.'))
        self.output_folder_text_ctrl = wx.TextCtrl(panel, value=initial_output_folder)
        self.output_folder_text_ctrl.SetEditable(False)
        output_folder_button = wx.Button(panel, label="📂 Select")
        output_folder_button.Bind(wx.EVT_BUTTON, self.open_output_folder_dialog)
        sizer.Add(output_folder_label, pos=(10, 0), flag=wx.ALL, border=border)
        sizer.Add(self.output_folder_text_ctrl, pos=(10, 1), flag=wx.ALL | wx.EXPAND, border=border)
        sizer.Add(output_folder_button, pos=(11, 1), flag=wx.ALL, border=border)

        self._update_engine_ui()
        self._sync_ref_audio_text()

    def create_synthesis_panel(self):
        panel_box = wx.Panel(self.right_panel, style=wx.SUNKEN_BORDER)
        panel_box_sizer = wx.StaticBoxSizer(wx.VERTICAL, panel_box, "Audiobook Generation Status")
        panel_box.SetSizer(panel_box_sizer)

        panel = self.synth_panel = wx.Panel(panel_box)
        panel_box_sizer.Add(panel, 1, wx.ALL | wx.EXPAND, 5)
        self.right_sizer.Add(panel_box, 1, wx.ALL | wx.EXPAND, 5)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)

        self.start_button = wx.Button(panel, label="🚀 Start Audiobook Synthesis")
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start)
        sizer.Add(self.start_button, 0, wx.ALL, 5)

        self.cancel_button = wx.Button(panel, label="⛔ Cancel Synthesis")
        self.cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)
        self.cancel_button.Hide()
        sizer.Add(self.cancel_button, 0, wx.ALL, 5)

        self.progress_bar_label = wx.StaticText(panel, label="Synthesis Progress:")
        sizer.Add(self.progress_bar_label, 0, wx.ALL, 5)
        self.progress_bar = wx.Gauge(panel, range=100, style=wx.GA_PROGRESS)
        self.progress_bar.SetMinSize((-1, 30))
        sizer.Add(self.progress_bar, 0, wx.ALL | wx.EXPAND, 5)
        self.progress_bar_label.Hide()
        self.progress_bar.Hide()

        self.eta_label = wx.StaticText(panel, label="Estimated Time Remaining: ")
        self.eta_label.Hide()
        sizer.Add(self.eta_label, 0, wx.ALL, 5)

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

    def on_ai_enabled_changed(self, event):
        self.save_current_settings()

    def on_ai_api_key_changed(self, event):
        self.save_current_settings()

    def on_ai_model_changed(self, event):
        self.save_current_settings()

    def _update_engine_ui(self):
        chatterbox = bool(self.chatterbox_radio.GetValue())
        # The voice dropdown and compute device stay visible for both engines:
        # the dropdown picks the Kokoro voice *and* the Chatterbox voice preset,
        # and the device drives Kokoro sample building plus Chatterbox itself.
        self.compute_label.Show()
        self.compute_radio_panel.Show()
        self.voice_label.Show()
        self.voice_dropdown.Show()

        for widget in (self.voice_source_label, self.voice_source_radio_panel,
                       self.ref_audio_label, self.ref_audio_text_input):
            widget.Show(chatterbox)

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

    def get_selected_tts_engine(self):
        return 'chatterbox' if self.chatterbox_radio.GetValue() else 'kokoro'

    def get_chatterbox_device(self):
        return 'cuda' if self.cuda_radio.GetValue() else 'cpu'

    def save_current_settings(self):
        """Save current GUI settings via core's save_settings."""
        output_folder = self.output_folder_text_ctrl.GetValue() if (hasattr(self, 'output_folder_text_ctrl') and self.output_folder_text_ctrl) else self.settings.get('output_folder', '.')
        voice = self.get_selected_voice() if hasattr(self, 'selected_voice') else self.settings.get('voice', 'af_heart')
        speed = self.selected_speed if hasattr(self, 'selected_speed') else self.settings.get('speed', 1.0)
        gemini_api_key = self.ai_api_key_text_ctrl.GetValue() if (hasattr(self, 'ai_api_key_text_ctrl') and self.ai_api_key_text_ctrl) else self.settings.get('gemini_api_key', '')
        gemini_api_key = ' '.join(gemini_api_key.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').split())
        gemini_model = self.ai_model_dropdown.GetValue() if (hasattr(self, 'ai_model_dropdown') and self.ai_model_dropdown) else self.settings.get('gemini_model', 'gemini-3.1-flash-lite')
        gemini_enabled = self.ai_enabled_checkbox.GetValue() if (hasattr(self, 'ai_enabled_checkbox') and self.ai_enabled_checkbox) else self.settings.get('gemini_enabled', False)
        last_open_dir = self.last_open_dir if hasattr(self, 'last_open_dir') else self.settings.get('last_open_dir', '')
        tts_engine = self.get_selected_tts_engine() if hasattr(self, 'chatterbox_radio') else self.settings.get('tts_engine', 'kokoro')
        chatterbox_ref_audio = self.selected_ref_audio if hasattr(self, 'selected_ref_audio') else self.settings.get('chatterbox_ref_audio', '')
        chatterbox_device = self.get_chatterbox_device() if hasattr(self, 'cuda_radio') else self.settings.get('chatterbox_device', 'cuda')
        chatterbox_voice_source = self.get_preset_source() if hasattr(self, 'preset_radio') else self.settings.get('chatterbox_voice_source', 'preset')
        voice_samples_dir = self.settings.get('voice_samples_dir', '')

        save_settings(
            output_folder=output_folder,
            voice=voice,
            speed=speed,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            gemini_enabled=gemini_enabled,
            last_open_dir=last_open_dir,
            tts_engine=tts_engine,
            chatterbox_ref_audio=chatterbox_ref_audio,
            chatterbox_device=chatterbox_device,
            chatterbox_voice_source=chatterbox_voice_source,
            voice_samples_dir=voice_samples_dir,
        )

    def open_epub(self, file_path):
        if hasattr(self, 'selected_book'):
            ai_enabled = self.ai_enabled_checkbox.GetValue() if (hasattr(self, 'ai_enabled_checkbox') and self.ai_enabled_checkbox) else self.settings.get('gemini_enabled', False)
            self.splitter.DestroyChildren()
        else:
            ai_enabled = self.settings.get('gemini_enabled', False)

        self.selected_file_path = file_path
        print(f"Opening file: {file_path}")

        import voxograph.core as core
        from ebooklib import epub

        try:
            book = epub.read_epub(file_path)
        except Exception as e:
            wx.MessageBox(f"Error opening EPUB file: {e}", "Error", wx.OK | wx.ICON_ERROR)
            print(f"Error reading EPUB file '{file_path}': {e}")
            return

        meta_title = book.get_metadata('DC', 'title')
        self.selected_book_title = meta_title[0][0] if meta_title else ''
        meta_creator = book.get_metadata('DC', 'creator')
        self.selected_book_author = meta_creator[0][0] if meta_creator else ''
        self.selected_book = book

        self.document_chapters = core.find_document_chapters_and_extract_texts(book, ai_enabled=ai_enabled)
        good_chapters = core.find_good_chapters(self.document_chapters)
        self.selected_chapter = good_chapters[0] if good_chapters else None
        if self.selected_chapter is None:
            wx.MessageBox("No readable chapters found in this EPUB.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        for chapter in self.document_chapters:
            chapter.short_name = (chapter.get_name()
                                  .replace('.xhtml', '').replace('xhtml/', '')
                                  .replace('.html', '').replace('Text/', ''))
            chapter.is_selected = chapter in good_chapters

        self.create_layout_for_ebook(self.splitter)

        cover = core.find_cover(book)
        if cover is not None:
            pil_image = Image.open(io.BytesIO(cover.content))
            wx_img = wx.EmptyImage(pil_image.size[0], pil_image.size[1])
            wx_img.SetData(pil_image.convert("RGB").tobytes())
            cover_h = 200
            cover_w = int(cover_h * pil_image.size[0] / pil_image.size[1])
            wx_img.Rescale(cover_w, cover_h)
            self.cover_bitmap.SetBitmap(wx_img.ConvertToBitmap())
            self.cover_bitmap.SetMaxSize((200, cover_h))

        chapters_panel = self.create_chapters_table_panel(good_chapters)

        if self.chapters_panel:
            self.left_sizer.Replace(self.chapters_panel, chapters_panel)
            self.chapters_panel.Destroy()
            self.chapters_panel = chapters_panel
        else:
            self.left_sizer.Add(chapters_panel, 1, wx.ALL | wx.EXPAND, 5)
            self.chapters_panel = chapters_panel

        self.splitter_left.Layout()
        self.splitter_right.Layout()
        self.splitter.Layout()
        if hasattr(self.right_panel, 'SetupScrolling'):
            self.right_panel.SetupScrolling(scroll_x=False, scroll_y=True)

        if self.selected_chapter:
            self.text_area.SetValue(self.selected_chapter.extracted_text)
            self.chapter_label.SetLabel(f'Edit / Preview content for section "{self.selected_chapter.short_name}":')

    def on_table_checked(self, event):
        self.document_chapters[event.GetIndex()].is_selected = True

    def on_table_unchecked(self, event):
        self.document_chapters[event.GetIndex()].is_selected = False

    def on_table_selected(self, event):
        chapter = self.document_chapters[event.GetIndex()]
        print('Selected', event.GetIndex(), chapter.short_name)
        self.selected_chapter = chapter
        self.text_area.SetValue(chapter.extracted_text)
        self.chapter_label.SetLabel(f'Edit / Preview content for section "{chapter.short_name}":')

    def create_chapters_table_panel(self, good_chapters):
        panel = ScrolledPanel(self.splitter_left, -1, style=wx.TAB_TRAVERSAL | wx.SUNKEN_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)

        self.table = table = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        table.InsertColumn(0, "Included")
        table.InsertColumn(1, "Chapter Name")
        table.InsertColumn(2, "Chapter Length")
        table.InsertColumn(3, "Status")
        table.SetColumnWidth(0, 80)
        table.SetColumnWidth(1, 150)
        table.SetColumnWidth(2, 150)
        table.SetColumnWidth(3, 100)
        table.SetSize((250, -1))
        table.EnableCheckBoxes()
        table.Bind(wx.EVT_LIST_ITEM_CHECKED, self.on_table_checked)
        table.Bind(wx.EVT_LIST_ITEM_UNCHECKED, self.on_table_unchecked)
        table.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_table_selected)

        for i, chapter in enumerate(self.document_chapters):
            auto_selected = chapter in good_chapters
            table.Append(['', chapter.short_name, f"{len(chapter.extracted_text):,}"])
            if auto_selected:
                table.CheckItem(i)

        title_text = wx.StaticText(panel, label="Select chapters to include in the audiobook:")
        sizer.Add(title_text, 0, wx.ALL, 5)
        sizer.Add(table, 1, wx.ALL | wx.EXPAND, 5)
        return panel

    def get_selected_voice(self):
        """Return just the voice code, stripping any leading flag emoji."""
        parts = self.selected_voice.split(' ')
        return parts[1] if len(parts) > 1 else self.selected_voice

    def get_selected_speed(self):
        return float(self.selected_speed)

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

        text = self.selected_chapter.extracted_text[:300]
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
                        tts_engine=engine)
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
                    payload = json.dumps({
                        "text": preview_text,
                        "output_path": tmp_path,
                        "device": device,
                        "language_id": "en",
                        "audio_prompt_path": ref_audio,
                        "t3_model": "t3_mtl23ls_v3.safetensors",
                    })
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
                    tts_engine=tts_engine
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
        self.synth_panel.Layout()
        self.params_panel.Disable()
        self.table.EnableCheckBoxes(False)

        self.stop_event = threading.Event()
        print('Starting Audiobook Synthesis', dict(file_path=file_path, voice=voice, speed=speed))
        self.core_thread = CoreThread(
            params=dict(file_path=file_path, voice=voice, pick_manually=False, speed=speed,
                        output_folder=self.output_folder_text_ctrl.GetValue(),
                        selected_chapters=selected_chapters,
                        tts_engine=self.get_selected_tts_engine()),
            stop_event=self.stop_event)
        self.core_thread.start()

    def on_open(self, event):
        with wx.FileDialog(self, "Open EPUB File", wildcard="*.epub",
                           defaultDir=self.last_open_dir,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            file_path = dialog.GetPath()
            if not file_path:
                return
            self.last_open_dir = str(Path(file_path).parent)
            self.save_current_settings()
            if self.synthesis_in_progress:
                wx.MessageBox("Audiobook synthesis is still in progress. Please wait.",
                              "Synthesis in Progress")
            else:
                wx.CallAfter(self.open_epub, file_path)

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

    def on_exit(self, event):
        if self.synthesis_in_progress:
            answer = wx.MessageBox(
                "Audiobook synthesis is still in progress.\nStop synthesis and exit?",
                "Exit Voxograph", wx.YES_NO | wx.ICON_WARNING)
            if answer != wx.YES:
                return
            if self.stop_event:
                self.stop_event.set()
        self.save_current_settings()
        self.Close()

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


class CoreThread(threading.Thread):
    def __init__(self, params, stop_event):
        super().__init__(daemon=True)
        self.params = params
        self.stop_event = stop_event

    def run(self):
        import voxograph.core as core
        core.main(**self.params, stop_event=self.stop_event, post_event=self.post_event)

    def post_event(self, event_name, **kwargs):
        EventObject, EVENT_CODE = EVENTS[event_name]
        event_object = EventObject()
        for k, v in kwargs.items():
            setattr(event_object, k, v)
        wx.PostEvent(wx.GetApp().GetTopWindow(), event_object)


def main():
    print('Starting GUI...')
    app = wx.App(False)
    frame = MainWindow(None, "Voxograph - Generate Audiobooks from E-books")
    frame.Show(True)
    frame.Layout()
    app.SetTopWindow(frame)
    print('Done.')
    app.MainLoop()


if __name__ == '__main__':
    main()
