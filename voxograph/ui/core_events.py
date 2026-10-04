"""voxograph.ui.core_events - MainWindow CoreEventsMixin methods."""
import wx


class CoreEventsMixin:
    def on_core_started(self, event):
        print('CORE_STARTED')
        self.progress_bar_label.Show()
        self.progress_bar.Show()
        self.progress_bar.SetValue(0)
        self.progress_bar.Layout()
        self.eta_label.Show()
        self.params_panel.Layout()
        self._relayout_right_panel()
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
        self._relayout_right_panel()
        wx.MessageBox(msg, "Synthesis Error", wx.OK | wx.ICON_ERROR)
    def on_core_finished(self, event):
        self.synthesis_in_progress = False
        self.cancel_button.Hide()
        self.start_button.Enable()
        self._relayout_right_panel()
        self.save_current_settings()
        self.open_folder_with_explorer(self.output_folder_text_ctrl.GetValue())
