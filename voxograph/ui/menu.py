"""voxograph.ui.menu - MainWindow MenuMixin methods."""
from pathlib import Path
import wx


class MenuMixin:
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
    def about_dialog(self):
        msg = ("A simple tool to generate audiobooks from EPUB files using Kokoro-82M models\n"
               "Distributed under the MIT License.\n\n"
               "Originally by Claudio Santini 2025 — https://claudio.uk\n"
               "Fork by Vlad Reshetov 2025\n")
        wx.MessageBox(msg, "Voxograph")
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
