"""voxograph.ui.app - wx.App bootstrap (voxograph-ui entry point)."""
import wx

from voxograph.ui.window import MainWindow


def main():
    print('Starting GUI...')
    app = wx.App(False)
    frame = MainWindow(None, "Voxograph - Generate Audiobooks from E-books")
    frame.Show(True)
    frame.Layout()
    app.SetTopWindow(frame)
    print('Done.')
    app.MainLoop()
