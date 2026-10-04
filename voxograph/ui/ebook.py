"""voxograph.ui.ebook - MainWindow EbookMixin methods."""
from PIL import Image
from wx.lib.scrolledpanel import ScrolledPanel
import io
import wx


class EbookMixin:
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
