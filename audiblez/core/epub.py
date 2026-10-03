"""audiblez.core.epub - EPUB parsing, chapter extraction/selection, and CLI chapter display."""


import ebooklib
import platform
import re
from tabulate import tabulate
from bs4 import BeautifulSoup
from pick import pick

from .text import clean_text


def find_cover(book):
    def is_image(item):
        return item is not None and item.media_type.startswith('image/')

    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        if is_image(item):
            return item

    for meta in book.get_metadata('OPF', 'cover'):
        if is_image(item := book.get_item_with_id(meta[1]['content'])):
            return item

    if is_image(item := book.get_item_with_id('cover')):
        return item

    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if 'cover' in item.get_name().lower() and is_image(item):
            return item

    return None


def print_selected_chapters(document_chapters, chapters):
    ok = 'X' if platform.system() == 'Windows' else '✅'
    print(tabulate([
        [i, c.get_name(), len(c.extracted_text), ok if c in chapters else '', chapter_beginning_one_liner(c)]
        for i, c in enumerate(document_chapters, start=1)
    ], headers=['#', 'Chapter', 'Text Length', 'Selected', 'First words']))
def find_document_chapters_and_extract_texts(book, ai_enabled=False):
    """Returns every chapter that is an ITEM_DOCUMENT and enriches each
    chapter with extracted text.

    When ai_enabled is True the raw extracted text is left untouched —
    clean_text() is skipped — because the AI phonetic rewrite step handles
    all text normalization before TTS.

    Iterates book.spine rather than book.get_items() so that:
      - Only items in the actual reading order are processed (no orphaned
        manifest resources that are never shown to the reader).
      - Duplicate manifest entries for the same content are naturally
        avoided, since the spine lists each idref at most once.
    """
    document_chapters = []
    for idref, _linear in book.spine:
        chapter = book.get_item_with_id(idref)
        if chapter is None or chapter.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        xml = chapter.get_body_content()
        soup = BeautifulSoup(xml, features='lxml')
        chapter.extracted_text = ''
        html_content_tags = ['title', 'p', 'h1', 'h2', 'h3', 'h4', 'li']
        for text in [c.text.strip() for c in soup.find_all(html_content_tags) if c.text]:
            # fix #2: only append a period when the sentence doesn't already
            # end with terminal punctuation (., ?, !)
            if text and text[-1] not in '.?!':
                text += '.'
            chapter.extracted_text += text + '\n'

        # Apply automated text cleaning unless AI pronunciation rewriting is
        # enabled — in that case the AI step handles normalization instead.
        if not ai_enabled:
            chapter.extracted_text = clean_text(chapter.extracted_text)

        document_chapters.append(chapter)
    for i, c in enumerate(document_chapters):
        c.chapter_index = i
    return document_chapters


def is_chapter(c):
    name = c.get_name().lower()
    has_min_len = len(c.extracted_text) > 100
    title_looks_like_chapter = bool(
        'chapter' in name
        or re.search(r'part_?\d{1,3}', name)
        or re.search(r'split_?\d{1,3}', name)
        or re.search(r'ch_?\d{1,3}', name)
        or re.search(r'chap_?\d{1,3}', name)
    )
    return has_min_len and title_looks_like_chapter


def chapter_beginning_one_liner(c, chars=20):
    s = c.extracted_text[:chars].strip().replace('\n', ' ').replace('\r', ' ')
    return s + '…' if len(s) > 0 else ''


def find_good_chapters(document_chapters):
    chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if len(chapters) == 0:
        print('Not easy to recognize the chapters, defaulting to all non-empty documents.')
        chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and len(c.extracted_text) > 10]
    return chapters


def pick_chapters(chapters):
    chapters_by_names = {
        f'{c.get_name()}\t({len(c.extracted_text)} chars)\t[{chapter_beginning_one_liner(c, 50)}]': c
        for c in chapters}
    title = 'Select which chapters to read in the audiobook'
    ret = pick(list(chapters_by_names.keys()), title, multiselect=True, min_selection_count=1)
    selected_chapters_out_of_order = [chapters_by_names[r[0]] for r in ret]
    selected_chapters = [c for c in chapters if c in selected_chapters_out_of_order]
    return selected_chapters
