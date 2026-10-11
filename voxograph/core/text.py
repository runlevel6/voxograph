"""voxograph.core.text - Text cleaning: abbreviations, Roman numerals, Unicode punctuation."""


import re

from .constants import (
    ABBREVIATIONS, _ABBREV_PATTERNS, _ROMAN_HEADING_RE, _ROMAN_KEYWORD_RE,
    _ROMAN_RE_SRC,
)


def _roman_to_int(s: str) -> int:
    """Convert a Roman numeral string to an integer.  Returns 0 for empty/invalid."""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
              'C': 100, 'D': 500, 'M': 1000}
    s = s.upper().strip()
    if not s:
        return 0
    total, prev = 0, 0
    for ch in reversed(s):
        v = values.get(ch, 0)
        if v == 0:
            return 0   # invalid character — bail out
        total += v if v >= prev else -v
        prev = v
    return total


def _int_to_words(n: int) -> str:
    """Convert a positive integer (1–3999) to English words."""
    if n <= 0 or n > 3999:
        return str(n)
    ones = [
        '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
        'sixteen', 'seventeen', 'eighteen', 'nineteen',
    ]
    tens_words = [
        '', '', 'twenty', 'thirty', 'forty', 'fifty',
        'sixty', 'seventy', 'eighty', 'ninety',
    ]
    parts = []
    if n >= 1000:
        parts.append(ones[n // 1000] + ' thousand')
        n %= 1000
    if n >= 100:
        parts.append(ones[n // 100] + ' hundred')
        n %= 100
    if n >= 20:
        t = tens_words[n // 10]
        o = ones[n % 10]
        parts.append(t + ('-' + o if o else ''))
    elif n > 0:
        parts.append(ones[n])
    return ' '.join(parts)


def _roman_match_to_words(roman_str: str) -> str:
    """
    Convert a Roman numeral string to its English word equivalent (always
    in lowercase base form). Returns the original string unchanged if it
    doesn't parse as a valid Roman numeral (guards against false positives
    like a lone 'C' or 'D').

    Casing is intentionally NOT decided here — callers apply whatever
    capitalisation fits their context (heading vs. inline), so there is a
    single, unambiguous place that decides the final case.
    """
    n = _roman_to_int(roman_str)
    if n == 0:
        return roman_str
    return _int_to_words(n)


def expand_roman_numerals(text: str) -> str:
    """
    Replace Roman numerals in two contexts:

    1. Standalone headings — a line whose entire content is a Roman numeral
       (optionally followed by . : or -).
       e.g.  "III."  →  "Three."

    2. After structural keywords — Chapter, Part, Section, Book, Volume,
       Act, Scene, Article, Appendix, Canto.
       e.g.  "Chapter XIV"  →  "Chapter Fourteen"
             "Act V, Scene i"  →  "Act Five, Scene i"  (second pass)
    """
    # Pass 1 — standalone headings. Headings are always rendered with a
    # leading capital regardless of source casing.
    def _replace_heading(m: re.Match) -> str:
        numeral, punctuation = m.group(1), m.group(2)
        words = _roman_match_to_words(numeral)
        if words == numeral:          # failed to parse — leave untouched
            return m.group(0)
        return words.capitalize() + punctuation

    text = _ROMAN_HEADING_RE.sub(_replace_heading, text)

    # Pass 2 — inline after keyword. Mirror the capitalisation style of the
    # source numeral so the result blends naturally with the surrounding text.
    def _replace_inline(m: re.Match) -> str:
        keyword, space, numeral = m.group(1), m.group(2), m.group(3)
        words = _roman_match_to_words(numeral)
        if words == numeral:
            return m.group(0)
        if numeral.isupper():
            words = words.title()
        elif numeral[:1].isupper():
            words = words.capitalize()
        return keyword + space + words

    text = _ROMAN_KEYWORD_RE.sub(_replace_inline, text)

    return text


def clean_text(text: str) -> str:
    """
    Clean raw text extracted from an EPUB chapter before TTS synthesis.

    Steps applied in order:
    1. Remove soft hyphens (U+00AD).
    2. Expand common abbreviations ending in '.' to full words.
    3. Expand Roman numerals (standalone headings and after structural keywords).
    4. Normalise typographic / Unicode punctuation to plain ASCII equivalents.
    5. Collapse runs of whitespace (spaces, tabs) to a single space.
    6. Trim leading/trailing whitespace on each line.
    7. Collapse runs of 3+ newlines to a single blank line.
    8. Strip repeated punctuation (e.g. "!!!" → "!"), preserving ellipsis.
    """
    # 1. Remove soft hyphens
    text = text.replace('\u00ad', '')

    # 2. Expand abbreviations
    for pattern, replacement in _ABBREV_PATTERNS:
        text = pattern.sub(replacement, text)

    # 3. Expand Roman numerals
    text = expand_roman_numerals(text)

    # 4. Normalise Unicode punctuation
    unicode_replacements = {
        '\u2018': "'",    # left single quotation mark
        '\u2019': "'",    # right single quotation mark
        '\u201c': '"',    # left double quotation mark
        '\u201d': '"',    # right double quotation mark
        '\u2013': '-',    # en dash
        '\u2014': ' - ',  # em dash (spaces so TTS pauses naturally)
        '\u2026': '...',  # horizontal ellipsis
        '\u00b7': '.',    # middle dot
        '\u2022': '',     # bullet — drop it
        '\xa0': ' ',      # non-breaking space
    }
    for orig, repl in unicode_replacements.items():
        text = text.replace(orig, repl)

    # 5. Collapse runs of spaces/tabs within a line
    text = re.sub(r'[ \t]+', ' ', text)

    # 6. Trim each line
    text = '\n'.join(line.strip() for line in text.splitlines())

    # 7. Collapse 3+ consecutive newlines to two
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 8. Collapse repeated punctuation; preserve ellipsis and --
    text = re.sub(r'([!?])\1+', r'\1', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r'\.{4,}', '...', text)

    return text
