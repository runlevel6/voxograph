"""voxograph.core.text - Text cleaning: abbreviations, Roman numerals, Unicode punctuation."""


import re


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Fix #1: Street pattern MUST come before the generic Saint pattern so that
# "St. James" → "Street James" does not accidentally fire first.
ABBREVIATIONS = [
    # Titles
    (r'\bMrs\.', 'Missus'),
    (r'\bMr\.', 'Mister'),
    (r'\bMs\.', 'Miz'),
    (r'\bMiss\.', 'Miss'),
    (r'\bDr\.', 'Doctor'),
    (r'\bProf\.', 'Professor'),
    (r'\bRev\.', 'Reverend'),
    (r'\bHon\.', 'Honorable'),
    # Street before Saint so the lookahead fires first
    (r'\bSt\.(?=\s+[A-Z])', 'Street'),
    (r'\bSt\.', 'Saint'),
    # Military / professional ranks
    (r'\bGen\.', 'General'),
    (r'\bCol\.', 'Colonel'),
    (r'\bMaj\.', 'Major'),
    (r'\bCapt\.', 'Captain'),
    (r'\bLt\.', 'Lieutenant'),
    (r'\bSgt\.', 'Sergeant'),
    (r'\bCpl\.', 'Corporal'),
    (r'\bPvt\.', 'Private'),
    (r'\bAdm\.', 'Admiral'),
    # Name suffixes
    (r'\bJr\.', 'Junior'),
    (r'\bSr\.', 'Senior'),
    (r'\bEsq\.', 'Esquire'),
    # Common Latin / general abbreviations
    (r'\betc\.', 'et cetera'),
    (r'\bvs\.', 'versus'),
    (r'\bVs\.', 'Versus'),
    (r'\bapprox\.', 'approximately'),
    (r'\bcf\.', 'compare'),
    (r'\be\.g\.', 'for example'),
    (r'\bi\.e\.', 'that is'),
    (r'\bviz\.', 'namely'),
    (r'\bib\.', 'in the same place'),
    (r'\bibid\.', 'in the same place'),
    (r'\bop\. cit\.', 'in the work cited'),
    (r'\bno\.?\s*(?=\d)', 'number '),
    (r'\bNo\.?\s*(?=\d)', 'Number '),
    (r'\bvol\.', 'volume'),
    (r'\bVol\.', 'Volume'),
    (r'\bch\.', 'chapter'),
    (r'\bCh\.', 'Chapter'),
    (r'\bfig\.', 'figure'),
    (r'\bFig\.', 'Figure'),
    (r'\bp\.', 'page'),
    (r'\bpp\.', 'pages'),
    # Addresses
    (r'\bAve\.', 'Avenue'),
    (r'\bBlvd\.', 'Boulevard'),
    (r'\bRd\.', 'Road'),
    (r'\bDept\.', 'Department'),
    (r'\bGovt\.', 'Government'),
]

# Compile once for efficiency
_ABBREV_PATTERNS = [(re.compile(pat), repl) for pat, repl in ABBREVIATIONS]


# ---------------------------------------------------------------------------
# Roman numeral conversion
# ---------------------------------------------------------------------------

# Matches a valid Roman numeral token (1–3999).  The alternation structure
# ensures only well-formed sequences match — e.g. "IIII" won't match because
# there is no four-I combination in standard notation.
_ROMAN_RE_SRC = (
    r'M{0,4}'           # thousands: 0–4000
    r'(?:CM|CD|D?C{0,3})'   # hundreds: 900, 400, 0–300, 500–800
    r'(?:XC|XL|L?X{0,3})'   # tens: 90, 40, 0–30, 50–80
    r'(?:IX|IV|V?I{0,3})'   # ones: 9, 4, 0–3, 5–8
)

# Case 1 — standalone heading: a line that is *only* a Roman numeral,
# optionally followed by a period, colon, or dash (and nothing else).
# Examples:  "III."   "XIV"   "IV:"   "  ii.  "
_ROMAN_HEADING_RE = re.compile(
    r'^(' + _ROMAN_RE_SRC + r')([.:\-]?)$',
    re.IGNORECASE | re.MULTILINE,
)

# Case 2 — inline after a structural keyword.
# Examples:  "Chapter III"   "Part IV:"   "Book II,"   "Act V, Scene i"
_ROMAN_KEYWORD_RE = re.compile(
    r'\b(Chapter|Part|Section|Book|Volume|Vol|Act|Scene|Article|Appendix|Canto)'
    r'(\s+)(' + _ROMAN_RE_SRC + r')\b',
    re.IGNORECASE,
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
