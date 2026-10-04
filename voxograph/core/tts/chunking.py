"""voxograph.core.tts.chunking - Sentence-aware chunking for Chatterbox synthesis."""


import re


CHATTERBOX_MAX_CHUNK_CHARS = 300

_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?\u2026])\s+')
_CLAUSE_BOUNDARY_RE = re.compile(r'(?<=[,;:])\s+')


def _split_oversized_sentence(sentence, max_chars):
    """Break a single sentence longer than `max_chars` at clause boundaries,
    falling back to word boundaries and finally a hard character split."""
    out = []
    current = ''
    for clause in _CLAUSE_BOUNDARY_RE.split(sentence):
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) > max_chars:
            for word in clause.split():
                candidate = f'{current} {word}'.strip() if current else word
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        out.append(current)
                    current = word
        else:
            candidate = f'{current} {clause}'.strip() if current else clause
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    out.append(current)
                current = clause
    if current:
        out.append(current)

    final = []
    for piece in out:
        if len(piece) <= max_chars:
            final.append(piece)
        else:
            final.extend(piece[i:i + max_chars] for i in range(0, len(piece), max_chars))
    return final or [sentence[:max_chars]]


def split_chatterbox_text(text, max_chars=CHATTERBOX_MAX_CHUNK_CHARS):
    """Split `text` into chunks no longer than `max_chars` for Chatterbox.

    Break priority: paragraph -> sentence -> clause -> word -> hard split.
    Original punctuation is preserved so prosody stays as written. Returns a
    list of non-empty chunks (empty list for empty input).
    """
    text = (text or '').strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    for paragraph in re.split(r'\n{2,}', text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        current = ''
        for sentence in _SENTENCE_BOUNDARY_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            pieces = ([sentence] if len(sentence) <= max_chars
                      else _split_oversized_sentence(sentence, max_chars))
            for piece in pieces:
                candidate = f'{current} {piece}'.strip() if current else piece
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = piece
        if current:
            chunks.append(current)
    return chunks


def preview_excerpt(text, max_chars=CHATTERBOX_MAX_CHUNK_CHARS):
    """Return a <= `max_chars` prefix of `text` cut only at a real boundary.

    Used for the GUI preview, which must stay short but must never slice a word
    or a sentence in half: it takes the first chunk of the same sentence-aware
    splitter that drives full-book synthesis, so a preview is always one whole
    thought. Returns `''` when there is nothing to read.
    """
    chunks = split_chatterbox_text(text, max_chars=max_chars)
    return chunks[0] if chunks else ''
