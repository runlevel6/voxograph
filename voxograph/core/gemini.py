"""voxograph.core.gemini - Gemini retry helper, phonetic rules, and AI pronunciation rewrites."""


import re
import time
from google import genai


# ---------------------------------------------------------------------------
# Gemini API usage guidance (AFC)
# ---------------------------------------------------------------------------
# Automatic Function Calling (AFC) is how the google-genai SDK executes the
# function calls the model returns. Whenever Gemini is invoked with tools /
# function declarations, drive AFC through a Chat session:
#
#   chat = client.chats.create(model=model, config=config_with_tools)
#   chat.send_message(contents)          # non-streaming  → recommended
#   chat.send_message_stream(contents)   # streaming      → recommended
#
# Direct use of AFC through Models.generate_content / Models.generate_content_stream
# is NOT recommended: those raw entry points bypass chat-managed context, and the
# SDK logs the warning "Direct use of automatic function calling (AFC) in
# Models.generate_content_stream is not recommended. Instead, we recommend to use
# AFC in Chat.send_message_stream." Use Chat.send_message (or
# Chat.send_message_stream) instead. The plain no-tools calls below are the only
# place the code may still hit client.models.generate_content directly.


# ---------------------------------------------------------------------------
# Gemini retry helper
# ---------------------------------------------------------------------------

# Backoff schedule requested: retry after 1 minute, then 2, then 3, then 4
# (each pause is the previous pause + 1 minute). That's 4 retries (5 total
# attempts) before giving up.
_GEMINI_RETRY_DELAYS_SECS = [60, 120, 180, 240]

# Errors in this set are permanent (bad model name, bad key, no permission)
# and will never succeed no matter how many times we retry, so we bail out
# immediately instead of making the caller wait up to 10 minutes for nothing.
_NON_RETRYABLE_ERROR_MARKERS = ('NOT_FOUND', 'PERMISSION_DENIED', 'INVALID_ARGUMENT', 'UNAUTHENTICATED')


def _is_retryable_gemini_error(exc) -> bool:
    msg = str(exc)
    return not any(marker in msg for marker in _NON_RETRYABLE_ERROR_MARKERS)


def _sleep_with_stop_event(seconds, stop_event=None):
    """Sleep in small increments so a stop_event can interrupt promptly
    instead of blocking for the full backoff duration."""
    elapsed = 0.0
    step = 1.0
    while elapsed < seconds:
        if stop_event and stop_event.is_set():
            return
        time.sleep(min(step, seconds - elapsed))
        elapsed += step


def _call_gemini_with_retry(func, *args, stop_event=None, post_event=None, **kwargs):
    """
    Call func(*args, **kwargs), retrying on failure with an increasing
    backoff of 1, 2, 3, then 4 minutes (4 retries / 5 attempts total).
    Permanent-looking errors (invalid model, bad key, permission denied)
    are not retried. If stop_event fires during a backoff wait, the last
    exception is raised immediately.

    Returns func's return value on success, or raises the last exception
    once attempts are exhausted. If post_event is provided, it is called
    with event name 'CORE_AI_RETRY_EXHAUSTED' and the error message when
    all retries are exhausted due to transient errors.
    """
    last_exc = None
    for attempt in range(len(_GEMINI_RETRY_DELAYS_SECS) + 1):
        if stop_event and stop_event.is_set():
            if last_exc:
                raise last_exc
            raise RuntimeError('Stopped by user.')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            more_attempts_left = attempt < len(_GEMINI_RETRY_DELAYS_SECS)
            if more_attempts_left and _is_retryable_gemini_error(e):
                delay = _GEMINI_RETRY_DELAYS_SECS[attempt]
                print(f'\033[93mGemini call failed ({e}). Retrying in {delay // 60} '
                      f'minute(s)... (attempt {attempt + 2}/{len(_GEMINI_RETRY_DELAYS_SECS) + 1})\033[0m')
                _sleep_with_stop_event(delay, stop_event)
            else:
                break
    if post_event and last_exc and _is_retryable_gemini_error(last_exc):
        post_event('CORE_AI_RETRY_EXHAUSTED', message=str(last_exc))
    raise last_exc
def _sanitize_api_key(api_key):
    if not api_key:
        return ''
    cleaned = api_key.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()
    if cleaned != api_key or any(c.isspace() for c in api_key):
        cleaned = ' '.join(cleaned.split())
    return cleaned



# ---------------------------------------------------------------------------
# Shared phonetic rewrite rules — single source of truth used by BOTH the
# silent rewrite path (_ai_rewrite_single_chunk) and the human-readable
# analysis path (check_phonetic_transcription_ai), so the two can never
# describe/apply different rules for what counts as "needs fixing".
#
# Note: espeak-ng's [[...]] phoneme-override syntax is intentionally NOT
# offered here. That syntax expects espeak's own Kirshenbaum-based ASCII
# mnemonics (e.g. "w3:ld"), not standard Unicode IPA — Gemini reliably
# produces the latter but not the former, so asking it to emit [[...]]
# phonemes silently produces garbled/unrecognized input to espeak-ng.
# Kokoro's own inline IPA override (plain Unicode IPA, no brackets) is the
# only phonetic-override path offered, since Gemini can produce valid IPA.
# ---------------------------------------------------------------------------
_PHONETIC_RULES = (
    "Only flag/change words or formatting that affect pronunciation, such as:\n"
    "- Expand abbreviations (Dr. -> Doctor, St. -> Saint only when it is a name, "
    "  NASA -> N A S A, FBI -> F B I, etc.)\n"
    "- Spell out numbers and dates in words (2024 -> twenty twenty four, 3rd -> third)\n"
    "- Fix homophones, silent letters, or unusual stress by re-spelling the word\n\n"
    "FOREIGN PROPER NOUNS (mandatory, not optional):\n"
    "Any proper noun using non-English orthography or spelling conventions — accented "
    "letters (é, è, ç, œ, ü, etc.), unusual consonant clusters, silent letters, or "
    "foreign name endings — MUST be rewritten in some form. This applies to place names, "
    "personal names, military unit names, and any other proper noun. Do not leave a "
    "French, German, or other foreign-language name unchanged, even if you are uncertain "
    "of the exact native pronunciation — an approximate English rendering is always better "
    "than leaving the raw spelling for the default G2P to butcher.\n"
    "  Example: 'Amiens' -> 'Amyen' or ˈɑːmiˌæn\n"
    "  Example: 'Péronne' -> 'Peyron' or peɪˈrɒn\n"
    "  Example: 'Flixécourt' -> 'Flixaycoor'\n"
    "  Example: 'GUDERIAN' -> 'Guderian' (normalize casing)\n\n"
    "You MAY use the Kokoro inline phonetic override: replace a word directly with its "
    "standard Unicode IPA phonetic spelling (no brackets needed), using stress marks ˈ "
    "(primary) and ˌ (secondary). Example: Kokoro or Paris rewritten as kˈOkəɹO or ˈpæɹɪs. "
    "Do not use espeak-style double-bracket phoneme syntax (e.g. [[...]]) — it is not "
    "supported here and will not be pronounced correctly.\n\n"
    "Prosody: existing punctuation already controls intonation — "
    "; : , . ! ? — … \" ( ) \u201c \u201d all shape phrasing and pitch. "
    "Do not remove or alter punctuation; it is meaningful for Kokoro.\n\n"
    "- Prefer the simplest fix: only use inline IPA replacements for words that "
    "  the default G2P would misread (names, loanwords, acronyms). Plain English "
    "  respelling is preferred when it's simpler and equally accurate."
)

# Chatterbox has no inline phonetic-override syntax: IPA symbols, stress marks
# and [[...]] would be read literally (or dropped). Chatterbox rewrites must
# therefore use ordinary English letter respelling only.
_CHATTERBOX_PHONETIC_RULES = (
    "Only flag/change words or formatting that affect pronunciation, such as:\n"
    "- Expand abbreviations (Dr. -> Doctor, St. -> Saint only when it is a name, "
    "  NASA -> N A S A, FBI -> F B I, etc.)\n"
    "- Spell out numbers and dates in words (2024 -> twenty twenty four, 3rd -> third)\n"
    "- Respell words with unusual pronunciation using ordinary English letters\n\n"
    "FOREIGN PROPER NOUNS (mandatory, not optional):\n"
    "Any proper noun using non-English orthography or spelling conventions — accented "
    "letters (é, è, ç, œ, ü, etc.), unusual consonant clusters, silent letters, or "
    "foreign name endings — MUST be rewritten using ordinary English letters. This applies "
    "to place names, personal names, military unit names, and any other proper noun. Do not "
    "leave a French, German, or other foreign-language name unchanged, even if you are "
    "uncertain of the exact native pronunciation — an approximate English rendering is "
    "always better than leaving the raw spelling for the default G2P to butcher.\n"
    "  Example: 'Amiens' -> 'Amyen'\n"
    "  Example: 'Péronne' -> 'Peyron'\n"
    "  Example: 'Flixécourt' -> 'Flixaycoor'\n"
    "  Example: 'GUDERIAN' -> 'Guderian' (normalize casing)\n\n"
    "IMPORTANT: The Chatterbox TTS engine does NOT understand phonetic alphabets. "
    "Do NOT output IPA symbols, stress marks, slashes, or espeak [[...]] syntax. "
    "Use only ordinary English letters and normal punctuation — the ONLY square "
    "brackets you may ever output are the expressive tags listed at the end of "
    "these rules.\n\n"
    "Prosody: existing punctuation already controls intonation — "
    "; : , . ! ? — … \" ( ) \u201c \u201d all shape phrasing and pitch. "
    "Do not remove or alter punctuation; it is meaningful for Chatterbox."
)

# Chatterbox (Turbo) ships 19 expressive tokens trained natively into its
# tokenizer vocabulary: 9 non-speech vocal effects and 10 emotion/delivery
# styles. They are the only way to steer delivery from inside the text, so
# when AI rewriting runs for a Chatterbox engine we ask for these tags on top
# of the ordinary pronunciation fixes. Tags are lowercase and bracketed, and
# they are the sole bracketed syntax allowed for Chatterbox.
_CHATTERBOX_TAG_RULES = (
    "CHATTERBOX EXPRESSIVE TAGS (apply IN ADDITION to the pronunciation rewrites above):\n"
    "The selected Chatterbox model has 19 built-in special tokens that make the speech "
    "more natural, conversational, and expressive. Use ONLY these exact lowercase, "
    "square-bracketed tags, spelled exactly as shown:\n\n"
    "1) Vocal Sound Effects — insert inline or mid-sentence where a real speaker would "
    "physically react:\n"
    "   [laugh] [chuckle] [gasp] [sigh] [groan] [sniff] [cough] [clear throat] [shush]\n\n"
    "2) Emotion & Delivery Styles — place ONLY at the very beginning of a sentence or a "
    "distinct line, to set the tone for the words that follow:\n"
    "   [happy] [crying] [angry] [fear] [surprised] [whispering] [sarcastic] [dramatic] "
    "[narration] [advertisement]\n\n"
    "Tag instructions:\n"
    "- Analyze the context, emotion, and punctuation, then insert a tag only where a real "
    "human would naturally breathe, laugh, shift emotion, or react physically. Do NOT "
    "overuse them — not every sentence needs a tag.\n"
    "- Tags are always lowercase and enclosed in square brackets. Never translate, "
    "capitalize, alter, or invent a tag, and never emit any other bracketed syntax.\n"
    "- An emotion tag at the start of a line naturally bleeds into the words that follow, "
    "until a new tag or punctuation resets the pacing.\n"
    "- Do not change, add, or remove any of the author's words or punctuation while "
    "adding tags; the tags are the only insertions you may make.\n"
    "- The first tag may only be inserted where the input actually begins; never prepend "
    "a tag before the very first word of the supplied text."
)


def _phonetic_rules_for(tts_engine, chatterbox_model=None):
    """Pick the rewrite ruleset for the engine that will actually speak the text.

    Kokoro gets the phonetic rules alone (it has no tag vocabulary). Chatterbox
    gets the plain-English respelling rules, plus the expressive-tag ruleset
    only for the Turbo model — the 19 tags are trained into Turbo's tokenizer
    and Multilingual V3 would read them as literal words.
    """
    if tts_engine != 'chatterbox':
        return _PHONETIC_RULES
    rules = _CHATTERBOX_PHONETIC_RULES
    if chatterbox_model == 'turbo':
        rules = rules + '\n\n' + _CHATTERBOX_TAG_RULES
    return rules


def check_phonetic_transcription_ai(text, api_key, model='gemini-3.1-flash-lite',
                                    stop_event=None, tts_engine='kokoro',
                                    chatterbox_model=None):
    """
    Use Google Gemini AI to analyze text for potential TTS pronunciation issues
    and provide phonetic transcription guidance.

    Returns a string with the AI's explanation/suggestions, followed by the
    actual rewritten text — produced by the same correct_phonetics_ai() /
    _ai_rewrite_single_chunk() path used during real synthesis — so what's
    shown here is exactly what would be sent to the TTS engine, not just a
    description of the changes.

    The analysis prompt below shares _PHONETIC_RULES with
    _ai_rewrite_single_chunk so the explanation and the actual rewrite can
    never disagree about what counts as an issue worth fixing.
    """
    if not text.strip():
        return "Error: text is empty."

    api_key = _sanitize_api_key(api_key)
    if not api_key:
        return "Error: API key is missing. Please paste it in the AI Phonetic Check section."
    if not api_key.startswith('AIza'):
        return "Error: API key looks invalid (Gemini keys typically start with 'AIza'). Please check the value you pasted."

    rules = _phonetic_rules_for(tts_engine, chatterbox_model)
    engine_name = 'Chatterbox' if tts_engine == 'chatterbox' else 'Kokoro'

    def _do_call():
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(
            model=model,
            contents=(
                f"You are a phonetic transcription expert for a {engine_name} text-to-speech system. "
                "Analyze the following text from an audiobook and identify words or phrases "
                "that might be mispronounced by the TTS engine, using EXACTLY the same rules "
                "that will be used to actually rewrite this text (listed below), so your analysis "
                "matches what the rewrite step will do.\n\n"
                f"{rules}\n\n"
                "For each issue found, provide:\n"
                "1. The problematic word/phrase\n"
                "2. Why it might be mispronounced\n"
                "3. The suggested rewrite that will actually be applied\n"
                "4. A short explanation of the change\n\n"
                "Remember: foreign proper nouns are a MANDATORY category — do not skip any of "
                "them in your analysis, even if you're only approximating the pronunciation.\n\n"
                "If the text looks clean with no obvious issues, say so and provide a brief confirmation.\n\n"
                f"Text to analyze:\n\"\"\"\n{text}\n\"\"\""
            )
        )

    try:
        response = _call_gemini_with_retry(_do_call, stop_event=stop_event)
        analysis = response.text.strip()
    except Exception as e:
        msg = str(e)
        if '404' in msg and 'NOT_FOUND' in msg:
            return (f"Error: the model '{model}' is not available for your API key/account. "
                    f"Please pick a current model in the GUI (e.g. gemini-3.1-flash-lite, "
                    f"gemini-3.5-flash, or gemini-flash-lite-latest) and try again.\n\nDetails: {msg}")
        return f"Error during AI analysis: {msg}"

    # Produce the actual TTS-bound text using the exact same rewrite path
    # (including chunking for long text) that main() uses before synthesis,
    # so the preview matches reality rather than just describing changes.
    rewritten_text = correct_phonetics_ai(text, api_key, model=model,
                                          stop_event=stop_event, tts_engine=tts_engine,
                                          chatterbox_model=chatterbox_model)

    return (
        f"{analysis}\n\n"
        f"{'=' * 60}\n"
        f"REWRITTEN TEXT (this is what will be sent to the TTS engine)\n"
        f"{'=' * 60}\n\n"
        f"{rewritten_text}"
    )

# Rough char-to-token ratio used to size Gemini requests for the
# phonetic rewriter. ~4 chars/token is a defensible average for English
# prose, so 40K tokens ~= 160K characters of payload per request.
_AI_REWRITE_MAX_TOKENS = 40_000
_AI_REWRITE_CHARS_PER_TOKEN = 4
_AI_REWRITE_MAX_CHARS = _AI_REWRITE_MAX_TOKENS * _AI_REWRITE_CHARS_PER_TOKEN

# Break priority used when a stretch of text is too long for one request:
# end of a nearby paragraph -> end of sentence (.?!) -> clause (,;:) -> word.
_AI_PARAGRAPH_BREAK_RE = re.compile(r'\n+')
_AI_SENTENCE_BREAK_RE = re.compile(r'(?<=[.!?])\s+')
_AI_CLAUSE_BREAK_RE = re.compile(r'(?<=[,;:])\s+')

# A chapter below this size (in rough tokens) is never sent to Gemini as its
# own request: it is merged with the leading part of the next chapter so a
# request is never a tiny, context-poor snippet.
_AI_MIN_AGGREGATE_TOKENS = 15_000
_AI_MIN_AGGREGATE_CHARS = _AI_MIN_AGGREGATE_TOKENS * _AI_REWRITE_CHARS_PER_TOKEN

# Marker used to join several chapters into one request. The model is asked to
# preserve it verbatim; the response is split back on it, and if it is not
# preserved the unit falls back to one request per piece. The token itself is
# bracketed with newlines when joining, but only the bare token is searched for
# on the way back, so a dropped blank line still splits correctly.
_AI_PART_MARKER = '<VX_PART_BREAK>'
_AI_PART_JOIN = '\n\n' + _AI_PART_MARKER + '\n\n'

# Below this many characters of slack it is not worth splitting the next
# chapter just to top up a small unit; close the unit as-is instead.
_AI_MIN_PREFIX_CHARS = 200


def _ai_split_hard(segment, max_chars):
    """Last resort for a segment with no usable punctuation: break at word
    boundaries, hard-slicing a single monstrously long token if needed."""
    out, current = [], ''
    for word in segment.split():
        candidate = f'{current} {word}'.strip() if current else word
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            out.append(current)
        while len(word) > max_chars:
            out.append(word[:max_chars])
            word = word[max_chars:]
        current = word
    if current:
        out.append(current)
    return out


def _ai_split_overlong(segment, max_chars, _level=0):
    """Break one block that exceeds max_chars at the coarsest real boundary
    available, preferring the end of a nearby paragraph: newline -> sentence
    (.?!) -> clause (,;:) -> word -> hard split. `_level` is the next boundary
    class to try; recursion into an over-long piece resumes one class finer so
    a boundary is never reused on the same text."""
    if len(segment) <= max_chars:
        return [segment]

    levels = (
        ('\n', _AI_PARAGRAPH_BREAK_RE),
        (' ', _AI_SENTENCE_BREAK_RE),
        (' ', _AI_CLAUSE_BREAK_RE),
    )
    for level in range(_level, len(levels)):
        sep, regex = levels[level]
        units = [u.strip() for u in regex.split(segment) if u.strip()]
        if len(units) <= 1:
            continue
        chunks, current = [], ''
        for unit in units:
            if len(unit) > max_chars:
                if current:
                    chunks.append(current)
                    current = ''
                chunks.extend(_ai_split_overlong(unit, max_chars, level + 1))
                continue
            candidate = f'{current}{sep}{unit}' if current else unit
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = unit
        if current:
            chunks.append(current)
        return chunks
    return _ai_split_hard(segment, max_chars)


def _ai_split_paragraphs(text, max_chars):
    """Split text into chunks of <= max_chars, joining whole paragraphs back
    together greedily. A paragraph that is itself over the limit is broken at
    the coarsest real boundary (see _ai_split_overlong) instead of being
    sliced at an arbitrary character offset."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p for p in text.split('\n\n') if p]
    chunks, current = [], ''
    for para in paragraphs:
        pieces = [para] if len(para) <= max_chars else _ai_split_overlong(para, max_chars)
        for piece in pieces:
            if not current:
                current = piece
                continue
            if len(current) + 2 + len(piece) <= max_chars:
                current = current + '\n\n' + piece
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def _ai_take_prefix(text, budget):
    """Split `text` into (prefix, rest) with `len(prefix) <= budget`, cutting
    at the coarsest real boundary that fits: paragraph -> sentence (.?!) ->
    clause (,;:) -> word, then a hard cut as a last resort. Both halves are
    stripped, so joining them back with a blank line restores the structure."""
    if budget <= 0:
        return '', text
    if len(text) <= budget:
        return text, ''

    window = text[:budget]
    for regex in (_AI_PARAGRAPH_BREAK_RE, _AI_SENTENCE_BREAK_RE, _AI_CLAUSE_BREAK_RE):
        matches = list(regex.finditer(window))
        if not matches:
            continue
        cut = matches[-1].end()
        prefix, rest = text[:cut].strip(), text[cut:].strip()
        if prefix:
            return prefix, rest

    cut = window.rfind(' ')
    if cut > 0:
        return text[:cut].strip(), text[cut + 1:].strip()
    return text[:budget].strip(), text[budget:].strip()


def _ai_pack_units(chapters, max_chars, min_chars):
    """Pack (chapter_index, text) pairs into AI request units of <= max_chars.

    Whole chapters are joined greedily until the next one would overflow. When
    a unit would close below min_chars, the leading part of the next chapter is
    pulled in (splitting that chapter at the coarsest boundary that fits) so
    small chapters ride along with their neighbour instead of becoming their
    own tiny request. Over-long chapters are pre-split into <= max_chars
    fragments, which participate in the same packing.

    Returns a list of units, each a list of (chapter_index, piece_text).
    """
    fragments = []
    for chapter_index, text in chapters:
        if len(text) <= max_chars:
            fragments.append((chapter_index, text))
        else:
            for piece in _ai_split_paragraphs(text, max_chars):
                fragments.append((chapter_index, piece))

    units = []
    current = []
    current_len = 0
    index = 0
    while index < len(fragments):
        chapter_index, text = fragments[index]
        sep_len = len(_AI_PART_JOIN) if current else 0
        if current and current_len + sep_len + len(text) > max_chars:
            budget = max_chars - current_len - sep_len
            if current_len < min_chars and budget >= _AI_MIN_PREFIX_CHARS:
                prefix, rest = _ai_take_prefix(text, budget)
                if prefix:
                    current.append((chapter_index, prefix))
                    units.append(current)
                    current, current_len = [], 0
                    fragments[index] = (chapter_index, rest)
                    if not rest:
                        index += 1
                    continue
                units.append(current)
                current, current_len = [], 0
                continue
            units.append(current)
            current, current_len = [], 0
            continue
        current.append((chapter_index, text))
        current_len += sep_len + len(text)
        index += 1
    if current:
        units.append(current)
    return units


def correct_phonetics_ai(text, api_key, model='gemini-3.1-flash-lite',
                         stop_event=None, post_event=None,
                         chapter_index=None, chapter_total=None, tts_engine='kokoro',
                         chatterbox_model=None):
    """
    Use Google Gemini AI to silently rewrite text for TTS-friendly pronunciation.

    For long inputs the text is split into chunks of at most ~40K tokens
    (≈ 160K chars) and each chunk is rewritten in its own request. The
    chunks are joined with blank lines to mirror the paragraph structure
    of the input.

    `tts_engine` selects which phonetic ruleset to apply: Kokoro accepts inline
    IPA overrides, while Chatterbox must be fed plain-English respellings only.

    Returns the corrected text string suitable for direct use as TTS input.
    If AI cannot be reached or returns invalid output, the original text is
    returned unchanged so the caller can still proceed with TTS.
    """
    if not text.strip():
        return text

    api_key = _sanitize_api_key(api_key)
    if not api_key or not api_key.startswith('AIza'):
        return text

    chunks = _ai_split_paragraphs(text, _AI_REWRITE_MAX_CHARS)
    if len(chunks) == 1:
        return _ai_rewrite_single_chunk(chunks[0], api_key, model, stop_event=stop_event,
                                        post_event=post_event, tts_engine=tts_engine,
                                        chatterbox_model=chatterbox_model)

    rewritten = []
    for idx, chunk in enumerate(chunks, start=1):
        if stop_event and stop_event.is_set():
            return text
        if post_event:
            post_event('CORE_AI_REWRITE', chapter_index=chapter_index,
                       chapter_total=chapter_total,
                       chunk_index=idx, chunk_total=len(chunks))
        out = _ai_rewrite_single_chunk(chunk, api_key, model, stop_event=stop_event,
                                       post_event=post_event, tts_engine=tts_engine,
                                       chatterbox_model=chatterbox_model)
        if out == chunk:
            rewritten.append(chunk)
        else:
            rewritten.append(out)
    return '\n\n'.join(rewritten)


def _ai_rewrite_combined_unit(unit, api_key, model, stop_event=None, post_event=None,
                              tts_engine='kokoro', chatterbox_model=None):
    """Rewrite several chapters in one Gemini request, joined by
    _AI_PART_MARKER, and return a list of rewritten pieces matching `unit`.

    Falls back to one request per piece if the model does not preserve the
    separators (or returns a piece count that does not match), so a stray
    marker can never merge or drop a chapter's text.
    """
    combined = _AI_PART_JOIN.join(text for _, text in unit)
    rewritten = _ai_rewrite_single_chunk(
        combined, api_key, model, stop_event=stop_event, post_event=post_event,
        tts_engine=tts_engine, chatterbox_model=chatterbox_model,
        part_separator=_AI_PART_MARKER, part_count=len(unit))
    parts = [part.strip() for part in rewritten.split(_AI_PART_MARKER)]
    if len(parts) == len(unit) and all(parts):
        return parts

    print('\033[93mAI rewrite did not preserve chapter separators; '
          'rewriting each chapter separately.\033[0m')
    return [
        _ai_rewrite_single_chunk(text, api_key, model, stop_event=stop_event,
                                 post_event=post_event, tts_engine=tts_engine,
                                 chatterbox_model=chatterbox_model)
        for _, text in unit
    ]


def correct_phonetics_ai_chapters(chapters, api_key, model='gemini-3.1-flash-lite',
                                  stop_event=None, post_event=None, tts_engine='kokoro',
                                  chatterbox_model=None, chapter_total=None):
    """Rewrite a run of chapters for TTS, packing them into AI requests.

    `chapters` is a list of (chapter_index, text). Chapters below
    ~15K tokens are merged with the leading part of the next chapter, and
    every request is kept at or below ~40K tokens (the chunks are packed whole
    chapters first, splitting a chapter at a real boundary only to top up a
    request that would otherwise close too small).

    Returns a dict mapping chapter_index -> rewritten text. Chapters are joined
    back with blank lines, mirroring correct_phonetics_ai's multi-chunk output.
    """
    if not chapters:
        return {}

    api_key = _sanitize_api_key(api_key)
    if not api_key or not api_key.startswith('AIza'):
        return {}

    units = _ai_pack_units(chapters, _AI_REWRITE_MAX_CHARS, _AI_MIN_AGGREGATE_CHARS)
    pieces_by_chapter = {}
    for index, unit in enumerate(units, start=1):
        if stop_event and stop_event.is_set():
            break
        if post_event:
            post_event('CORE_AI_REWRITE', chapter_index=unit[0][0],
                       chapter_total=chapter_total,
                       chunk_index=index, chunk_total=len(units))
        if len(unit) == 1:
            chapter_index, text = unit[0]
            rewritten = [_ai_rewrite_single_chunk(
                text, api_key, model, stop_event=stop_event, post_event=post_event,
                tts_engine=tts_engine, chatterbox_model=chatterbox_model)]
        else:
            rewritten = _ai_rewrite_combined_unit(
                unit, api_key, model, stop_event=stop_event, post_event=post_event,
                tts_engine=tts_engine, chatterbox_model=chatterbox_model)
        for (chapter_index, _text), out in zip(unit, rewritten):
            pieces_by_chapter.setdefault(chapter_index, []).append(out)

    return {chapter_index: '\n\n'.join(piece for piece in pieces if piece)
            for chapter_index, pieces in pieces_by_chapter.items()}


def _ai_rewrite_single_chunk(text, api_key, model, stop_event=None, post_event=None,
                             tts_engine='kokoro', chatterbox_model=None,
                             part_separator=None, part_count=0):
    """Single-chunk Gemini call used by correct_phonetics_ai. Falls back to
    the original text on any failure or invalid output (after retries).

    Uses a two-section output format (NAMES_FOUND then REWRITTEN_TEXT) to
    force the model to explicitly enumerate foreign/unusual proper nouns
    before it writes the rewrite. Shares the engine-specific ruleset with
    check_phonetic_transcription_ai so both prompts apply identical rules.

    When `part_separator` is set, `text` is several chapters joined by that
    exact marker; the model is told to preserve every marker so the caller can
    split the rewrite back into chapters.
    """
    engine_name = 'Chatterbox' if tts_engine == 'chatterbox' else 'Kokoro'
    rules = _phonetic_rules_for(tts_engine, chatterbox_model)

    separator_instruction = ''
    if part_separator and part_count > 1:
        separator_instruction = (
            f"The text is made of {part_count} chapters joined by the exact separator "
            f"`{part_separator}`. Rewrite the whole text as one, but keep every separator "
            "EXACTLY as written, in place, on its own — never translate, edit, remove, "
            "duplicate or reorder a separator. The separators are the only reason the "
            "rewritten chapters can be split back apart.\n\n"
        )

    def _do_call():
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(
            model=model,
            contents=(
                f"You are a phonetic preprocessing step for the {engine_name} TTS engine. "
                "You will do this in two steps, and your response MUST contain both "
                "sections below, in order, with the exact headers shown.\n\n"
                "STEP 1 — Find every proper noun in the text that does not use standard "
                "English spelling or pronunciation: place names, personal names, military "
                "unit/formation names, or any other proper noun with accented letters "
                "(é, è, ç, œ, ü, etc.), unusual consonant clusters, silent letters, or "
                "foreign-language endings. List each one exactly as it appears in the text, "
                "one per line. If there are none, write 'None found.'\n\n"
                "STEP 2 — Rewrite the full text so the TTS engine reads it correctly. Keep "
                "the meaning, punctuation, and sentence structure exactly the same, changing "
                "only what the rules below require. Apply these rules:\n\n"
                f"{rules}\n\n"
                "Every proper noun you listed in Step 1 MUST be changed in some way in the "
                "Step 2 rewrite.\n\n"
                f"{separator_instruction}"
                "FORMAT YOUR RESPONSE EXACTLY LIKE THIS (including the headers, nothing before or after):\n"
                "===NAMES_FOUND===\n"
                "<one name per line, or 'None found.'>\n"
                "===REWRITTEN_TEXT===\n"
                "<the full rewritten text, nothing else — no commentary, no quotes, no labels>\n\n"
                f"Text:\n\"\"\"\n{text}\n\"\"\""
            )
        )

    try:
        response = _call_gemini_with_retry(_do_call, stop_event=stop_event, post_event=post_event)
    except Exception as e:
        print(f'\033[91mGemini rewrite failed, keeping original text for this chunk: {e}\033[0m')
        return text

    raw = (response.text or '').strip()
    if not raw or raw.startswith('Error'):
        return text

    corrected = _extract_rewritten_section(raw)
    if corrected is None:
        print('\033[93mAI response missing REWRITTEN_TEXT header; using raw response as-is.\033[0m')
        corrected = raw

    if not corrected or len(corrected) > len(text) * 2:
        return text
    return corrected


def _extract_rewritten_section(raw: str):
    """Pull the REWRITTEN_TEXT section out of the two-section AI response.
    Returns None if the expected header isn't present, so the caller can
    fall back gracefully instead of silently shipping the names list (or
    other junk) to the TTS engine."""
    marker = '===REWRITTEN_TEXT==='
    idx = raw.find(marker)
    if idx == -1:
        return None

    names_section = raw[:idx].replace('===NAMES_FOUND===', '').strip()
    if names_section and names_section.lower() != 'none found.':
        found = [line.strip() for line in names_section.splitlines() if line.strip()]
        print(f'AI flagged {len(found)} foreign/unusual proper noun(s): {", ".join(found)}')

    return raw[idx + len(marker):].strip()
