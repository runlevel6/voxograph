"""audiblez.core.gemini - Gemini retry helper, phonetic rules, and AI pronunciation rewrites."""


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
    "Do NOT output IPA symbols, stress marks, slashes, square brackets, or espeak "
    "[[...]] syntax. Use only ordinary English letters and normal punctuation.\n\n"
    "Prosody: existing punctuation already controls intonation — "
    "; : , . ! ? — … \" ( ) \u201c \u201d all shape phrasing and pitch. "
    "Do not remove or alter punctuation; it is meaningful for Chatterbox."
)


def _phonetic_rules_for(tts_engine):
    """Pick the rewrite ruleset for the engine that will actually speak the text."""
    return _CHATTERBOX_PHONETIC_RULES if tts_engine == 'chatterbox' else _PHONETIC_RULES


def check_phonetic_transcription_ai(text, api_key, model='gemini-3.1-flash-lite',
                                    stop_event=None, tts_engine='kokoro'):
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

    rules = _phonetic_rules_for(tts_engine)
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
                                          stop_event=stop_event, tts_engine=tts_engine)

    return (
        f"{analysis}\n\n"
        f"{'=' * 60}\n"
        f"REWRITTEN TEXT (this is what will be sent to the TTS engine)\n"
        f"{'=' * 60}\n\n"
        f"{rewritten_text}"
    )

# Rough char-to-token ratio used to size Gemini requests for the
# phonetic rewriter. ~4 chars/token is a defensible average for English
# prose, so 300K tokens ~= 1.2M characters of payload per request.
_AI_REWRITE_MAX_TOKENS = 300_000
_AI_REWRITE_CHARS_PER_TOKEN = 4
_AI_REWRITE_MAX_CHARS = _AI_REWRITE_MAX_TOKENS * _AI_REWRITE_CHARS_PER_TOKEN


def _ai_split_paragraphs(text, max_chars):
    """Split text into chunks of <= max_chars on whitespace, joining whole
    paragraphs back together greedily. Falls back to hard word-splitting
    when a single paragraph exceeds max_chars (rare for cleaned EPUBs)."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p for p in text.split('\n\n') if p]
    chunks, current = [], ''
    for para in paragraphs:
        if not current:
            current = para
            continue
        if len(current) + 2 + len(para) <= max_chars:
            current = current + '\n\n' + para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)

    if any(len(c) > max_chars for c in chunks):
        refined = []
        for c in chunks:
            if len(c) <= max_chars:
                refined.append(c)
                continue
            for i in range(0, len(c), max_chars):
                refined.append(c[i:i + max_chars])
        chunks = refined
    return chunks


def correct_phonetics_ai(text, api_key, model='gemini-3.1-flash-lite',
                         stop_event=None, post_event=None,
                         chapter_index=None, chapter_total=None, tts_engine='kokoro'):
    """
    Use Google Gemini AI to silently rewrite text for TTS-friendly pronunciation.

    For long inputs the text is split into chunks of at most ~300K tokens
    (≈ 1.2M chars) and each chunk is rewritten in its own request. The
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
                                        post_event=post_event, tts_engine=tts_engine)

    rewritten = []
    for idx, chunk in enumerate(chunks, start=1):
        if stop_event and stop_event.is_set():
            return text
        if post_event:
            post_event('CORE_AI_REWRITE', chapter_index=chapter_index,
                       chapter_total=chapter_total,
                       chunk_index=idx, chunk_total=len(chunks))
        out = _ai_rewrite_single_chunk(chunk, api_key, model, stop_event=stop_event,
                                       post_event=post_event, tts_engine=tts_engine)
        if out == chunk:
            rewritten.append(chunk)
        else:
            rewritten.append(out)
    return '\n\n'.join(rewritten)

def _ai_rewrite_single_chunk(text, api_key, model, stop_event=None, post_event=None,
                             tts_engine='kokoro'):
    """Single-chunk Gemini call used by correct_phonetics_ai. Falls back to
    the original text on any failure or invalid output (after retries).

    Uses a two-section output format (NAMES_FOUND then REWRITTEN_TEXT) to
    force the model to explicitly enumerate foreign/unusual proper nouns
    before it writes the rewrite. Shares the engine-specific ruleset with
    check_phonetic_transcription_ai so both prompts apply identical rules.
    """
    engine_name = 'Chatterbox' if tts_engine == 'chatterbox' else 'Kokoro'
    rules = _phonetic_rules_for(tts_engine)

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
                "the meaning, punctuation, and sentence structure exactly the same. Apply "
                "these rules:\n\n"
                f"{rules}\n\n"
                "Every proper noun you listed in Step 1 MUST be changed in some way in the "
                "Step 2 rewrite.\n\n"
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
