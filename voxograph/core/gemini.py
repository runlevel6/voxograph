"""voxograph.core.gemini - Gemini retry helper, phonetic rules, and AI pronunciation rewrites."""


import math
import threading
import time
from google import genai
from google.genai import types

from .constants import (
    GEMINI_MODELS,
    _AI_CLAUSE_BREAK_RE, _AI_MAX_OUTPUT_TOKENS, _AI_MIN_AGGREGATE_CHARS,
    _AI_MIN_AGGREGATE_TOKENS, _AI_MIN_PREFIX_CHARS, _AI_PARAGRAPH_BREAK_RE,
    _AI_PART_JOIN, _AI_PART_MARKER, _AI_REWRITE_CHARS_PER_TOKEN,
    _AI_REWRITE_MAX_CHARS, _AI_REWRITE_MAX_TOKENS, _AI_SENTENCE_BREAK_RE,
    _CHATTERBOX_PHONETIC_RULES, _CHATTERBOX_TAG_RULES,
    _GEMINI_REQUEST_TIMEOUT_MS, _GEMINI_REQUEST_TIMEOUT_SECS, _GEMINI_RETRY_DELAYS_SECS,
    _NON_RETRYABLE_ERROR_MARKERS, _PHONETIC_RULES,
)


def _ai_generate_config():
    """Build the generation config shared by every Gemini call in this module."""
    return types.GenerateContentConfig(max_output_tokens=_AI_MAX_OUTPUT_TOKENS)


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


class GeminiTimeoutError(RuntimeError):
    """No response from a Gemini model within _GEMINI_REQUEST_TIMEOUT_SECS.

    The message carries no non-retryable marker, so the model-fallback loop
    treats it as transient and moves straight on to the next model."""


def _gemini_client(api_key):
    """Build a Gemini client whose HTTP requests time out after
    _GEMINI_REQUEST_TIMEOUT_SECS seconds."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=_GEMINI_REQUEST_TIMEOUT_MS),
    )


def _render_progress_bar(elapsed, total, width=20):
    """Render a text progress bar showing elapsed/total progress."""
    fraction = max(0.0, min(1.0, elapsed / total))
    filled = int(width * fraction)
    bar = '█' * filled + '░' * (width - filled)
    return f'[{bar}]'


def _call_gemini_with_timeout(func, *args, timeout=_GEMINI_REQUEST_TIMEOUT_SECS,
                              stop_event=None, label=None, **kwargs):
    """Run func(*args, **kwargs) under a watchdog and print a live progress bar.

    The call runs in a daemon worker thread so the main thread can update the
    progress bar each second. If func does not return within `timeout`
    seconds, a GeminiTimeoutError is raised so the caller can try the next
    model. The worker also carries the same HTTP timeout on its client, so a
    hung request is abandoned on both the watchdog and the socket side.
    """
    outcome = {}

    def _run():
        try:
            outcome['value'] = func(*args, **kwargs)
        except BaseException as exc:  # reported to the caller below
            outcome['exc'] = exc

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    start = time.monotonic()
    label = label or 'model'
    printed = False

    while worker.is_alive():
        elapsed = time.monotonic() - start
        remaining = timeout - elapsed
        if remaining <= 0:
            break
        if stop_event and stop_event.is_set():
            break
        bar = _render_progress_bar(elapsed, timeout)
        print(f'\033[96m[Gemini] {label}: waiting for response {bar} '
              f'{math.ceil(remaining)}s left\033[0m', end='\r', flush=True)
        printed = True
        worker.join(min(1.0, remaining))

    if printed:
        # Move to a fresh line so subsequent output does not overwrite the bar.
        print()

    if stop_event and stop_event.is_set():
        raise RuntimeError('Stopped by user.')
    if worker.is_alive():
        raise GeminiTimeoutError(
            f'No response from {label} within {timeout}s.')
    if 'exc' in outcome:
        raise outcome['exc']
    return outcome['value']


def _try_gemini_models(func, *args, stop_event=None, post_event=None, **kwargs):
    """
    Try calling func(model, *args, **kwargs) with each model in GEMINI_MODELS
    until one succeeds. func must accept `model` as its first argument.

    Each model is tried once; a busy/unavailable model (retryable error, e.g.
    503) waits 1 minute and retries the SAME model once before moving on — a
    demand spike is usually temporary and the first model is often still the
    best bet. A model that produces no response within
    _GEMINI_REQUEST_TIMEOUT_SECS (60s) is abandoned and the next model is
    tried. Only when EVERY model fails in a pass does the whole list wait a
    backoff delay and get retried (1, 2, 3, then 4 minutes); after the last
    round the final exception is raised.

    Permanent errors (bad key, permission denied, invalid argument) are not
    model-specific, so they abort immediately without trying other models.
    """
    last_exc = None
    total_rounds = len(_GEMINI_RETRY_DELAYS_SECS) + 1
    for round_index in range(total_rounds):
        if round_index > 0:
            print(f'\033[96m[Gemini] Retry round {round_index}/{total_rounds - 1} '
                  f'(all {len(GEMINI_MODELS)} models)\033[0m')
        for i, model in enumerate(GEMINI_MODELS):
            if stop_event and stop_event.is_set():
                if last_exc:
                    raise last_exc
                raise RuntimeError('Stopped by user.')
            print(f'\033[96m[Gemini] Trying model {i+1}/{len(GEMINI_MODELS)}: {model}\033[0m')
            try:
                return _call_gemini_with_timeout(func, model, *args,
                                                 stop_event=stop_event, label=model, **kwargs)
            except GeminiTimeoutError as e:
                last_exc = e
                print(f'\033[93m[Gemini] Model {model} gave no response within '
                      f'{_GEMINI_REQUEST_TIMEOUT_SECS}s. Trying next model...\033[0m')
                continue
            except Exception as e:
                last_exc = e
                if stop_event and stop_event.is_set():
                    raise
                # Permanent error: not model-specific, so no point trying others.
                if not _is_retryable_gemini_error(e):
                    print(f'\033[91m[Gemini] Model {model} failed with non-retryable error: {e}\033[0m')
                    print('\033[91m[Gemini] Stopping model fallback (error likely not model-specific)\033[0m')
                    raise
                # Busy/unavailable (e.g. 503): wait 1 minute and retry the
                # SAME model once — a demand spike is usually temporary and
                # the first model is often still the best bet. If the retry
                # also fails, move on to the next model.
                print(f'\033[93m[Gemini] Model {model} failed: {e}. Waiting 1 '
                      f'minute before retrying {model}...\033[0m')
                _sleep_with_stop_event(60, stop_event)
                if stop_event and stop_event.is_set():
                    if last_exc:
                        raise last_exc
                    raise RuntimeError('Stopped by user.')
                print(f'\033[96m[Gemini] Retrying model {i+1}/{len(GEMINI_MODELS)}: '
                      f'{model}\033[0m')
                try:
                    return _call_gemini_with_timeout(func, model, *args,
                                                     stop_event=stop_event, label=model, **kwargs)
                except Exception as retry_exc:
                    last_exc = retry_exc
                    if not _is_retryable_gemini_error(retry_exc):
                        print(f'\033[91m[Gemini] Model {model} retry failed with '
                              f'non-retryable error: {retry_exc}\033[0m')
                        print('\033[91m[Gemini] Stopping model fallback '
                              '(error likely not model-specific)\033[0m')
                        raise
                    print(f'\033[93m[Gemini] Model {model} retry also failed: '
                          f'{retry_exc}. Trying next model...\033[0m')
        if round_index < len(_GEMINI_RETRY_DELAYS_SECS):
            delay = _GEMINI_RETRY_DELAYS_SECS[round_index]
            print(f'\033[93m[Gemini] All {len(GEMINI_MODELS)} models failed. Waiting '
                  f'{delay // 60} minute(s) before retrying the full list '
                  f'(round {round_index + 2}/{total_rounds})...\033[0m')
            _sleep_with_stop_event(delay, stop_event)
    if post_event and last_exc and _is_retryable_gemini_error(last_exc):
        post_event('CORE_AI_RETRY_EXHAUSTED', message=f'All models failed. Last error: {last_exc}')
    print(f'\033[91m[Gemini] All {len(GEMINI_MODELS)} models failed across '
          f'{total_rounds} round(s). Giving up.\033[0m')
    raise last_exc


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
    total_attempts = len(_GEMINI_RETRY_DELAYS_SECS) + 1
    for attempt in range(total_attempts):
        if stop_event and stop_event.is_set():
            if last_exc:
                raise last_exc
            raise RuntimeError('Stopped by user.')
        if attempt > 0:
            print(f'\033[96m[Gemini] Retry attempt {attempt}/{total_attempts - 1}\033[0m')
        try:
            return _call_gemini_with_timeout(func, *args, stop_event=stop_event, **kwargs)
        except Exception as e:
            last_exc = e
            more_attempts_left = attempt < len(_GEMINI_RETRY_DELAYS_SECS)
            if more_attempts_left and _is_retryable_gemini_error(e):
                delay = _GEMINI_RETRY_DELAYS_SECS[attempt]
                print(f'\033[93m[Gemini] Call failed ({e}). Retrying in {delay // 60} '
                      f'minute(s)... (attempt {attempt + 2}/{total_attempts})\033[0m')
                _sleep_with_stop_event(delay, stop_event)
            else:
                if not more_attempts_left:
                    print(f'\033[91m[Gemini] All retry attempts exhausted ({total_attempts} total)\033[0m')
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
                                    chatterbox_model=None, use_model_fallback=True):
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

    If `use_model_fallback` is True (default), tries GEMINI_MODELS in order
    until one succeeds. If False, uses the specified `model` only.
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

    def _do_call(m):
        client = _gemini_client(api_key)
        return client.models.generate_content(
            model=m,
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
            ),
            config=_ai_generate_config(),
        )

    try:
        if use_model_fallback:
            response = _try_gemini_models(_do_call, stop_event=stop_event)
        else:
            response = _call_gemini_with_retry(lambda: _do_call(model), stop_event=stop_event)
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
                                          chatterbox_model=chatterbox_model,
                                          use_model_fallback=use_model_fallback)

    return (
        f"{analysis}\n\n"
        f"{'=' * 60}\n"
        f"REWRITTEN TEXT (this is what will be sent to the TTS engine)\n"
        f"{'=' * 60}\n\n"
        f"{rewritten_text}"
    )


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
                         chatterbox_model=None, use_model_fallback=True):
    """
    Use Google Gemini AI to silently rewrite text for TTS-friendly pronunciation.

    For long inputs the text is split into chunks of at most ~8K tokens
    (≈ 32K chars) and each chunk is rewritten in its own request. The
    chunks are joined with blank lines to mirror the paragraph structure
    of the input.

    `tts_engine` selects which phonetic ruleset to apply: Kokoro accepts inline
    IPA overrides, while Chatterbox must be fed plain-English respellings only.

    If `use_model_fallback` is True (default), tries GEMINI_MODELS in order
    until one succeeds. If False, uses the specified `model` only.

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
                                        chatterbox_model=chatterbox_model, use_model_fallback=use_model_fallback)

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
                                       chatterbox_model=chatterbox_model, use_model_fallback=use_model_fallback)
        if out == chunk:
            rewritten.append(chunk)
        else:
            rewritten.append(out)
    return '\n\n'.join(rewritten)


def _ai_rewrite_combined_unit(unit, api_key, model, stop_event=None, post_event=None,
                              tts_engine='kokoro', chatterbox_model=None, use_model_fallback=True):
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
        part_separator=_AI_PART_MARKER, part_count=len(unit), use_model_fallback=use_model_fallback)
    parts = [part.strip() for part in rewritten.split(_AI_PART_MARKER)]
    if len(parts) == len(unit) and all(parts):
        return parts

    print('\033[93mAI rewrite did not preserve chapter separators; '
          'rewriting each chapter separately.\033[0m')
    return [
        _ai_rewrite_single_chunk(text, api_key, model, stop_event=stop_event,
                                 post_event=post_event, tts_engine=tts_engine,
                                 chatterbox_model=chatterbox_model, use_model_fallback=use_model_fallback)
        for _, text in unit
    ]


def _rewrite_packed_unit(unit, api_key, model, stop_event=None, post_event=None,
                         tts_engine='kokoro', chatterbox_model=None, use_model_fallback=True):
    """Rewrite exactly one packed AI unit and return the rewritten piece texts
    in the same order as `unit`. A single-piece unit is a plain request; a
    multi-piece unit is joined with the part marker and split back."""
    if len(unit) == 1:
        return [_ai_rewrite_single_chunk(
            unit[0][1], api_key, model, stop_event=stop_event, post_event=post_event,
            tts_engine=tts_engine, chatterbox_model=chatterbox_model, use_model_fallback=use_model_fallback)]
    return _ai_rewrite_combined_unit(
        unit, api_key, model, stop_event=stop_event, post_event=post_event,
        tts_engine=tts_engine, chatterbox_model=chatterbox_model, use_model_fallback=use_model_fallback)


def _iter_correct_phonetics_ai_pieces(chapters, api_key, model='gemini-3.1-flash-lite',
                                      stop_event=None, post_event=None, tts_engine='kokoro',
                                      chatterbox_model=None, chapter_total=None, use_model_fallback=True):
    """Rewrite a run of chapters for TTS and yield every piece as it returns.

    Same packing and per-request rewriting as
    iter_correct_phonetics_ai_chapters, but WITHOUT its all-or-nothing chapter
    gate: each rewritten piece is handed to the caller the moment its request
    comes back, tagged with `is_last_piece` (True on the final piece of a
    chapter). That lets the synthesis loop start speaking a long chapter after
    its first fragment instead of waiting for the whole chapter to be
    rewritten — a 300K+ char chapter is split into a dozen units, so the gate
    used to leave the TTS loop silent for a dozen back-to-back Gemini requests.

    Yields (chapter_index, rewritten_piece, is_last_piece). A chapter that fits
    in one piece yields exactly one tuple with is_last_piece=True.
    """
    if not chapters:
        return

    api_key = _sanitize_api_key(api_key)
    if not api_key or not api_key.startswith('AIza'):
        return

    units = _ai_pack_units(chapters, _AI_REWRITE_MAX_CHARS, _AI_MIN_AGGREGATE_CHARS)

    # How many pieces each chapter is split into across the units, so the last
    # one can be flagged for the caller.
    pieces_total = {}
    for unit in units:
        for chapter_index, _text in unit:
            pieces_total[chapter_index] = pieces_total.get(chapter_index, 0) + 1

    pieces_seen = {}
    for index, unit in enumerate(units, start=1):
        if stop_event and stop_event.is_set():
            return
        if post_event:
            post_event('CORE_AI_REWRITE', chapter_index=unit[0][0],
                       chapter_total=chapter_total,
                       chunk_index=index, chunk_total=len(units))
        rewritten = _rewrite_packed_unit(
            unit, api_key, model, stop_event=stop_event, post_event=post_event,
            tts_engine=tts_engine, chatterbox_model=chatterbox_model, use_model_fallback=use_model_fallback)
        for (chapter_index, _text), out in zip(unit, rewritten):
            seen = pieces_seen.get(chapter_index, 0) + 1
            pieces_seen[chapter_index] = seen
            yield chapter_index, out, seen >= pieces_total[chapter_index]


def iter_correct_phonetics_ai_chapters(chapters, api_key, model='gemini-3.1-flash-lite',
                                       stop_event=None, post_event=None, tts_engine='kokoro',
                                       chatterbox_model=None, chapter_total=None, use_model_fallback=True):
    """Rewrite a run of chapters for TTS, one packed AI request at a time.

    `chapters` is a list of (chapter_index, text). Chapters below
    ~1K tokens are merged with the leading part of the next chapter, and
    every request is kept at or below ~8K tokens (whole chapters are packed
    first, splitting a chapter at a real boundary only to top up a request
    that would otherwise close too small).

    Yields (chapter_index, rewritten_text) as soon as every piece of that
    chapter has been rewritten, in chapter order. Because this is a generator,
    the caller drives it lazily: TTS can run on each chapter before the next
    AI request is issued, spreading the API calls out in time (AI -> TTS ->
    AI -> TTS) instead of firing every request up front.

    If `use_model_fallback` is True (default), tries GEMINI_MODELS in order
    until one succeeds. If False, uses the specified `model` only.
    """
    pieces_by_chapter = {}
    for chapter_index, out, is_last in _iter_correct_phonetics_ai_pieces(
            chapters, api_key, model=model, stop_event=stop_event, post_event=post_event,
            tts_engine=tts_engine, chatterbox_model=chatterbox_model,
            chapter_total=chapter_total, use_model_fallback=use_model_fallback):
        pieces = pieces_by_chapter.setdefault(chapter_index, [])
        pieces.append(out)
        if is_last:
            yield chapter_index, '\n\n'.join(piece for piece in pieces if piece)


def correct_phonetics_ai_chapters(chapters, api_key, model='gemini-3.1-flash-lite',
                                  stop_event=None, post_event=None, tts_engine='kokoro',
                                  chatterbox_model=None, chapter_total=None, use_model_fallback=True):
    """Rewrite a run of chapters for TTS, packing them into AI requests.

    Eager variant of iter_correct_phonetics_ai_chapters: consumes the whole
    series up front and returns a dict mapping chapter_index -> rewritten text.
    Chapters are joined back with blank lines, mirroring correct_phonetics_ai's
    multi-chunk output.

    If `use_model_fallback` is True (default), tries GEMINI_MODELS in order
    until one succeeds. If False, uses the specified `model` only.
    """
    return dict(iter_correct_phonetics_ai_chapters(
        chapters, api_key, model=model, stop_event=stop_event, post_event=post_event,
        tts_engine=tts_engine, chatterbox_model=chatterbox_model,
        chapter_total=chapter_total, use_model_fallback=use_model_fallback))


def _ai_rewrite_single_chunk(text, api_key, model='gemini-3.1-flash-lite', stop_event=None, post_event=None,
                             tts_engine='kokoro', chatterbox_model=None,
                             part_separator=None, part_count=0, use_model_fallback=True):
    """Single-chunk Gemini call used by correct_phonetics_ai. Falls back to
    the original text on any failure or invalid output (after retries).

    Uses a two-section output format (NAMES_FOUND then REWRITTEN_TEXT) to
    force the model to explicitly enumerate foreign/unusual proper nouns
    before it writes the rewrite. Shares the engine-specific ruleset with
    check_phonetic_transcription_ai so both prompts apply identical rules.

    When `part_separator` is set, `text` is several chapters joined by that
    exact marker; the model is told to preserve every marker so the caller can
    split the rewrite back into chapters.

    If `use_model_fallback` is True (default), tries GEMINI_MODELS in order
    until one succeeds. If False, uses the specified `model` only.
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

    def _do_call(m):
        client = _gemini_client(api_key)
        return client.models.generate_content(
            model=m,
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
            ),
            config=_ai_generate_config(),
        )

    try:
        if use_model_fallback:
            response = _try_gemini_models(_do_call, stop_event=stop_event, post_event=post_event)
        else:
            response = _call_gemini_with_retry(lambda: _do_call(model), stop_event=stop_event, post_event=post_event)
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
