"""voxograph.core.gemini - Gemini provider (google-genai).

Gemini-specific code only: the SDK client, generation config, one
text-generation call, and the ordered Gemini model catalogue. Provider
dispatch, retry/fallback, and the phonetic rewrite pipeline live in
``voxograph.core.ai``.
"""


from google import genai
from google.genai import types

from .constants import (
    GEMINI_MODELS,
    _AI_MAX_OUTPUT_TOKENS,
    _GEMINI_REQUEST_TIMEOUT_MS,
)


def _ai_generate_config():
    """Build the generation config shared by every Gemini call."""
    return types.GenerateContentConfig(
        max_output_tokens=_AI_MAX_OUTPUT_TOKENS,
        # These calls declare no tools, so AFC has nothing to execute. Without
        # this the SDK still runs its AFC wrapper and logs a multi-line
        # "Direct use of automatic function calling (AFC) ..." warning before
        # the first request. Disabling it keeps the request path plain and the
        # console clean (the code never uses tools, so no Chat session needed).
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


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
# Chat.send_message_stream) instead. The plain no-tools call below is the only
# place the code may still hit client.models.generate_content directly.


def _gemini_client(api_key):
    """Build a Gemini client whose HTTP requests time out after
    _GEMINI_REQUEST_TIMEOUT_SECS seconds."""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=_GEMINI_REQUEST_TIMEOUT_MS),
    )


def gemini_models_for():
    """Ordered Gemini fallback list (newest/flashiest first)."""
    return GEMINI_MODELS


def _gemini_generate(api_key, model, contents):
    """One non-streaming Gemini text-generation call."""
    client = _gemini_client(api_key)
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=_ai_generate_config(),
    )
