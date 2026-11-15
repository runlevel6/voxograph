"""voxograph.core.deepseek - DeepSeek provider (OpenAI-compatible Chat Completions)."""


import json
import urllib.error
import urllib.request

from .constants import (
    DEEPSEEK_CHAT_COMPLETIONS_URL,
    DEEPSEEK_MODELS,
    _DEEPSEEK_MAX_OUTPUT_TOKENS,
    _DEEPSEEK_REQUEST_TIMEOUT_SECS,
)


class _AIResponse:
    """Minimal response wrapper shared by both providers; callers only need
    the generated text, matching the google-genai response's `.text`."""

    __slots__ = ('text',)

    def __init__(self, text):
        self.text = text


class DeepSeekAPIError(RuntimeError):
    """A non-success response from the DeepSeek API."""

    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(f'DeepSeek API error {status_code}: {message}')


# 400/401/402/403/404/422 cannot be fixed by retrying or by another model.
_DEEPSEEK_NON_RETRYABLE_STATUS = frozenset({400, 401, 402, 403, 404, 422})


def _is_deepseek_model(model):
    """True when `model` names a DeepSeek model rather than a Gemini one."""
    return str(model or '').strip().lower().startswith('deepseek')


def deepseek_models_for(model):
    """DeepSeek fallback order: the requested model first, then the rest."""
    return (model,) + tuple(m for m in DEEPSEEK_MODELS if m != model)


def _deepseek_error_detail(body):
    """Extract a human-readable message from a DeepSeek error body."""
    try:
        payload = json.loads(body)
    except (TypeError, ValueError):
        return (body or '').strip()[:500] or 'empty response body'
    error = payload.get('error', payload) if isinstance(payload, dict) else payload
    if isinstance(error, dict):
        detail = error.get('message') or error.get('detail') or json.dumps(error)
    else:
        detail = str(error)
    return detail[:500]


def _deepseek_generate(api_key, model, contents):
    """One non-streaming DeepSeek Chat Completions call.

    Thinking mode is disabled: the phonetic rewriter wants the final rewrite,
    not a chain-of-thought, and non-thinking mode gives the requested
    max_tokens budget entirely to the output.
    """
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': contents}],
        'max_tokens': _DEEPSEEK_MAX_OUTPUT_TOKENS,
        'thinking': {'type': 'disabled'},
        'stream': False,
    }
    request = urllib.request.Request(
        DEEPSEEK_CHAT_COMPLETIONS_URL,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(
                request, timeout=_DEEPSEEK_REQUEST_TIMEOUT_SECS) as response:
            body = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as exc:
        detail = _deepseek_error_detail(exc.read().decode('utf-8', 'replace'))
        raise DeepSeekAPIError(exc.code, detail) from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f'DeepSeek request failed: {exc.reason}') from exc

    try:
        data = json.loads(body)
    except ValueError as exc:
        raise RuntimeError(f'DeepSeek returned invalid JSON: {body[:300]}') from exc

    choices = data.get('choices') or []
    if not choices:
        detail = _deepseek_error_detail(body)
        raise RuntimeError(f'DeepSeek returned no choices: {detail}')
    message = choices[0].get('message') or {}
    finish_reason = choices[0].get('finish_reason')
    if finish_reason == 'length':
        raise RuntimeError(
            'DeepSeek response hit the max_tokens limit and was truncated.')
    text = (message.get('content') or '').strip()
    if not text:
        raise RuntimeError(
            f'DeepSeek returned no text content (finish_reason={finish_reason}).')
    return _AIResponse(text)