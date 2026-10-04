#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatterbox Multilingual V3 (English-locked) audio generation bridge.

Reads a JSON request from stdin, generates a WAV, and writes JSON result
to stdout. Designed to be invoked from another Python venv via subprocess.

Two modes:

  * one-shot (default): read one JSON object from stdin, write one JSON
    result to stdout, exit. This is what the preview button uses.

  * --serve: read newline-delimited JSON requests from stdin forever and
    write one newline-delimited JSON result per request to stdout. The
    model stays loaded in-process, so full-book synthesis pays the model
    load cost exactly once. Used by voxograph core.main().
"""
import sys

# Third-party libraries (notably perth) print to stdout while loading the model.
# Keep the real stdout pristine so the parent process receives only our JSON
# result, and route everything else to stderr.
_REAL_STDOUT = sys.stdout
sys.stdout = sys.stderr

import json
import os
import traceback
import numpy as np
import torch
import soundfile as sf
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


MODEL = None


def _to_numpy(wav):
    if hasattr(wav, "cpu"):
        wav = wav.cpu().numpy()
    return np.asarray(wav).flatten().astype(np.float32)


def _get_model(device, t3_model):
    global MODEL
    if MODEL is None:
        MODEL = ChatterboxMultilingualTTS.from_pretrained(
            device=device, t3_model=t3_model
        )
    return MODEL


def _generate_request(data):
    """Generate one WAV from a parsed request dict and return the result dict."""
    text = data.get("text", "")
    if not text.strip():
        raise ValueError("Missing or empty 'text'")

    output_path = data.get("output_path", "")
    if not output_path:
        raise ValueError("Missing 'output_path'")

    device = data.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    language_id = data.get("language_id", "en")
    audio_prompt_path = data.get("audio_prompt_path")
    t3_model = data.get("t3_model", "t3_mtl23ls_v3.safetensors")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    model = _get_model(device, t3_model)

    kwargs = {"language_id": language_id}
    if audio_prompt_path:
        kwargs["audio_prompt_path"] = audio_prompt_path

    wav = model.generate(text, **kwargs)
    wav = _to_numpy(wav)

    sf.write(output_path, wav, model.sr)

    return {
        "success": True,
        "sample_rate": int(model.sr),
        "output_path": output_path,
        "device": device,
    }


def _emit(result):
    _REAL_STDOUT.write(json.dumps(result) + "\n")
    _REAL_STDOUT.flush()


def run_once():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            raise ValueError("Empty request payload")
        data = json.loads(raw)
        _emit(_generate_request(data))
    except Exception as exc:
        result = {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        sys.stderr.write(json.dumps(result) + "\n")
        sys.stderr.flush()
        sys.exit(1)


def serve():
    """Persistent request/response loop: one JSON per line, model kept warm.

    A failed request is reported as {"success": false, ...} and the loop
    keeps going, so one bad chunk cannot take down a whole book synthesis.
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            result = _generate_request(data)
        except Exception as exc:
            result = {
                "success": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        _emit(result)


if __name__ == "__main__":
    if "--serve" in sys.argv[1:]:
        serve()
    else:
        run_once()
