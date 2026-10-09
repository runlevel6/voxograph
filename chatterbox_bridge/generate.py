#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chatterbox audio generation bridge (Multilingual V3 and Turbo).

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

# Pin torch's intra-op thread count once, before any model is loaded.
# Chatterbox does not benefit from torch's default (all cores) and
# oversubscribes the CPU; 4 threads keeps generation steady.
torch.set_num_threads(4)


MODEL = None
# Which Chatterbox family member is currently loaded: 'multilingual' or 'turbo'.
MODEL_KIND = None

# Supported Chatterbox models. Both clone a voice from a reference WAV; they
# differ in speed/size (Turbo is ~350M, English-only) and in the controls they
# honor (Turbo has no CFG/exaggeration; it adds native paralinguistic tags like
# [laugh] and [cough]).
CHATTERBOX_MODELS = ('multilingual', 'turbo')


def _to_numpy(wav):
    if hasattr(wav, "cpu"):
        wav = wav.cpu().numpy()
    return np.asarray(wav).flatten().astype(np.float32)


def _get_model(device, t3_model, model_kind):
    global MODEL, MODEL_KIND
    if MODEL is None or MODEL_KIND != model_kind:
        if model_kind == 'turbo':
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            MODEL = ChatterboxTurboTTS.from_pretrained(device=device)
        else:
            MODEL = ChatterboxMultilingualTTS.from_pretrained(
                device=device, t3_model=t3_model
            )
        MODEL_KIND = model_kind
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
    model_kind = str(data.get("model", "multilingual")).lower()
    if model_kind not in CHATTERBOX_MODELS:
        model_kind = "multilingual"
    # Chatterbox has no speed control; these two shape delivery instead.
    # exaggeration: expressiveness/drama (higher also speeds pacing up).
    # cfg_weight: adherence to the reference clip's style/pacing (0.3 slows
    # and clarifies a fast reference; 0 avoids accent bleeding cross-lingually).
    # Both are ignored by Turbo, which warns when either is non-zero, so keep
    # them at 0.0 for Turbo.
    exaggeration = float(data.get("exaggeration", 0.5))
    cfg_weight = float(data.get("cfg_weight", 0.5))
    if model_kind == "turbo":
        exaggeration = 0.0
        cfg_weight = 0.0

    # Turbo exposes sampling knobs that Multilingual does not; defaults mirror
    # ChatterboxTurboTTS.generate() so an absent key changes nothing.
    temperature = max(0.0, float(data.get("turbo_temperature", 0.8)))
    top_p = min(1.0, max(0.0, float(data.get("turbo_top_p", 0.95))))
    top_k = max(0, int(data.get("turbo_top_k", 1000)))
    repetition_penalty = max(0.0, float(data.get("turbo_repetition_penalty", 1.2)))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    model = _get_model(device, t3_model, model_kind)

    if model_kind == "turbo":
        # Turbo has no language_id parameter; it is English-only.
        kwargs = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
    else:
        kwargs = {
            "language_id": language_id,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
        }
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
        "model": model_kind,
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "turbo_temperature": temperature,
        "turbo_top_p": top_p,
        "turbo_top_k": top_k,
        "turbo_repetition_penalty": repetition_penalty,
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
