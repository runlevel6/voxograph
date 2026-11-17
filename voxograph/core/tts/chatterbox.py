"""voxograph.core.tts.chatterbox - Persistent Chatterbox bridge and per-chapter synthesis."""


import tempfile
from queue import Queue, Empty
import json
import os
import soundfile
import numpy as np
import time
import shutil
import subprocess
import threading
from pathlib import Path

from ..constants import (
    CHATTERBOX_BRIDGE_DIR, CHATTERBOX_BRIDGE_PYTHON, CHATTERBOX_BRIDGE_SCRIPT,
    CHATTERBOX_DEFAULT_EXAGGERATION, CHATTERBOX_DEFAULT_CFG_WEIGHT,
    CHATTERBOX_DEFAULT_MODEL, CHATTERBOX_LANGUAGE_ID, CHATTERBOX_T3_MODEL,
    CHATTERBOX_CUDA_ALLOC_CONF,
    CHATTERBOX_MAX_CHUNK_TOKENS, CHATTERBOX_MAX_CHUNK_SILENCE_RATIO,
    CHATTERBOX_CHUNK_MAX_RETRIES, CHATTERBOX_SILENCE_THRESHOLD,
    CHATTERBOX_MAX_INTERNAL_SILENCE_MS, CHATTERBOX_INTERNAL_SILENCE_TARGET_MS,
    _CHATTERBOX_NOISE_MARKERS, sample_rate,
    CHATTERBOX_TURBO_DEFAULT_TEMPERATURE, CHATTERBOX_TURBO_DEFAULT_TOP_P,
    CHATTERBOX_TURBO_DEFAULT_TOP_K, CHATTERBOX_TURBO_DEFAULT_REPETITION_PENALTY,
    CHATTERBOX_TURBO_TEMPERATURE_RANGE, CHATTERBOX_TURBO_TOP_P_RANGE,
    CHATTERBOX_TURBO_TOP_K_RANGE, CHATTERBOX_TURBO_REPETITION_PENALTY_RANGE,
)
from ..settings import is_chatterbox_model
from ..utils import (
    _clamp_float, _clamp_int, _clamp_unit_float, _apply_fade,
    _compress_internal_silence, strfdelta,
)
from .chunking import split_chatterbox_text


def _is_chatterbox_progress_noise(line):
    """True for tqdm/pipeline progress chatter that should not be echoed."""
    return any(marker in line for marker in _CHATTERBOX_NOISE_MARKERS)


def chatterbox_bridge_request(text, output_path, device, ref_audio='',
                              model=CHATTERBOX_DEFAULT_MODEL,
                              language_id=CHATTERBOX_LANGUAGE_ID,
                              t3_model=CHATTERBOX_T3_MODEL,
                              exaggeration=CHATTERBOX_DEFAULT_EXAGGERATION,
                              cfg_weight=CHATTERBOX_DEFAULT_CFG_WEIGHT,
                              turbo_temperature=CHATTERBOX_TURBO_DEFAULT_TEMPERATURE,
                              turbo_top_p=CHATTERBOX_TURBO_DEFAULT_TOP_P,
                              turbo_top_k=CHATTERBOX_TURBO_DEFAULT_TOP_K,
                              turbo_repetition_penalty=CHATTERBOX_TURBO_DEFAULT_REPETITION_PENALTY):
    """Build the JSON request body for the Chatterbox bridge.

    Single source of truth shared by the persistent `ChatterboxBridge` and the
    GUI's one-shot preview subprocess, so the bridge schema and the default
    language/t3 constants can never drift between whole-book synthesis and
    preview.
    """
    return {
        'text': text,
        'output_path': str(output_path),
        'device': device,
        'language_id': language_id,
        'audio_prompt_path': ref_audio,
        't3_model': t3_model,
        'model': model,
        'exaggeration': exaggeration,
        'cfg_weight': cfg_weight,
        'turbo_temperature': turbo_temperature,
        'turbo_top_p': turbo_top_p,
        'turbo_top_k': turbo_top_k,
        'turbo_repetition_penalty': turbo_repetition_penalty,
    }


class ChatterboxError(RuntimeError):
    """Raised when the Chatterbox bridge fails or dies."""


class ChatterboxCancelled(ChatterboxError):
    """Raised when generation was interrupted by a stop_event."""


class ChatterboxBridge:
    """Persistent subprocess wrapper around the Chatterbox venv bridge.

    Starts `generate.py --serve` once, then talks newline-delimited JSON to
    it, so the TTS model stays loaded for the whole book.
    """

    def __init__(self, device='cuda', ref_audio='', language_id=CHATTERBOX_LANGUAGE_ID,
                 t3_model=CHATTERBOX_T3_MODEL, python=None, script=None,
                 exaggeration=CHATTERBOX_DEFAULT_EXAGGERATION,
                 cfg_weight=CHATTERBOX_DEFAULT_CFG_WEIGHT,
                 model=CHATTERBOX_DEFAULT_MODEL,
                 turbo_temperature=CHATTERBOX_TURBO_DEFAULT_TEMPERATURE,
                 turbo_top_p=CHATTERBOX_TURBO_DEFAULT_TOP_P,
                 turbo_top_k=CHATTERBOX_TURBO_DEFAULT_TOP_K,
                 turbo_repetition_penalty=CHATTERBOX_TURBO_DEFAULT_REPETITION_PENALTY):
        self.device = device
        self.ref_audio = ref_audio
        self.language_id = language_id
        self.t3_model = t3_model
        self.model = model if is_chatterbox_model(model) else CHATTERBOX_DEFAULT_MODEL
        self.exaggeration = _clamp_unit_float(
            exaggeration, CHATTERBOX_DEFAULT_EXAGGERATION)
        self.cfg_weight = _clamp_unit_float(cfg_weight, CHATTERBOX_DEFAULT_CFG_WEIGHT)
        # Turbo's sampling knobs. Multilingual ignores them (it keeps its own
        # defaults), but they travel in every payload so the bridge picks the
        # right set per model.
        self.turbo_temperature = _clamp_float(
            turbo_temperature, *CHATTERBOX_TURBO_TEMPERATURE_RANGE,
            CHATTERBOX_TURBO_DEFAULT_TEMPERATURE)
        self.turbo_top_p = _clamp_float(
            turbo_top_p, *CHATTERBOX_TURBO_TOP_P_RANGE,
            CHATTERBOX_TURBO_DEFAULT_TOP_P)
        self.turbo_top_k = _clamp_int(
            turbo_top_k, *CHATTERBOX_TURBO_TOP_K_RANGE,
            CHATTERBOX_TURBO_DEFAULT_TOP_K)
        self.turbo_repetition_penalty = _clamp_float(
            turbo_repetition_penalty, *CHATTERBOX_TURBO_REPETITION_PENALTY_RANGE,
            CHATTERBOX_TURBO_DEFAULT_REPETITION_PENALTY)
        self.python = Path(python) if python else CHATTERBOX_BRIDGE_PYTHON
        self.script = Path(script) if script else CHATTERBOX_BRIDGE_SCRIPT
        if not self.python.is_file():
            raise ChatterboxError(f'Chatterbox python interpreter not found: {self.python}')
        if not self.script.is_file():
            raise ChatterboxError(f'Chatterbox bridge script not found: {self.script}')

        self._responses = Queue()
        # The Turbo model (~6.9 GB) nearly fills the 8 GB Quadro P4000, so the
        # default (non-expandable) CUDA allocator fragments under repeated
        # generate() calls and the subprocess dies with OOM. PyTorch's
        # expandable-segments allocator grows segments on demand and never
        # leaves holes behind, keeping the footprint contiguous. Inherit the
        # parent env so the user can override it, then set the value.
        bridge_env = dict(os.environ)
        bridge_env['PYTORCH_CUDA_ALLOC_CONF'] = CHATTERBOX_CUDA_ALLOC_CONF
        self.proc = subprocess.Popen(
            [str(self.python), str(self.script), '--serve'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', bufsize=1, env=bridge_env)
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        threading.Thread(target=self._read_responses, daemon=True).start()

    def _drain_stderr(self):
        # Chatterbox/tqdm write many progress-bar updates (with \r) to stderr.
        # Drain them so the pipe never fills, but don't echo the noise for a
        # whole book; keep real diagnostics (warnings, tracebacks, errors).
        try:
            for line in self.proc.stderr:
                line = line.rstrip('\n')
                if not line or _is_chatterbox_progress_noise(line):
                    continue
                print(f'[chatterbox] {line}')
        except Exception:
            pass

    def _read_responses(self):
        try:
            for line in self.proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # Third-party stdout noise that slipped through; ignore it.
                    continue
                self._responses.put(obj)
        finally:
            self._responses.put(None)

    def generate(self, text, output_path, stop_event=None):
        """Synthesize `text` to `output_path`; return the bridge's result dict."""
        payload = json.dumps(chatterbox_bridge_request(
            text=text,
            output_path=output_path,
            device=self.device,
            ref_audio=self.ref_audio,
            model=self.model,
            language_id=self.language_id,
            t3_model=self.t3_model,
            exaggeration=self.exaggeration,
            cfg_weight=self.cfg_weight,
            turbo_temperature=self.turbo_temperature,
            turbo_top_p=self.turbo_top_p,
            turbo_top_k=self.turbo_top_k,
            turbo_repetition_penalty=self.turbo_repetition_penalty,
        ))
        try:
            self.proc.stdin.write(payload + '\n')
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise ChatterboxError(f'Chatterbox bridge is not running: {e}')

        while True:
            if stop_event and stop_event.is_set():
                raise ChatterboxCancelled('Stopped by user.')
            try:
                obj = self._responses.get(timeout=0.25)
            except Empty:
                if self.proc.poll() is not None and self._responses.empty():
                    raise ChatterboxError('Chatterbox bridge exited unexpectedly.')
                continue
            if obj is None:
                raise ChatterboxError('Chatterbox bridge closed unexpectedly.')
            if obj.get('success'):
                return obj
            raise ChatterboxError(obj.get('error') or 'Unknown Chatterbox error.')

    def close(self):
        proc = getattr(self, 'proc', None)
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass



def gen_audio_segments_chatterbox(bridge, text, stats=None, max_chunks=None,
                                  post_event=None, stop_event=None, on_halfway=None):
    """Synthesize `text` with Chatterbox in <=300-char chunks.

    Returns (audio_segments, sample_rate). Each chunk is generated by the
    persistent bridge to a temp WAV, read back as a mono float array, faded,
    and returned for the caller to concatenate and normalize.

    `on_halfway`, when set, is called once the current piece is ~50% spoken
    (by characters, which tracks audio duration far better than a raw chunk
    count would). The caller uses it to kick off the next Gemini rewrite in
    the background while synthesis of the current piece continues.
    """
    chunks = split_chatterbox_text(text)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    if not chunks:
        return [], sample_rate

    segments = []
    write_sr = sample_rate
    processed = 0
    started = time.time()
    # Splitting consumes the whitespace at each chunk/paragraph boundary, so the
    # chunks sum to slightly fewer chars than the source. Add that back on the
    # final chunk so per-chapter progress can actually reach 100%.
    separator_chars = max(0, len(text) - sum(len(c) for c in chunks))
    piece_total = sum(len(c) for c in chunks) + separator_chars or 1
    piece_done = 0
    halfway_fired = False

    def _maybe_halfway():
        nonlocal halfway_fired
        if not halfway_fired and piece_done >= piece_total / 2:
            halfway_fired = True
            if on_halfway is not None:
                on_halfway()

    tmp_dir = tempfile.mkdtemp(prefix='voxograph_chatterbox_')
    try:
        for idx, chunk in enumerate(chunks, start=1):
            if stop_event and stop_event.is_set():
                print('Synthesis stopped by user.')
                break
            chunk_path = Path(tmp_dir) / f'chunk_{idx:05d}.wav'
            
            # Retry loop for degenerate chunks (too many tokens or too much silence)
            chunk_retries = 0
            while True:
                try:
                    result = bridge.generate(chunk, chunk_path, stop_event=stop_event)
                except ChatterboxCancelled:
                    print('Synthesis stopped by user.')
                    break
                write_sr = int(result.get('sample_rate') or write_sr)
                if not chunk_path.is_file():
                    raise ChatterboxError('Chatterbox did not write a chunk WAV.')
                audio, _sr = soundfile.read(chunk_path, dtype='float32')
                try:
                    chunk_path.unlink()
                except OSError:
                    pass
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                audio = _apply_fade(np.asarray(audio, dtype=np.float32))
                
                # Degenerate chunk detection. The token count is a duration
                # proxy (25 tokens/s), not a quality signal, so only a chunk
                # that ran into Turbo's generation cap is degenerate: it never
                # emitted the stop token and padded the tail with garbage.
                is_degenerate = False
                degenerate_reason = ''
                token_count = result.get('speech_token_count')
                if token_count is not None and token_count >= CHATTERBOX_MAX_CHUNK_TOKENS:
                    is_degenerate = True
                    degenerate_reason = (f'{token_count} tokens (hit the '
                                         f'{CHATTERBOX_MAX_CHUNK_TOKENS}-token '
                                         f'cap, no stop token)')
                # Silence ratio check (head/tail only; internal gaps are
                # compressed below). A chunk with no audible samples at all is
                # always degenerate — the old `loud.size > 0` guard skipped it,
                # so a fully silent generation was accepted as-is.
                loud = np.flatnonzero(np.abs(audio) > CHATTERBOX_SILENCE_THRESHOLD)
                if loud.size == 0:
                    is_degenerate = True
                    degenerate_reason = 'entirely silent'
                else:
                    silent_ratio = 1.0 - (loud[-1] - loud[0] + 1) / audio.size
                    if silent_ratio > CHATTERBOX_MAX_CHUNK_SILENCE_RATIO:
                        is_degenerate = True
                        degenerate_reason = (
                            f'{silent_ratio*100:.1f}% silent '
                            f'(>{CHATTERBOX_MAX_CHUNK_SILENCE_RATIO*100:.0f}%)')

                if is_degenerate:
                    if chunk_retries < CHATTERBOX_CHUNK_MAX_RETRIES:
                        chunk_retries += 1
                        print(f'\nChunk {idx}: {degenerate_reason}, retrying '
                              f'({chunk_retries}/{CHATTERBOX_CHUNK_MAX_RETRIES})...')
                        continue  # retry this chunk
                    print(f'\nChunk {idx}: {degenerate_reason}, accepting after '
                          f'{CHATTERBOX_CHUNK_MAX_RETRIES} retries.')

                # Turbo can leave multi-second dead air inside a chunk (e.g. an
                # 18 s gap after a sentence) that the head/tail ratio cannot
                # see. Compress those runs to a natural pause.
                before = audio.size
                audio = _compress_internal_silence(
                    audio,
                    threshold=CHATTERBOX_SILENCE_THRESHOLD,
                    max_silence_ms=CHATTERBOX_MAX_INTERNAL_SILENCE_MS,
                    target_ms=CHATTERBOX_INTERNAL_SILENCE_TARGET_MS,
                )
                removed_secs = (before - audio.size) / float(write_sr)
                if removed_secs >= 3.0:
                    print(f'\nChunk {idx}: shortened {removed_secs:.1f}s of '
                          f'internal silence to natural pauses.')

                segments.append(audio)
                break  # accept this chunk (either good, or max retries reached)
            
            if stop_event and stop_event.is_set():
                break

            counted = len(chunk) + (separator_chars if idx == len(chunks) else 0)
            processed += counted
            piece_done += counted
            _maybe_halfway()
            if stats:
                stats.processed_chars += counted
                elapsed = max(time.time() - started, 1e-6)
                stats.chars_per_sec = max(1.0, processed / elapsed)
                tts_share = getattr(stats, 'tts_progress_share', 1.0)
                stats.progress = int(stats.processed_chars / stats.total_chars * tts_share * 100)
                remaining_tts = (stats.total_chars - stats.processed_chars) / stats.chars_per_sec
                remaining_encode = getattr(stats, 'estimated_encode_secs', 0)
                stats.eta = strfdelta(remaining_tts + remaining_encode)
                if post_event:
                    post_event('CORE_PROGRESS', stats=stats)
                print(f'\rChatterbox chunk {idx}/{len(chunks)} — '
                      f'{stats.progress}% (ETA {stats.eta})', end='', flush=True)
        print()  # End the line after all chunks are done
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return segments, write_sr
