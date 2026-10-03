"""audiblez.core.tts.chatterbox - Persistent Chatterbox bridge and per-chapter synthesis."""


import os
import tempfile
from queue import Queue, Empty
import json
import soundfile
import numpy as np
import time
import shutil
import subprocess
import threading
from pathlib import Path

from ..constants import (
    CHATTERBOX_DEFAULT_EXAGGERATION, CHATTERBOX_DEFAULT_CFG_WEIGHT,
    CHATTERBOX_DEFAULT_MODEL, sample_rate,
)
from ..settings import is_chatterbox_model
from ..utils import _clamp_unit_float, _apply_fade, strfdelta
from .chunking import split_chatterbox_text


CHATTERBOX_BRIDGE_DIR = Path(
    os.environ.get('AUDIBLEZ_CHATTERBOX_BRIDGE_DIR', '/home/vlad/chatterbox_venv'))
CHATTERBOX_BRIDGE_PYTHON = CHATTERBOX_BRIDGE_DIR / 'bin' / 'python3'
CHATTERBOX_BRIDGE_SCRIPT = CHATTERBOX_BRIDGE_DIR / 'generate.py'
CHATTERBOX_T3_MODEL = 't3_mtl23ls_v3.safetensors'
CHATTERBOX_LANGUAGE_ID = 'en'
_CHATTERBOX_NOISE_MARKERS = ('it/s', '?it/s', 'Sampling:', 'Fetching', '%|')


def _is_chatterbox_progress_noise(line):
    """True for tqdm/pipeline progress chatter that should not be echoed."""
    return any(marker in line for marker in _CHATTERBOX_NOISE_MARKERS)


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
                 model=CHATTERBOX_DEFAULT_MODEL):
        self.device = device
        self.ref_audio = ref_audio
        self.language_id = language_id
        self.t3_model = t3_model
        self.model = model if is_chatterbox_model(model) else CHATTERBOX_DEFAULT_MODEL
        self.exaggeration = _clamp_unit_float(
            exaggeration, CHATTERBOX_DEFAULT_EXAGGERATION)
        self.cfg_weight = _clamp_unit_float(cfg_weight, CHATTERBOX_DEFAULT_CFG_WEIGHT)
        self.python = Path(python) if python else CHATTERBOX_BRIDGE_PYTHON
        self.script = Path(script) if script else CHATTERBOX_BRIDGE_SCRIPT
        if not self.python.is_file():
            raise ChatterboxError(f'Chatterbox python interpreter not found: {self.python}')
        if not self.script.is_file():
            raise ChatterboxError(f'Chatterbox bridge script not found: {self.script}')

        self._responses = Queue()
        self.proc = subprocess.Popen(
            [str(self.python), str(self.script), '--serve'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', bufsize=1)
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
        payload = json.dumps({
            'text': text,
            'output_path': str(output_path),
            'device': self.device,
            'language_id': self.language_id,
            'audio_prompt_path': self.ref_audio,
            't3_model': self.t3_model,
            'model': self.model,
            'exaggeration': self.exaggeration,
            'cfg_weight': self.cfg_weight,
        })
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
                                  post_event=None, stop_event=None):
    """Synthesize `text` with Chatterbox in <=300-char chunks.

    Returns (audio_segments, sample_rate). Each chunk is generated by the
    persistent bridge to a temp WAV, read back as a mono float array, faded,
    and returned for the caller to concatenate and normalize.
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
    tmp_dir = tempfile.mkdtemp(prefix='audiblez_chatterbox_')
    try:
        for idx, chunk in enumerate(chunks, start=1):
            if stop_event and stop_event.is_set():
                print('Synthesis stopped by user.')
                break
            chunk_path = Path(tmp_dir) / f'chunk_{idx:05d}.wav'
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
            segments.append(audio)

            counted = len(chunk) + (separator_chars if idx == len(chunks) else 0)
            processed += counted
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
                print(f'Chatterbox chunk {idx}/{len(chunks)} — '
                      f'{stats.progress}% (ETA {stats.eta})')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return segments, write_sr
