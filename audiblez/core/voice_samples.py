"""audiblez.core.voice_samples - Kokoro-rendered voice presets that Chatterbox clones from."""


import json
import soundfile
import numpy as np
import time
from pathlib import Path
from kokoro import KPipeline

from .constants import DEFAULT_VOICE_SAMPLES_DIR, sample_rate
from .settings import load_settings
from .nlp import lang_code_from_voice, set_espeak_library
from .tts.kokoro import gen_audio_segments


# ---------------------------------------------------------------------------
# Voice sample library — Kokoro voices rendered as Chatterbox voice presets.
#
# Chatterbox ships no named voice list: `generate()` either uses its single
# built-in speaker or clones one from a reference WAV. To give Chatterbox the
# same voice dropdown as Kokoro, we render one short clip per Kokoro voice
# using Kokoro itself and hand that clip to Chatterbox as audio_prompt_path.
# Preset name == Kokoro voice name, so the two engines share one dropdown and
# no .wav path ever has to be managed by hand (a custom WAV stays available).
# ---------------------------------------------------------------------------

# Neutral English narration with wide phoneme coverage and no proper nouns
# (proper nouns get mispronounced by G2P and would pollute the clone).
VOICE_SAMPLE_TEXT = (
    "The north wind and the south wind move gently across the quiet valley, "
    "where a small brown fox jumps over a lazy dog beside the riverbank. "
    "She sells seashells by the shimmering seashore, and the old sailors sing "
    "songs about silver moonlit oceans and patient constellations."
)

# Bump when VOICE_SAMPLE_TEXT changes: samples recorded under an older script
# are reported as missing so they get rebuilt instead of silently cloned.
VOICE_SAMPLE_TEXT_VERSION = 1

VOICE_SAMPLE_INDEX = 'index.json'
VOICE_SAMPLE_MIN_SECONDS = 5.0
VOICE_SAMPLE_TRIM_THRESHOLD = 0.005   # ~ -46 dBFS, safely below speech level

CHATTERBOX_VOICE_SOURCES = ('preset', 'custom')


def voice_samples_dir(settings=None):
    """Directory holding the rendered samples; overridable via settings."""
    settings = settings if settings is not None else load_settings()
    raw = str(settings.get('voice_samples_dir') or '').strip()
    return Path(raw).expanduser() if raw else DEFAULT_VOICE_SAMPLES_DIR


def voice_sample_path(voice, settings=None):
    """Where the sample for `voice` lives, whether or not it has been built."""
    return voice_samples_dir(settings) / f'{voice}.wav'


def _read_sample_index(settings=None):
    try:
        with open(voice_samples_dir(settings) / VOICE_SAMPLE_INDEX) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_sample_index(index, settings=None):
    path = voice_samples_dir(settings) / VOICE_SAMPLE_INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(index, f, indent=2)
    tmp.replace(path)


def voice_sample_info(voice, settings=None):
    """Index entry for `voice` (duration, creation date), or {} if unknown."""
    return _read_sample_index(settings).get('voices', {}).get(voice, {})


def voice_sample_exists(voice, settings=None):
    """True only when the clip is on disk and was built from the current text."""
    if not voice:
        return False
    if not voice_sample_path(voice, settings).is_file():
        return False
    return int(voice_sample_info(voice, settings).get('text_version', 0)) == VOICE_SAMPLE_TEXT_VERSION


def missing_voice_samples(voices_to_check, settings=None):
    """Subset of `voices_to_check` that still needs a sample rendered."""
    return [v for v in dict.fromkeys(voices_to_check) if v and not voice_sample_exists(v, settings)]


def _trim_edge_silence(audio, threshold=VOICE_SAMPLE_TRIM_THRESHOLD):
    """Drop near-silent head/tail so Chatterbox conditions on speech, not gaps."""
    if audio.size == 0:
        return audio
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    start, end = int(loud[0]), int(loud[-1]) + 1
    if (end - start) < sample_rate * VOICE_SAMPLE_MIN_SECONDS:
        return audio
    return audio[start:end]


def generate_voice_sample(voice, speed=1.0, overwrite=False, stop_event=None,
                          settings=None, pipeline=None):
    """Render `voice` with Kokoro and store it as that voice's Chatterbox preset.

    Returns the sample path, or None when the run was cancelled, produced no
    audio, or the sample was already current and `overwrite` is False.
    An injected `pipeline` must have been built for `lang_code_from_voice(voice)`
    (generate_voice_samples() guarantees that).
    """
    if voice_sample_exists(voice, settings) and not overwrite:
        return voice_sample_path(voice, settings)

    set_espeak_library()
    if pipeline is None:
        pipeline = KPipeline(lang_code=lang_code_from_voice(voice))
    segments = gen_audio_segments(pipeline, VOICE_SAMPLE_TEXT, voice=voice,
                                  speed=speed, stop_event=stop_event)
    if stop_event and stop_event.is_set():
        return None
    if not segments:
        return None

    audio = _trim_edge_silence(np.concatenate(segments).astype(np.float32))
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (0.708 / peak)   # same target peak as chapter audio

    path = voice_sample_path(voice, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.wav.tmp')
    soundfile.write(tmp, audio, sample_rate, format='WAV', subtype='PCM_16')
    tmp.replace(path)

    index = _read_sample_index(settings)
    index['text_version'] = VOICE_SAMPLE_TEXT_VERSION
    index.setdefault('voices', {})[voice] = {
        'file': path.name,
        'sample_rate': sample_rate,
        'duration_sec': round(len(audio) / sample_rate, 2),
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'text_version': VOICE_SAMPLE_TEXT_VERSION,
    }
    _write_sample_index(index, settings)
    print(f'Voice sample for {voice} written to {path} '
          f'({len(audio) / sample_rate:.1f}s)')
    return path


def generate_voice_samples(voices_to_build, speed=1.0, overwrite=False,
                           stop_event=None, progress=None, settings=None):
    """Build samples for many voices, reusing one Kokoro pipeline per language.

    `progress(voice, path_or_None, done, total)` is called after each voice.
    Returns the list of paths written.
    """
    pending = list(dict.fromkeys(v for v in voices_to_build if v))
    if not overwrite:
        pending = missing_voice_samples(pending, settings)

    pipelines = {}
    written = []
    total = len(pending)
    for done, voice in enumerate(pending, start=1):
        if stop_event and stop_event.is_set():
            print('Voice sample build stopped by user.')
            break
        lang_code = lang_code_from_voice(voice)
        if lang_code not in pipelines:
            set_espeak_library()
            pipelines[lang_code] = KPipeline(lang_code=lang_code)
        path = generate_voice_sample(voice, speed=speed, overwrite=True,
                                     stop_event=stop_event, settings=settings,
                                     pipeline=pipelines[lang_code])
        if path:
            written.append(path)
        if progress:
            progress(voice, path, done, total)
    return written


def resolve_chatterbox_ref_audio(voice, source='preset', custom_path='', settings=None):
    """The one place that decides which WAV Chatterbox should clone.

    'preset' → the Kokoro-rendered sample for `voice`, empty when not built yet.
    'custom' → the user's own clip, empty when it has gone missing.
    """
    if source == 'custom':
        return custom_path if custom_path and Path(custom_path).is_file() else ''
    if not voice_sample_exists(voice, settings):
        return ''
    return str(voice_sample_path(voice, settings))
