"""voxograph.core.constants - Global constants shared across the package."""


from pathlib import Path


sample_rate = 24000

_SPOKEN_CHARS_PER_SEC = 12.5  # ~150 wpm * 5 chars / 60 s
_AAC_ENCODE_RT_FACTOR = 50    # ffmpeg aac runs ~50x realtime

# Fix #10: use path relative to this file so it resolves consistently
# regardless of where the process is launched from.
CONFIG_FILE = Path(__file__).resolve().parent.parent / 'config.json'

# Default home for the rendered voice samples (one WAV per Kokoro voice) that
# Chatterbox clones from. Kept outside the package directory on purpose: a
# pip reinstall of voxograph wipes site-packages but must not wipe samples.
DEFAULT_VOICE_SAMPLES_DIR = Path.home() / '.voxograph' / 'voice_samples'

# Chatterbox has no speed knob — these two shape delivery instead.
# Exaggeration: how expressive/dramatic the speech is. Higher values also
# speed the pacing up naturally. CFG weight: how closely the output sticks to
# the reference clip's style/pacing. Lower (0.3) slows a too-fast reference
# down and clarifies it; 0 avoids accent bleeding on cross-lingual transfers.
CHATTERBOX_DEFAULT_EXAGGERATION = 0.5
CHATTERBOX_DEFAULT_CFG_WEIGHT = 0.5

# Turbo ignores exaggeration/CFG weight (its T3 config disables emotion
# conditioning and hard-codes CFG to 0.0), but exposes these sampling knobs.
# Defaults mirror ChatterboxTurboTTS.generate() so an unset value is a no-op.
CHATTERBOX_TURBO_DEFAULT_TEMPERATURE = 0.8
CHATTERBOX_TURBO_DEFAULT_TOP_P = 0.95
CHATTERBOX_TURBO_DEFAULT_TOP_K = 1000
CHATTERBOX_TURBO_DEFAULT_REPETITION_PENALTY = 1.2
CHATTERBOX_TURBO_TEMPERATURE_RANGE = (0.0, 2.0)
CHATTERBOX_TURBO_TOP_P_RANGE = (0.0, 1.0)
CHATTERBOX_TURBO_TOP_K_RANGE = (0, 100000)
CHATTERBOX_TURBO_REPETITION_PENALTY_RANGE = (1.0, 2.0)

# Which Chatterbox model to synthesize with. 'multilingual' is the Multilingual
# V3 model (English-locked here, 23 languages available); 'turbo' is the smaller,
# faster English-only model with native paralinguistic tags but no CFG/exaggeration.
CHATTERBOX_MODEL_MULTILINGUAL = 'multilingual'
CHATTERBOX_MODEL_TURBO = 'turbo'
CHATTERBOX_MODELS = (CHATTERBOX_MODEL_MULTILINGUAL, CHATTERBOX_MODEL_TURBO)
CHATTERBOX_DEFAULT_MODEL = CHATTERBOX_MODEL_MULTILINGUAL
DEFAULT_VOICE = 'af_heart'
