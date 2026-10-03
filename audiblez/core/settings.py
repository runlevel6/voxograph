"""audiblez.core.settings - Settings persistence (single source of truth, fix #9)."""


import json
from pathlib import Path

from .constants import (
    CONFIG_FILE, DEFAULT_VOICE_SAMPLES_DIR,
    CHATTERBOX_DEFAULT_EXAGGERATION, CHATTERBOX_DEFAULT_CFG_WEIGHT,
    CHATTERBOX_DEFAULT_MODEL, CHATTERBOX_MODELS,
)
from .utils import _clamp_unit_float


def is_chatterbox_model(value):
    """True when `value` names a supported Chatterbox model; else False."""
    return value in CHATTERBOX_MODELS
def load_settings():
    """Loads settings from the JSON configuration file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            settings = json.load(f)
            if 'output_folder' in settings:
                settings['output_folder'] = str(Path(settings['output_folder']))
            settings.setdefault('gemini_api_key', '')
            settings.setdefault('gemini_model', 'gemini-3.1-flash-lite')
            settings.setdefault('gemini_enabled', False)
            settings.setdefault('last_open_dir', str(Path.home()))
            settings.setdefault('tts_engine', 'kokoro')
            settings.setdefault('chatterbox_ref_audio', '')
            settings.setdefault('chatterbox_device', 'cuda')
            settings.setdefault('chatterbox_voice_source', 'preset')
            settings.setdefault('chatterbox_model', CHATTERBOX_DEFAULT_MODEL)
            settings.setdefault('chatterbox_exaggeration', CHATTERBOX_DEFAULT_EXAGGERATION)
            settings.setdefault('chatterbox_cfg_weight', CHATTERBOX_DEFAULT_CFG_WEIGHT)
            settings.setdefault('voice_samples_dir', str(DEFAULT_VOICE_SAMPLES_DIR))
            return settings
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(output_folder, voice, speed=1.0, gemini_api_key='', gemini_model='gemini-3.1-flash-lite', gemini_enabled=False, last_open_dir='', tts_engine='kokoro', chatterbox_ref_audio='', chatterbox_device='cuda', chatterbox_voice_source='preset', voice_samples_dir='', chatterbox_exaggeration=CHATTERBOX_DEFAULT_EXAGGERATION, chatterbox_cfg_weight=CHATTERBOX_DEFAULT_CFG_WEIGHT, chatterbox_model=CHATTERBOX_DEFAULT_MODEL):
    """
    Saves settings to the JSON configuration file.
    Accepts an optional speed parameter so both core and UI write a
    consistent schema to the same file.
    """
    settings = {
        'output_folder': str(output_folder),
        'voice': voice,
        'speed': float(speed),
        'gemini_api_key': gemini_api_key,
        'gemini_model': gemini_model,
        'gemini_enabled': gemini_enabled,
        'last_open_dir': last_open_dir,
        'tts_engine': tts_engine,
        'chatterbox_ref_audio': chatterbox_ref_audio,
        'chatterbox_device': chatterbox_device,
        'chatterbox_voice_source': chatterbox_voice_source,
        'chatterbox_model': (chatterbox_model if is_chatterbox_model(chatterbox_model)
                             else CHATTERBOX_DEFAULT_MODEL),
        'chatterbox_exaggeration': _clamp_unit_float(
            chatterbox_exaggeration, CHATTERBOX_DEFAULT_EXAGGERATION),
        'chatterbox_cfg_weight': _clamp_unit_float(
            chatterbox_cfg_weight, CHATTERBOX_DEFAULT_CFG_WEIGHT),
        'voice_samples_dir': voice_samples_dir or str(DEFAULT_VOICE_SAMPLES_DIR),
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"Settings saved to {CONFIG_FILE}.")
    except IOError as e:
        print(f"Error saving settings to {CONFIG_FILE}: {e}")
