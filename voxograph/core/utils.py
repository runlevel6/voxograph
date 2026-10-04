"""voxograph.core.utils - Small shared helpers (clamping, fades, time formatting)."""


import numpy as np
from string import Formatter

from .constants import sample_rate


def _clamp_unit_float(value, default):
    """Coerce `value` to a float in [0, 1], falling back to `default`."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return float(default)
def _apply_fade(audio, fade_ms=5.0):
    """Apply a short fade-in/out so segment boundaries never click."""
    fade_samples = min(int(sample_rate * fade_ms / 1000.0), len(audio) // 4)
    if fade_samples <= 0:
        return audio
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=audio.dtype)
    audio = audio.copy()
    audio[:fade_samples] *= ramp
    audio[-fade_samples:] *= ramp[::-1]
    return audio
def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s'):
    remainder = int(tdelta)
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)
