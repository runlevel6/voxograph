"""audiblez.core.tts.kokoro - Kokoro TTS segment generation."""


import soundfile
import numpy as np
import subprocess
from kokoro import KPipeline

from ..constants import sample_rate, DEFAULT_VOICE
from ..nlp import get_nlp, lang_code_from_voice
from ..utils import _apply_fade, strfdelta


def gen_audio_segments(pipeline, text, voice, speed, stats=None, max_sentences=None, post_event=None, stop_event=None):
    nlp = get_nlp()  # fix #4: use cached instance
    audio_segments = []
    doc = nlp(text)
    sentences = list(doc.sents)
    for i, sent in enumerate(sentences):
        if stop_event and stop_event.is_set():
            print('Synthesis stopped by user.')
            break
        if max_sentences is not None and i >= max_sentences:  # fix #3: >= not >, and treat 0 correctly
            break
        for gs, ps, audio in pipeline(sent.text, voice=voice, speed=speed, split_pattern=r'\n\n\n'):
            # Fix: Kokoro returns a torch tensor; convert to numpy array so
            # that numpy operations (linspace dtype, arithmetic) work correctly.
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()

            # Apply a short 5 ms fade-in / fade-out to each segment to
            # prevent clicks caused by DC-offset discontinuities at
            # sentence boundaries.
            audio_segments.append(_apply_fade(np.asarray(audio)))
        if stats:
            stats.processed_chars += len(sent.text)
            tts_share = getattr(stats, 'tts_progress_share', 1.0)
            stats.progress = int(
                stats.processed_chars / stats.total_chars * tts_share * 100)
            remaining_tts = (stats.total_chars - stats.processed_chars) / stats.chars_per_sec
            remaining_encode = getattr(stats, 'estimated_encode_secs', 0)
            stats.eta = strfdelta(remaining_tts + remaining_encode)
            if post_event:
                post_event('CORE_PROGRESS', stats=stats)
            print(f'Estimated time remaining: {stats.eta}')
            print('Progress:', f'{stats.progress}%\n')
    return audio_segments


def gen_text(text, voice=DEFAULT_VOICE, output_file='text.wav', speed=1, play=False):
    pipeline = KPipeline(lang_code=lang_code_from_voice(voice))
    audio_segments = gen_audio_segments(pipeline, text, voice=voice, speed=speed)
    final_audio = np.concatenate(audio_segments)
    soundfile.write(output_file, final_audio, sample_rate)
    if play:
        subprocess.run(['ffplay', '-autoexit', '-nodisp', output_file])
