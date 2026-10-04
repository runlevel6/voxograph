"""voxograph.core.pipeline - Top-level audiobook synthesis pipeline (core.main)."""


import torch.cuda
import soundfile
import numpy as np
import time
import shutil
from types import SimpleNamespace
from pathlib import Path
from kokoro import KPipeline
from ebooklib import epub

from .constants import (
    sample_rate, _SPOKEN_CHARS_PER_SEC, _AAC_ENCODE_RT_FACTOR,
    DEFAULT_VOICE, DEFAULT_VOICE_SAMPLES_DIR,
    CHATTERBOX_DEFAULT_EXAGGERATION, CHATTERBOX_DEFAULT_CFG_WEIGHT,
    CHATTERBOX_DEFAULT_MODEL, CHATTERBOX_MODEL_TURBO,
)
from .utils import _clamp_unit_float, strfdelta
from .settings import load_settings, save_settings, is_chatterbox_model
from .nlp import get_nlp, lang_code_from_voice, set_espeak_library
from .gemini import (
    correct_phonetics_ai, _sanitize_api_key, _AI_REWRITE_MAX_TOKENS,
)
from .epub import (
    find_cover, find_document_chapters_and_extract_texts,
    find_good_chapters, pick_chapters, print_selected_chapters,
)
from .tts.kokoro import gen_audio_segments
from .tts.chatterbox import (
    ChatterboxBridge, ChatterboxError, gen_audio_segments_chatterbox,
)
from .voice_samples import generate_voice_sample, resolve_chatterbox_ref_audio
from .audio import create_index_file, create_m4b, delete_wav_files


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


# _m4b_progress_reporter removed: progress is now driven by real-time parsing
# of ffmpeg's stderr inside create_m4b, using the actual speed=Nx value.


def main(file_path, voice=None, pick_manually=False, speed=1, output_folder='.',
         max_chapters=None, max_sentences=None, selected_chapters=None, post_event=None,
         stop_event=None, tts_engine=None):
    if post_event:
        post_event('CORE_STARTED')

    # Ensure spaCy is ready before we do anything else
    get_nlp()

    settings = load_settings()

    output_folder = Path(output_folder) if output_folder != '.' else Path(settings.get('output_folder', '.'))
    voice = voice if voice else settings.get('voice', DEFAULT_VOICE)

    ai_enabled = bool(settings.get('gemini_enabled', False))
    ai_api_key = settings.get('gemini_api_key', '') or ''
    ai_model = settings.get('gemini_model', 'gemini-3.1-flash-lite') or 'gemini-3.1-flash-lite'
    if ai_enabled:
        if not _sanitize_api_key(ai_api_key).startswith('AIza'):
            print('\033[93m' + 'AI Phonetic Check is enabled but the API key is missing or invalid. '
                  'Falling back to original text for the whole book.' + '\033[0m')
            ai_enabled = False
        else:
            print(f'AI Phonetic Check enabled (model={ai_model}). Chapters will be '
                  f'rewritten in chunks of ~{_AI_REWRITE_MAX_TOKENS:,} tokens before TTS.')

    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder set to: {output_folder}")
    print(f"Voice selected: {voice}")

    filename = Path(file_path).name
    extension = '.epub'
    try:
        book = epub.read_epub(file_path)
    except Exception as e:
        print(f'\033[91mFailed to read EPUB file "{file_path}": {e}\033[0m')
        if post_event:
            post_event('CORE_ERROR', message=f'Failed to read EPUB file: {e}')
        raise

    meta_title = book.get_metadata('DC', 'title')
    title = meta_title[0][0] if meta_title else ''
    meta_creator = book.get_metadata('DC', 'creator')
    creator = meta_creator[0][0] if meta_creator else ''

    cover_maybe = find_cover(book)
    cover_image = cover_maybe.get_content() if cover_maybe else b""
    if cover_maybe:
        print(f'Found cover image {cover_maybe.file_name} in {cover_maybe.media_type} format')

    document_chapters = find_document_chapters_and_extract_texts(book, ai_enabled=ai_enabled)

    if not selected_chapters:
        if pick_manually is True:
            selected_chapters = pick_chapters(document_chapters)
        else:
            selected_chapters = find_good_chapters(document_chapters)
    print_selected_chapters(document_chapters, selected_chapters)
    texts = [c.extracted_text for c in selected_chapters]

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print('\033[91m' + 'ffmpeg not found. Please install ffmpeg to create mp3 and m4b audiobook files.' + '\033[0m')

    stats = SimpleNamespace(
        total_chars=sum(map(len, texts)),
        processed_chars=0,
        chars_per_sec=500 if torch.cuda.is_available() else 50)
    _est_audio_secs = stats.total_chars / _SPOKEN_CHARS_PER_SEC
    _est_encode_secs = _est_audio_secs / _AAC_ENCODE_RT_FACTOR
    _est_tts_secs = stats.total_chars / stats.chars_per_sec
    # Fix: guard against ZeroDivisionError when there's nothing to process
    # (e.g. no chapters selected, or every chapter extracted to empty text).
    _est_total_secs = _est_tts_secs + _est_encode_secs
    stats.tts_progress_share = (_est_tts_secs / _est_total_secs) if _est_total_secs > 0 else 1.0
    stats.estimated_encode_secs = _est_encode_secs
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {stats.total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
    print(f'Estimated time remaining (assuming {stats.chars_per_sec} chars/sec): {eta}')

    tts_engine = tts_engine or settings.get('tts_engine', 'kokoro')
    if tts_engine not in ('kokoro', 'chatterbox'):
        tts_engine = 'kokoro'
    print(f'TTS engine: {tts_engine}')

    pipeline = None
    bridge = None
    # Chatterbox replaces the Kokoro speed control with these two style knobs.
    chatterbox_exaggeration = _clamp_unit_float(
        settings.get('chatterbox_exaggeration', CHATTERBOX_DEFAULT_EXAGGERATION),
        CHATTERBOX_DEFAULT_EXAGGERATION)
    chatterbox_cfg_weight = _clamp_unit_float(
        settings.get('chatterbox_cfg_weight', CHATTERBOX_DEFAULT_CFG_WEIGHT),
        CHATTERBOX_DEFAULT_CFG_WEIGHT)
    chatterbox_model = settings.get('chatterbox_model', CHATTERBOX_DEFAULT_MODEL)
    if not is_chatterbox_model(chatterbox_model):
        chatterbox_model = CHATTERBOX_DEFAULT_MODEL
    if tts_engine == 'chatterbox':
        source = settings.get('chatterbox_voice_source', 'preset')
        custom_ref = settings.get('chatterbox_ref_audio', '') or ''
        ref_audio = resolve_chatterbox_ref_audio(voice, source, custom_ref, settings)
        if not ref_audio and source == 'preset':
            print(f'Chatterbox voice preset for {voice} is not built yet — '
                  f'rendering it with Kokoro first...')
            set_espeak_library()
            built = generate_voice_sample(voice, settings=settings, stop_event=stop_event)
            if built:
                ref_audio = str(built)
        if not ref_audio:
            msg = ('Chatterbox needs a reference audio clip to clone for the selected '
                   'voice. Build the voice preset (or pick a custom WAV) and try again.')
            print('\033[91m' + msg + '\033[0m')
            if post_event:
                post_event('CORE_ERROR', message=msg)
            return
        print(f'Chatterbox clone source: {ref_audio}')
        print(f'Chatterbox model: {chatterbox_model}')
        print(f'Chatterbox style: exaggeration={chatterbox_exaggeration} '
              f'cfg_weight={chatterbox_cfg_weight}')
        try:
            bridge = ChatterboxBridge(
                device=settings.get('chatterbox_device', 'cuda'), ref_audio=ref_audio,
                exaggeration=chatterbox_exaggeration, cfg_weight=chatterbox_cfg_weight,
                model=chatterbox_model)
        except Exception as e:
            print(f'\033[91mFailed to start the Chatterbox bridge: {e}\033[0m')
            if post_event:
                post_event('CORE_ERROR', message=f'Failed to start Chatterbox bridge: {e}')
            raise
    else:
        set_espeak_library()
        try:
            pipeline = KPipeline(lang_code=lang_code_from_voice(voice))  # fix #5
        except Exception as e:
            print(f'\033[91mFailed to initialize the Kokoro TTS pipeline: {e}\033[0m')
            if post_event:
                post_event('CORE_ERROR', message=f'Failed to initialize TTS pipeline: {e}')
            raise

    chapter_wav_files = []
    try:
        for i, chapter in enumerate(selected_chapters, start=1):
            if stop_event and stop_event.is_set():
                print('Synthesis stopped by user.')
                break
            if max_chapters is not None and i > max_chapters:
                break
            text = chapter.extracted_text
            xhtml_file_name = chapter.get_name().replace(' ', '_').replace('/', '_').replace('\\', '_')
            # Fix: include `speed` in the cache-key filename. Previously only
            # `voice` was encoded, so re-running with a different speed would
            # silently reuse WAVs generated at the old speed.
            speed_tag = str(speed).replace('.', 'p')
            # Chatterbox WAVs are cached separately from Kokoro WAVs, so switching
            # engines never silently reuses the other engine's output. Chatterbox
            # has no speed control — exaggeration/CFG weight shape its output, so
            # those go in the tag and the Kokoro speed stays out of it. Turbo
            # ignores those knobs, so its tag names the model instead; this also
            # keeps Turbo and Multilingual audio from being reused interchangeably.
            if tts_engine == 'chatterbox':
                if chatterbox_model == CHATTERBOX_MODEL_TURBO:
                    engine_tag = '_chatterbox_turbo'
                else:
                    engine_tag = ('_chatterbox'
                                  f'_ex{str(chatterbox_exaggeration).replace(".", "p")}'
                                  f'_cfg{str(chatterbox_cfg_weight).replace(".", "p")}')
                speed_tag = ''
            else:
                engine_tag = ''
            chapter_wav_path = Path(output_folder) / filename.replace(
                extension, f'_chapter_{i}_{voice}{engine_tag}_{speed_tag}_{xhtml_file_name}.wav')
            chapter_wav_files.append(chapter_wav_path)

            if Path(chapter_wav_path).exists():
                print(f'File for chapter {i} already exists. Skipping')
                stats.processed_chars += len(text)
                if post_event:
                    post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
                continue
            if len(text.strip()) < 10:
                print(f'Skipping empty chapter {i}')
                chapter_wav_files.remove(chapter_wav_path)
                # Fix: still count these characters as processed so progress/ETA
                # tracking doesn't permanently under-count the total.
                stats.processed_chars += len(text)
                continue
            if ai_enabled:
                if post_event:
                    post_event('CORE_AI_REWRITE', chapter_index=chapter.chapter_index,
                               chapter_total=len(selected_chapters),
                               chunk_index=0, chunk_total=0)
                print(f'AI rewrite: chapter {i} ({len(text):,} chars)')
                text = correct_phonetics_ai(
                    text, ai_api_key, model=ai_model,
                    stop_event=stop_event, post_event=post_event,
                    chapter_index=chapter.chapter_index,
                    chapter_total=len(selected_chapters),
                    tts_engine=tts_engine,
                )
                if stop_event and stop_event.is_set():
                    print('Synthesis stopped by user during AI rewrite.')
                    break
            if i == 1:
                text = f'{title} – {creator}.\n\n' + text

            start_time = time.time()
            if post_event:
                post_event('CORE_CHAPTER_STARTED', chapter_index=chapter.chapter_index)

            if tts_engine == 'chatterbox':
                try:
                    audio_segments, write_sample_rate = gen_audio_segments_chatterbox(
                        bridge, text, stats=stats, post_event=post_event,
                        max_chunks=max_sentences, stop_event=stop_event)
                except ChatterboxError as e:
                    print(f'\033[91mChatterbox generation failed: {e}\033[0m')
                    if post_event:
                        post_event('CORE_ERROR', message=f'Chatterbox generation failed: {e}')
                    break
            else:
                audio_segments = gen_audio_segments(
                    pipeline, text, voice, speed, stats,
                    post_event=post_event, max_sentences=max_sentences, stop_event=stop_event)
                write_sample_rate = sample_rate

            if audio_segments:
                final_audio = np.concatenate(audio_segments)
                peak = np.abs(final_audio).max()
                if peak > 0:
                    final_audio = final_audio * (0.708 / peak)
                # Fix: write to a temp file and rename atomically, so a run that
                # is killed mid-write never leaves behind a partial WAV that a
                # later "already exists" resume check would mistake for done.
                tmp_wav_path = chapter_wav_path.with_suffix('.wav.tmp')
                soundfile.write(tmp_wav_path, final_audio, write_sample_rate,
                                format='WAV', subtype='PCM_16')
                tmp_wav_path.replace(chapter_wav_path)
                end_time = time.time()
                delta_seconds = end_time - start_time
                chars_per_sec = len(text) / delta_seconds
                print('Chapter written to', chapter_wav_path)
                if post_event:
                    post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
                print(f'Chapter {i} read in {delta_seconds:.2f} seconds ({chars_per_sec:.0f} characters per second)')
            else:
                print(f'Warning: No audio generated for chapter {i}')
                chapter_wav_files.remove(chapter_wav_path)
    finally:
        if bridge is not None:
            bridge.close()

    if has_ffmpeg and not (stop_event and stop_event.is_set()):
        # Fix: guard against an empty chapter_wav_files list (e.g. every
        # chapter got skipped/empty) instead of feeding ffmpeg an empty
        # concat list and failing with a confusing error.
        if not chapter_wav_files:
            print('\033[93mNo audio was generated for any chapter — skipping M4B creation.\033[0m')
            if post_event:
                post_event('CORE_ERROR', message='No audio was generated for any chapter.')
        else:
            total_audio_secs, chapters_txt_path = create_index_file(title, creator, chapter_wav_files, output_folder)
            create_m4b(chapter_wav_files, filename, cover_image, output_folder,
                       chapters_txt_path=chapters_txt_path,
                       total_audio_secs=total_audio_secs, stats=stats,
                       post_event=post_event, stop_event=stop_event)
            delete_wav_files(chapter_wav_files)
            if post_event:
                stats.progress = 100
                stats.eta = strfdelta(0)
                post_event('CORE_PROGRESS', stats=stats)
                post_event('CORE_FINISHED')

    settings = load_settings()
    save_settings(
        output_folder, voice, speed,
        gemini_api_key=settings.get('gemini_api_key', ''),
        gemini_model=settings.get('gemini_model', 'gemini-3.1-flash-lite'),
        gemini_enabled=settings.get('gemini_enabled', False),
        last_open_dir=settings.get('last_open_dir', ''),
        tts_engine=tts_engine,
        chatterbox_ref_audio=settings.get('chatterbox_ref_audio', ''),
        chatterbox_device=settings.get('chatterbox_device', 'cuda'),
        chatterbox_voice_source=settings.get('chatterbox_voice_source', 'preset'),
        voice_samples_dir=settings.get('voice_samples_dir', str(DEFAULT_VOICE_SAMPLES_DIR)),
        chatterbox_exaggeration=chatterbox_exaggeration,
        chatterbox_cfg_weight=chatterbox_cfg_weight,
        chatterbox_model=chatterbox_model,
    )
