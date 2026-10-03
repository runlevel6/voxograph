"""audiblez.core.audio - ffmpeg M4B encoding, chapter metadata, duration probing, process utils."""


import uuid
import soundfile
import subprocess
import re
import threading
from pathlib import Path

from .utils import strfdelta


def delete_wav_files(wav_files):
    """Deletes a list of WAV files."""
    print("Deleting temporary WAV files...")
    for wav_file in wav_files:
        try:
            Path(wav_file).unlink()
            print(f"Deleted: {wav_file}")
        except OSError as e:
            print(f"Error deleting {wav_file}: {e}")


def _popen_run(args, stop_event=None, on_stderr_line=None, **kwargs):
    """
    Drop-in replacement for subprocess.run() that can be interrupted.
    Polls every 0.5 s; if stop_event is set, terminates the child process
    and raises RuntimeError so callers can abort cleanly.

    Pipe buffer fix: if stdout/stderr are piped, drain them in background
    threads so the OS pipe buffer (typically 64 KB on Linux) never fills up
    and blocks the child process — which would cause an unrecoverable deadlock
    on long ffmpeg encodes.  Captured output is stored on proc._stdout_data
    and proc._stderr_data so callers can inspect it after the process exits.

    If on_stderr_line is given, it is called with each stderr line/fragment
    (split on \\r or \\n, matching how ffmpeg writes progress) as it arrives,
    so callers can do real-time progress parsing without duplicating the
    stream-draining logic themselves.
    """
    proc = subprocess.Popen(args, **kwargs)

    stdout_lines, stderr_lines = [], []

    def _drain_stdout(stream, buf):
        for line in stream:
            buf.append(line)

    def _drain_stderr(stream, buf, callback):
        chunk_buf = ''
        while True:
            chunk = stream.read(256)
            if not chunk:
                break
            chunk_buf += chunk
            parts = re.split(r'[\r\n]', chunk_buf)
            chunk_buf = parts[-1]        # keep the incomplete trailing fragment
            for part in parts[:-1]:
                buf.append(part)
                if callback:
                    callback(part)
        if chunk_buf:                    # flush any final fragment
            buf.append(chunk_buf)
            if callback:
                callback(chunk_buf)

    drain_threads = []
    if proc.stdout:
        t = threading.Thread(target=_drain_stdout, args=(proc.stdout, stdout_lines), daemon=True)
        t.start()
        drain_threads.append(t)
    if proc.stderr:
        t = threading.Thread(target=_drain_stderr, args=(proc.stderr, stderr_lines, on_stderr_line), daemon=True)
        t.start()
        drain_threads.append(t)

    while True:
        try:
            proc.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if stop_event and stop_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                for t in drain_threads:
                    t.join()
                raise RuntimeError('Stopped by user.')

    for t in drain_threads:
        t.join()

    proc._stdout_data = ''.join(stdout_lines)
    proc._stderr_data = ''.join(stderr_lines)

    return proc


def create_m4b(chapter_files, filename, cover_image, output_folder, chapters_txt_path,
               total_audio_secs=0, stats=None, post_event=None, stop_event=None):
    """Encode chapter WAVs directly to M4B in a single ffmpeg pass.

    Uses the concat demuxer as the audio source, eliminating the intermediate
    .tmp.mp4 file and halving total disk I/O compared to the two-step approach.
    64 k mono is transparent quality for speech and encodes ~2x faster than 128 k.

    When total_audio_secs > 0, parses ffmpeg's stderr in real time to derive
    accurate progress and ETA from the actual encode speed (speed=Nx lines).

    chapters_txt_path is produced by create_index_file() and passed in here
    (rather than each function independently hardcoding the same filename)
    so the two functions can't drift out of sync, and so the path can be
    made unique per-run (see create_index_file).
    """
    m4b_name = filename.replace('.epub', '.m4b')
    m4b_dir = Path(output_folder) / Path(m4b_name).stem
    m4b_dir.mkdir(parents=True, exist_ok=True)
    final_filename = m4b_dir / m4b_name
    safe_stem = Path(filename).stem.replace("'", "")
    list_file_path = Path(output_folder) / f"{safe_stem}_wav_list_{uuid.uuid4().hex[:8]}.txt"

    with open(list_file_path, 'w') as f:
        for chapter_file in chapter_files:
            escaped = str(Path(chapter_file).resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    print(f"WAV list ({len(chapter_files)} files): {list_file_path}")

    print('Creating M4B file...')
    ffmpeg_command = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0', '-i', str(list_file_path),
        '-i', str(chapters_txt_path),
    ]

    # Fix: give the cover temp file a unique name (like list_file_path
    # already had) so two concurrent conversions writing to the same
    # output_folder don't stomp on each other's temp files.
    cover_file_path = None
    if cover_image:
        cover_file_path = Path(output_folder) / f'cover_temp_{uuid.uuid4().hex[:8]}.jpg'
        cover_file_path.write_bytes(cover_image)
        ffmpeg_command.extend(['-i', str(cover_file_path)])
        ffmpeg_command.extend([
            '-map', '2:v',
            '-c:v', 'copy',
            '-disposition:v', 'attached_pic',
            '-metadata:s:v', 'title=Album cover',
            '-metadata:s:v', 'comment=Cover (front)',
        ])

    ffmpeg_command.extend([
        '-map', '0:a:0',
        '-map_metadata', '1',
        '-c:a', 'aac',
        '-ac', '1',
        '-b:a', '64k',
        '-f', 'mp4',
        str(final_filename),
    ])

    print("FFmpeg command:", " ".join(ffmpeg_command))

    tts_share = getattr(stats, 'tts_progress_share', 0.9) if stats else 0.9
    encode_share = 1.0 - tts_share

    def _process_ffmpeg_line(line):
        """Parse one ffmpeg progress line; post CORE_PROGRESS if we have enough info."""
        if not (stats and post_event and total_audio_secs > 0):
            return
        time_m = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)
        if not time_m:
            return
        encoded_secs = (int(time_m.group(1)) * 3600
                        + int(time_m.group(2)) * 60
                        + float(time_m.group(3)))
        fraction = min(encoded_secs / total_audio_secs, 1.0)
        stats.progress = int((tts_share + fraction * encode_share) * 100)
        speed_m = re.search(r'speed=\s*([\d.]+)x', line)
        if speed_m:
            speed = float(speed_m.group(1))
            remaining_secs = max(total_audio_secs - encoded_secs, 0) / speed if speed > 0 else 0
            stats.eta = strfdelta(remaining_secs)
            print(f'Encoding: {fraction*100:.1f}% | speed={speed}x | ETA {stats.eta}')
        post_event('CORE_PROGRESS', stats=stats)

    try:
        # Fix: reuse the shared _popen_run helper (stop_event handling +
        # non-blocking pipe draining) instead of duplicating that logic
        # inline, so there's a single implementation to maintain.
        proc = _popen_run(
            ffmpeg_command,
            stop_event=stop_event,
            on_stderr_line=_process_ffmpeg_line,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode == 0:
            print(f'{final_filename} created. Enjoy your audiobook.')
        else:
            print(f"Error creating M4B file. FFmpeg returned code: {proc.returncode}")
            print("FFmpeg stdout:\n", proc._stdout_data)
            print("FFmpeg stderr:\n", proc._stderr_data)
    except RuntimeError as e:
        print(str(e))
    finally:
        list_file_path.unlink(missing_ok=True)
        if cover_file_path and cover_file_path.exists():
            cover_file_path.unlink()
        if Path(chapters_txt_path).exists():
            Path(chapters_txt_path).unlink()


def probe_duration(file_name):
    """Return the duration (seconds) of a local WAV file.

    Uses soundfile (already a dependency, since we write these WAVs
    ourselves) instead of shelling out to ffprobe per chapter. This avoids
    an extra subprocess per chapter and a previously-unhandled exception
    path: if a WAV can't be read, we log a warning and fall back to 0s
    instead of crashing after all the TTS work for the book is done.
    """
    try:
        info = soundfile.info(str(file_name))
        return info.frames / float(info.samplerate)
    except Exception as e:
        print(f'\033[93mWarning: could not read duration of {file_name} ({e}); assuming 0s.\033[0m')
        return 0.0


def create_index_file(title, creator, chapter_mp3_files, output_folder):
    """Write ffmpeg chapter metadata and return (total_audio_secs, chapters_txt_path).

    The chapters.txt path is given a unique suffix (like the WAV concat
    list already was) so two concurrent conversions in the same
    output_folder can't clobber each other's metadata file, and the path
    is returned so create_m4b() doesn't have to independently guess it.
    """
    chapters_txt_path = Path(output_folder) / f"chapters_{uuid.uuid4().hex[:8]}.txt"
    total_secs = 0.0
    with open(chapters_txt_path, "w", encoding="utf-8") as f:
        f.write(f";FFMETADATA1\ntitle={title}\nartist={creator}\n\n")
        start = 0
        for i, c in enumerate(chapter_mp3_files):
            duration = probe_duration(c)
            total_secs += duration
            end = start + int(duration * 1000)
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle=Chapter {i}\n\n")
            start = end
    return total_secs, chapters_txt_path
