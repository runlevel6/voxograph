# AGENTS.md — Voxograph Memory Bank

> **Project**: voxograph 1.0.0 — EPUB → `.m4b` audiobooks (Kokoro-82M + optional Chatterbox)
> **Fork by**: Vlad Reshetov (based on original by Claudio Santini, santinic/audiblez)
> **Repo**: `/home/vlad/voxograph` (editable install; `origin` = runlevel6/voxograph)
> **Venv**: `/home/vlad/audiblez_venv` — *name intentionally not renamed* (92 `bin/*` shebangs)
> **Entry points**: `voxograph` (CLI), `voxograph-ui` (GUI)

---

## 0. Agent Operating Rules (MUST FOLLOW)

1. **Read AGENTS.md first** at the start of every session to load project context.
2. **Keep this file current, not append-only.** After a task, update §2–§9 where the change
   affects layout, dependencies, module map, configuration or conventions, then add a terse bullet
   entry to §10 (newest first). Consolidate/rewrite older entries when they go stale — a short
   accurate file beats a long historical one. Full archaeology lives in git history
   (`git log` in `/home/vlad/voxograph`).
3. **Never record secrets.** `voxograph/config.json` holds a real Gemini API key and is gitignored.
   Redact it to `"..."` if you ever quote the file.
4. **Verify before writing.** Line counts, entry points and dep versions drift; check the source
   rather than trusting this file.
5. **Do not commit AGENTS.md changes** unless the user explicitly asks — it is a working memory
   bank, not a deliverable.

---

## 1. Overview

Converts EPUB e-books into `.m4b` (plus intermediate per-chapter `.wav`) using Kokoro-82M, with
**Chatterbox** (Multilingual V3 / Turbo) as an interchangeable second engine. wxPython GUI + CLI.
9 languages, spaCy sentence segmentation, ffmpeg/ffprobe for encoding/duration, espeak-ng for
phonemization, optional Gemini AI for phonetic correction.

**Performance**: ~600 chars/s on a T4 (Kokoro); ~60 chars/s CPU-only.

---

## 2. Layout & Entry Points

### Package (`/home/vlad/voxograph/voxograph/`, ~4.4k lines)

| Path | Lines | Role |
|------|-------|------|
| `__init__.py` | 4 | Docstring only (no `sys.path` hack). |
| `__main__.py` | 4 | Enables `python -m voxograph`. |
| `cli.py` | 43 | argparse → `core.main()`. |
| `voices.py` | 27 | Voice table keyed by lang code + flag emoji. |
| `voice_samples_cli.py` | 110 | Headless Chatterbox preset builder (`--list/--voice/--lang/--all`). |
| `config.json` | — | Runtime settings, **gitignored** (holds the API key). |
| `core/` | 3002 | Engine/pipeline package (see §4.2). |
| `ui/` | 1584 | wxPython GUI package (see §4.3). |

`core/` = `__init__.py` 220 (re-exports 74 public names, so `import voxograph.core as core` and
`from voxograph.core import X` are the stable surface) · `constants.py` 48 · `utils.py` 48 ·
`settings.py` 102 · `text.py` 264 · `nlp.py` 86 · `gemini.py` 770 · `epub.py` 117 · `audio.py` 254 ·
`voice_samples.py` 200 · `pipeline.py` 449 (holds `main`) · `tts/{kokoro 56, chunking 98,
chatterbox 289}`.

`ui/` = `__init__.py` 25 (re-exports `main`, `MainWindow`, `EVENTS`, `CoreThread`,
`_extract_bridge_json`, `border`, `BOOK_DETAILS_WRAP_WIDTH`) · `events.py` 20 · `bridge.py` 24 ·
`window.py` 59 · `core_thread.py` 20 · `app.py` 15 · `__main__.py` 5 · mixins `menu` 53,
`layout` 166, `params` 373, `ref_audio` 170, `voice_samples` 135, `settings` 61, `ebook` 119,
`preview` 197, `synthesis` 77, `core_events` 65.

`MainWindow` is assembled in `ui/window.py` from the mixins; `super().__init__` still reaches
`wx.Frame`. One cohesive method group per module.

### Console scripts (venv `bin/`, from `[project.scripts]`)

```
voxograph    = voxograph.cli:cli_main
voxograph-ui = voxograph.ui:main
```
Installed dist-info: `site-packages/voxograph-1.0.0.dist-info/`. Entry points are only on `PATH`
after `source /home/vlad/audiblez_venv/bin/activate`.

### Data outside the package

- `~/.voxograph/voice_samples/` — one `<voice>.wav` per Kokoro voice + `index.json`.
  Rendered by Kokoro, consumed by Chatterbox as `audio_prompt_path`. Outside site-packages on
  purpose: a reinstall must not wipe samples. Overridable via `voice_samples_dir`.
- `chatterbox_bridge/generate.py` (repo) — the published copy of the bridge script. See §9.

### Chatterbox bridge environment

`/home/vlad/chatterbox_venv` — separate Python 3.12 venv (`torch 2.6.0+cu124`,
`chatterbox-tts==0.1.7`, patched `chatterbox/mtl_tts.py` accepting `t3_model=`) because its torch
pin conflicts with the main venv. Weights (~4 GB, HF cache) download on first use.

**Bridge script** `/home/vlad/chatterbox_venv/generate.py` (the live one the code spawns):
- JSON request on stdin → JSON result on stdout. Model cached in-process (`MODEL`/`MODEL_KIND`).
- One-shot mode (GUI preview) and `--serve` (newline-delimited loop used by
  `core.ChatterboxBridge` for whole books — model loads once per run).
- Request keys: `text`, `output_path`, `device`, `ref_audio`, `language_id`, `t3_model`,
  `model` (`multilingual` | `turbo`), `exaggeration`, `cfg_weight`, plus Turbo's
  `turbo_temperature`, `turbo_top_p`, `turbo_top_k`, `turbo_repetition_penalty`.
- Turbo loads `ChatterboxTurboTTS`, takes no `language_id`, and the bridge forces
  `exaggeration`/`cfg_weight` to `0.0` to avoid a per-chunk warning; it applies the four
  `turbo_*` keys and echoes them back in the result. `chatterbox_bridge/generate.py` in the repo
  is now an exact copy of the live script — keep them that way.
- The bridge pins `torch.set_num_threads(4)` once at import (before any model load), so
  both Multilingual V3 and Turbo run with 4 intra-op threads instead of torch's all-cores
  default.

---

## 3. Dependencies

From `pyproject.toml` (`requires-python = ">=3.10,<3.13"`):

```
beautifulsoup4, ebooklib, lxml, numpy, phonemizer, pick, soundfile, spacy (+xx_ent_wiki_sm),
tabulate, torch, kokoro==0.9.4, misaki[zh]>=0.9.2
```
- extras: `gui` = pillow + wxpython; `dev` = build + twine.
- **Not declared but imported**: `google-genai` (used by `core/gemini.py`). See §9.
- The Chatterbox bridge venv installs its own torch/chatterbox-tts.

**System (not pip)**: `ffmpeg` (M4B/AAC), `ffprobe` (duration), `ffplay` (GUI preview),
`espeak-ng` (phonemizer; path resolved by `set_espeak_library()`).

---

## 4. Module Map

### 4.1 `cli.py`

`epub_file_path` (positional) · `-v/--voice` (default `af_sky`) · `-p/--pick` · `-s/--speed`
(0.5–2.0) · `-c/--cuda` (sets torch default device) · `-o/--output` (default `.`, which resolves
to `config.json`'s `output_folder`). Then `from voxograph.core import main; main(...)`.
No `sys.path` manipulation — imports are absolute.

### 4.2 `core/`

**`constants.py`** — `sample_rate = 24000` · `_SPOKEN_CHARS_PER_SEC = 12.5` (ETA) ·
`_AAC_ENCODE_RT_FACTOR = 50` · `CONFIG_FILE = Path(__file__).resolve().parent.parent /
'config.json'` (two `.parent`s because the module moved one level deeper) ·
`DEFAULT_VOICE_SAMPLES_DIR = ~/.voxograph/voice_samples` · `CHATTERBOX_DEFAULT_{EXAGGERATION,
CFG_WEIGHT} = 0.5` · `CHATTERBOX_TURBO_DEFAULT_{TEMPERATURE = 0.8, TOP_P = 0.95, TOP_K = 1000,
REPETITION_PENALTY = 1.2}` with their `CHATTERBOX_TURBO_*_RANGE` bounds (drives both the spinners
and the clamping) · `CHATTERBOX_MODEL_{MULTILINGUAL,TURBO}` / `CHATTERBOX_MODELS` /
`CHATTERBOX_DEFAULT_MODEL = 'multilingual'` · `DEFAULT_VOICE = 'af_heart'`.

**`utils.py`** — `_clamp_unit_float(value, default)` (0–1 clamp) · `_clamp_float(value, low, high,
default)` / `_clamp_int(value, low, high, default)` (range clamp, used by the Turbo knobs) ·
`_apply_fade(audio, fade_ms=5)` (shared by both engines, prevents clicks) ·
`strfdelta(tdelta, fmt)`.

**`settings.py`** — `is_chatterbox_model(value)` · `load_settings()` (`.setdefault` migration for
every key, so old configs load, then `_clamp_turbo_settings()` normalizes the Turbo knobs) ·
`save_settings(...)` (one big explicit signature; the single source of truth, imported by the UI).

**`text.py`** — `ABBREVIATIONS` + pre-compiled `_ABBREV_PATTERNS`; Roman numeral expansion
(`_roman_to_int` / `_int_to_words` / `_roman_match_to_words` / `expand_roman_numerals` with
`_ROMAN_HEADING_RE`, `_ROMAN_KEYWORD_RE`, `_ROMAN_RE_SRC`) — only valid numerals 1–3999, ALL-CAPS
preserved as Title Case; `clean_text()` = soft hyphens → abbreviations → Roman numerals →
Unicode punctuation → whitespace → repeated-punctuation collapse.

**`nlp.py`** — `_nlp` module cache (Fix #4) · `get_nlp()` / `load_spacy()` (`xx_ent_wiki_sm` +
`sentencizer`) · `set_espeak_library()` (Linux/macOS/Windows lookup → `ESPEAK_LIBRARY`) ·
`lang_code_from_voice(voice)` (voice format `{lang}_{type}_{name}`, Fix #5).

**`epub.py`** — `find_cover()` (ITEM_COVER → OPF meta → id="cover" → filename) ·
`find_document_chapters_and_extract_texts(book, ai_enabled=False)` (spine order, BeautifulSoup +
lxml over `title/p/h1-h4/li`, Fix #2, applies `clean_text`) · `is_chapter()` · `find_good_chapters()`
· `pick_chapters()` · `print_selected_chapters()` · `chapter_beginning_one_liner()`.

**`gemini.py`** (770 lines) — `_call_gemini_with_retry()` (waits a minute on 503/high-demand, posts
`CORE_AI_RETRY_EXHAUSTED` when it gives up) · every `generate_content` call passes
`config=_ai_generate_config()`, i.e. `GenerateContentConfig(max_output_tokens=_AI_MAX_OUTPUT_TOKENS
=65536)`, to override the SDK/model default · `_ai_rewrite_single_chunk()` /
`_ai_split_paragraphs()` / `_ai_split_overlong()` / `_ai_split_hard()` / `_extract_rewritten_section()` · `correct_phonetics_ai()` (rewrites one
chapter) · `_rewrite_packed_unit()` (one packed request) · `iter_correct_phonetics_ai_chapters()`
(generator: cross-chapter packing via `_ai_pack_units()` / `_ai_take_prefix()` — chapters
<`_AI_MIN_AGGREGATE_TOKENS`=1K merge with the leading part of the next into requests
≤`_AI_REWRITE_MAX_TOKENS`=8K, joined by `_AI_PART_MARKER` and split back, with a per-chapter
fallback if the marker isn't preserved; yields each chapter as soon as all its pieces are rewritten
so the caller can run TTS between AI requests) · `correct_phonetics_ai_chapters()` (eager
`dict()` wrapper around the generator) ·
`check_phonetic_transcription_ai()` (check-only) · `_phonetic_rules_for(tts_engine,
chatterbox_model)` selects `_PHONETIC_RULES` (Kokoro: inline IPA `[text](/kˈOkəɹO/)`, stress
`(-1)`/`(+1)`, `[[espeak phonemes]]`) vs `_CHATTERBOX_PHONETIC_RULES` (plain-English respelling
only) — for **Chatterbox Turbo only** it appends `_CHATTERBOX_TAG_RULES`, the 19 expressive tags
(9 vocal effects + 10 emotion/delivery styles) the Turbo tokenizer knows, so one AI pass does
pronunciation + tag insertion; Multilingual V3 gets no tag rules · `_is_retryable_gemini_error()`,
`_NON_RETRYABLE_ERROR_MARKERS`, `_sanitize_api_key()`, `_sleep_with_stop_event()`.

**`audio.py`** — `_popen_run(args, stop_event, on_stderr_line)` — interruptible `subprocess.run`
with drain threads (avoids the 64 KB pipe-buffer deadlock) · `create_m4b()` (concat demuxer,
64k mono AAC, `chapters.txt` metadata, real-time `speed=Nx` progress, temp cleanup in `finally`) ·
`create_index_file()` (FFMETADATA1 chapters, TIMEBASE=1/1000) · `probe_duration()` (ffprobe) ·
`delete_wav_files()`.

**`voice_samples.py`** — `VOICE_SAMPLE_TEXT` + `VOICE_SAMPLE_TEXT_VERSION` (bump to force rebuilds;
`index.json` stores it per voice) · `VOICE_SAMPLE_TRIM_THRESHOLD = 0.005` (~−46 dBFS) ·
`voice_samples_dir()` / `voice_sample_path()` / `voice_sample_info()` / `voice_sample_exists()` /
`missing_voice_samples()` · `_trim_edge_silence()` · `generate_voice_sample()` (renders via Kokoro,
trims, peak-normalizes to 0.708, atomic `.wav.tmp` write with explicit
`format='WAV', subtype='PCM_16'` — required, soundfile cannot infer from `.wav.tmp`) ·
`generate_voice_samples()` (batch, one `KPipeline` per lang code) ·
`resolve_chatterbox_ref_audio(voice, source, custom_path, settings)` — **the single decision
point** for what Chatterbox clones; never compare/store `.wav` paths anywhere else.

**`tts/chunking.py`** — `CHATTERBOX_MAX_CHUNK_CHARS = 300` (Chatterbox degrades/hallucinates on
longer input) · `split_chatterbox_text()` splits at paragraph → sentence → clause → word →
hard, and **packs** whole sentences into a running buffer until the next would exceed the limit,
so chunks approach 280–300 chars and always end on real punctuation. Never a blind `text[:300]`.
`preview_excerpt()` returns the first chunk (used by the GUI preview for both engines).

**`tts/chatterbox.py`** — `CHATTERBOX_BRIDGE_DIR` (`VOXOGRAPH_CHATTERBOX_BRIDGE_DIR`, default
`/home/vlad/chatterbox_venv`) / `_PYTHON` / `_SCRIPT`, `CHATTERBOX_T3_MODEL`,
`CHATTERBOX_LANGUAGE_ID = 'en'` · `ChatterboxError` (base: dead/absent bridge, `BrokenPipeError`,
error payloads) / `ChatterboxCancelled(ChatterboxError)` (`stop_event`) · `ChatterboxBridge`
(spawns `generate.py --serve`, newline-delimited JSON, queue-based responses, stderr drain thread,
polls `stop_event` every 250 ms, `close()` terminates the child; carries `device`, `ref_audio`,
`exaggeration`, `cfg_weight`, `model`, `turbo_temperature`, `turbo_top_p`, `turbo_top_k`,
`turbo_repetition_penalty`) · `gen_audio_segments_chatterbox()` (per-chunk temp WAV →
mono float32 → `_apply_fade`, updates stats/ETA from measured throughput, posts `CORE_PROGRESS`,
restores separator chars on the last chunk so progress reaches 100 %) ·
`_is_chatterbox_progress_noise()` / `_CHATTERBOX_NOISE_MARKERS`.

**`tts/kokoro.py`** — `gen_audio_segments()` (spaCy sentences → `KPipeline`, 5 ms fades, stats,
`stop_event`, `max_sentences` with `>=` — Fix #3) · `gen_text()` (one-shot TTS to WAV).

**`pipeline.py`** — `main(file_path, voice=None, pick_manually=False, speed=1, output_folder='.',
max_chapters=None, max_sentences=None, selected_chapters=None, post_event=None, stop_event=None,
tts_engine=None)`: loads spaCy + settings → resolves engine/model/style from settings → reads EPUB
+ metadata/cover → selects chapters → estimates ETA → for Chatterbox resolves the clone source
(auto-builds the preset via Kokoro if missing) and opens one warm bridge → if AI is enabled, builds
the list of chapters that still need synthesis (existing WAVs / <10-char chapters skipped first)
and drives the lazy `iter_correct_phonetics_ai_chapters()` generator → per chapter: skips
existing WAVs, skips <10 chars, applies the AI rewrite pulled for this chapter, prepends
title/author to chapter 1, synthesizes (engine-specific), and only after the audio is written pulls
the next AI request — so AI and TTS interleave AI→TTS→AI→TTS instead of all AI up front →
concatenates, peak-normalizes to 0.708, writes WAV → builds the index + M4B → deletes WAVs →
re-saves settings. Posts `CORE_STARTED` … `CORE_FINISHED`. The Chatterbox per-chapter synthesis is
wrapped in a retry loop (`chatterbox_max_retries = 3`): a `ChatterboxError` (dead/absent bridge,
e.g. `BrokenPipeError` when the subprocess dies mid-chapter) closes the bridge, re-spawns it via
`_recreate_chatterbox_bridge()` and re-synthesizes that chapter from scratch; `ChatterboxCancelled`
is re-raised first for a clean `stop_event` stop, and once the retries are exhausted `CORE_ERROR`
is posted and the run ends gracefully.

### 4.3 `ui/`

**Events** (`events.py`): `CORE_STARTED`, `CORE_PROGRESS`, `CORE_CHAPTER_STARTED`,
`CORE_CHAPTER_FINISHED`, `CORE_AI_REWRITE`, `CORE_AI_RETRY_EXHAUSTED`, `CORE_ERROR`,
`CORE_FINISHED`. Plus `border = 5`, `BOOK_DETAILS_WRAP_WIDTH = 260`.

**Mixins** — `menu` (Open Ctrl+O / Exit Ctrl+Q) · `layout` (top bar, splitter, right column; the
right column is a `ScrolledPanel` so tall parameter stacks stay reachable, and boxes are added with
proportion `0` so they hug their content — only the params box takes the slack) · `params`
(TTS engine Kokoro/Chatterbox, Chatterbox model dropdown, compute device, voice dropdown,
engine-specific style rows, voice source + ref audio, output folder) · `ref_audio` (select/clear,
stale-path detection, hint text) · `voice_samples` (build one / build all) · `settings`
(`save_current_settings()` round-trip) · `ebook` (open EPUB, metadata, chapter table) · `preview`
(engine dispatch + `ffplay`) · `synthesis` (start/cancel, progress, ETA, relayout) · `core_events`
(event handlers, error dialogs, retry-exhausted warning).

**Chatterbox UX** — the voice dropdown is **shared**: with Chatterbox it names the preset that gets
cloned, so no `.wav` path is managed by hand. Chatterbox has no speed control, so the Kokoro
**Speed** row is swapped for the selected model's own knobs in its place: **Exaggeration** /
**CFG Weight** (`wx.SpinCtrlDouble`, 0.0–1.0, `inc=0.05`) for Multilingual V3, and **Temperature** /
**Top P** (`0.0–2.0` / `0.0–1.0`), **Top K** (`wx.SpinCtrl`, 0–100000) / **Repetition Penalty**
(1.0–2.0) for Turbo, which ignores the V3 pair entirely. Each model keeps its own values
(`chatterbox_*` vs `chatterbox_turbo_*`), so switching models never loses a tuning choice.
`get_ref_audio()` delegates to
`core.resolve_chatterbox_ref_audio()`; `_sync_ref_audio_text/_hint()` show `✓ Preset: <voice>`,
`✓ Cloning: <file>` or a `⚠` what-is-missing message. A Chatterbox preview with no clone source
offers to build the preset (then resumes the preview) or pick a WAV instead of dead-ending.

**`CoreThread`** — background thread calling `core.main()`, posting back via `wx.PostEvent`.

---

## 5. Configuration — `voxograph/config.json` (single source of truth, gitignored)

Loaded by `load_settings()`, written by `save_settings()` and the UI's
`save_current_settings()` on every change. Resolved relative to `core/constants.py` (Fix #10), so
CWD does not matter.

| Key | Values | Meaning |
|-----|--------|---------|
| `output_folder` | path | where M4B/WAV land |
| `voice` | voice code | shared by both engines — for Chatterbox it names the voice preset |
| `speed` | 0.5–2.0 | Kokoro playback speed (Kokoro only; Chatterbox has none) |
| `gemini_api_key` | `"AIza…"` | **secret — never commit or paste** |
| `gemini_model` | str | default `gemini-3.1-flash-lite` |
| `gemini_enabled` | bool | AI phonetic check on/off |
| `last_open_dir` | path | file-dialog start dir |
| `tts_engine` | `kokoro` \| `chatterbox` | which engine synthesizes *and* previews |
| `chatterbox_ref_audio` | path | custom clone source, used only when source = `custom` |
| `chatterbox_device` | `cuda` \| `cpu` | device for the Chatterbox bridge |
| `chatterbox_voice_source` | `preset` \| `custom` | clone the Kokoro voice sample, or the user's WAV |
| `chatterbox_model` | `multilingual` \| `turbo` | which Chatterbox family member (invalid → `multilingual`) |
| `chatterbox_exaggeration` | 0.0–1.0 | expressiveness/drama; higher also speeds pacing (V3 only — Turbo ignores it) |
| `chatterbox_cfg_weight` | 0.0–1.0 | adherence to the reference's style/pacing; 0.3 slows a fast reference, 0 stops accent bleeding (V3 only) |
| `chatterbox_turbo_temperature` | 0.0–2.0 | Turbo sampling temperature; higher = more varied, sloppier (default 0.8) |
| `chatterbox_turbo_top_p` | 0.0–1.0 | Turbo nucleus cutoff; lower = more predictable (default 0.95) |
| `chatterbox_turbo_top_k` | 0–100000 | Turbo top-k cutoff; 0 disables the filter (default 1000) |
| `chatterbox_turbo_repetition_penalty` | 1.0–2.0 | Turbo loop breaker; 1.0 disables it (default 1.2) |
| `voice_samples_dir` | path | where the Kokoro-rendered presets live |

---

## 6. Running

```bash
source /home/vlad/audiblez_venv/bin/activate      # puts voxograph/voxograph-ui on PATH

voxograph book.epub -v af_sky -s 1.0 -c -o /output/dir   # CLI (-c = CUDA, -p = pick chapters)
voxograph-ui                                          # GUI  (or python -m voxograph.ui)
python -m voxograph book.epub                         # same CLI via __main__

python -m voxograph.voice_samples_cli --list            # which presets exist
python -m voxograph.voice_samples_cli --voice af_heart  # build one
python -m voxograph.voice_samples_cli --all --force     # rebuild all (~75 s)
```
`~/convert_to_audio.sh` activates the venv and launches the GUI.

There is **no test suite** — verification is `py_compile`, import smoke checks, and headless
`DISPLAY=:1` GUI construction. Any refactor must be behaviour-preserving and incremental.

---

## 7. Output Structure

- Output dir (default `/home/vlad/AudioBooks`).
- Chapter WAVs: `{filename}_chapter_{i}_{voice}{engine_tag}_{speed_tag}_{xhtml_file_name}.wav`.
  `engine_tag` is `_chatterbox_turbo_t<v>_p<v>_k<v>_rp<v>` or `_chatterbox_ex<v>_cfg<v>` (and
  `speed_tag` is dropped) for Chatterbox, empty for Kokoro — engines, models and their style or
  sampling settings never reuse each other's audio.
- M4B: `{output}/{m4b_stem}/{filename_without_epub}.m4b`.
- `chapters.txt` — ffmpeg FFMETADATA chapter markers, temporary, deleted in `finally`.

---

## 8. Conventions & Hard-Won Gotchas

1. **Absolute imports only**: `from voxograph.core import main`, `import voxograph.core as core`.
   The old `sys.path` hack is gone — reintroducing it can load `core/` twice as a top-level module.
2. **Package splits are behaviour-preserving.** `core.py` (2118 lines) → `core/` and `ui.py`
   (1388) → `ui/` were done by verbatim line-range extraction; `__init__.py` re-exports keep every
   prior import path working. Keep modules single-purpose (~100–400 lines).
3. **Event-driven**: core functions take `post_event` + a `threading.Event` `stop_event`; mutable
   progress lives in a `SimpleNamespace`.
4. **Bridges must keep stdout pure JSON.** `perth` prints
   `loaded PerthNet (Implicit) at step 250,000` during model load, so `generate.py` captures
   `_REAL_STDOUT` and sets `sys.stdout = sys.stderr` before importing `chatterbox`. The GUI parses
   with `ui/bridge.py::_extract_bridge_json` (scans lines in reverse for the last JSON object)
   rather than `json.loads(proc.stdout)`.
5. **Never point `chatterbox_ref_audio` at `/tmp`** — it will not survive a reboot.
6. **Cloning cache keys must include every knob that changes the audio** (engine, model, style,
   speed) or a settings change silently reuses stale output.
7. **Gemini AFC must go through a Chat session** (`client.chats.create` + `Chat.send_message` /
   `send_message_stream`). Direct AFC in `Models.generate_content*` is not supported and the SDK
   warns about it. Plain no-tools calls may still use `client.models.generate_content`.
8. **AI rewrite is engine-aware**: Kokoro can take inline IPA / stress overrides / `[[espeak]]`;
   Chatterbox must get plain-English respelling only (`_phonetic_rules_for(tts_engine,
   chatterbox_model)`). When **Turbo** is selected the same ruleset also asks for the 19
   expressive tags (`_CHATTERBOX_TAG_RULES`), which the Turbo tokenizer supports natively;
   Multilingual V3 gets the respelling rules without the tags.
9. **`soundfile.write` on temp files needs explicit `format=`/`subtype=`** — `.wav.tmp` has no
   inferable format.
10. **Cleanup**: `finally` blocks for temp files and bridge shutdown; broad `except Exception` is
    acceptable in settings/UI paths.
11. **Paths**: always `pathlib.Path`.
12. **Chatterbox crashes are retried, not fatal — and the retry loop is import-sensitive.**
    `pipeline.main` retries a failed chapter by recreating the bridge (`chatterbox_max_retries = 3`).
    Because `ChatterboxCancelled` subclasses `ChatterboxError`, the `except ChatterboxCancelled:
    raise` clause **must be listed before** `except ChatterboxError`. Both names therefore have to be
    imported into `pipeline.py`: a `raise`-less `except ChatterboxCancelled:` whose name is missing
    raises `NameError` on the first bridge error and masks the real failure. `gen_audio_segments_chatterbox`
    itself already swallows `ChatterboxCancelled` and returns partial segments; `pipeline` catches
    `ChatterboxError` to recreate and retry (see §10, 2026-10-07).

---

## 9. Known Discrepancies / Open Items

- **`google-genai` is missing from `pyproject.toml`** even though `core/gemini.py` imports
  `from google import genai`. A clean install would fail at AI-rewrite time.
- **`pip check` reports venv/pyproject drift**: `phonemizer` not installed (a `phonemizer-fork` is
  installed instead), `kokoro` 0.7.16 vs the `==0.9.4` pin, `misaki` 0.7.17 vs `>=0.9.2`. The app
  works, but the pins no longer describe the machine.
- **The `v1.0.0` release tag predates the current work**: the UI package port, the Chatterbox
  Turbo/tag/bridge changes and the bridge-retry logic all landed in later commits
  (`9037f6c` … `8ac9bd4`), so the tag is not the working tree. The tree is otherwise clean; the
  only local change is the `ChatterboxCancelled` import hotfix (§10, 2026-10-07).
- **PyPI publish is not wired up**: the `v1.0.0` publish run failed with `invalid-publisher`; a
  Trusted Publisher must be registered for project `voxograph` / repo `voxograph` / workflow
  `publish.yml` / environment `pypi`.
- **Chatterbox is effectively English-locked** here: `CHATTERBOX_LANGUAGE_ID = 'en'`, and Turbo is
  English-only regardless.

---

## 10. Recent Changes

- **2026-10-09 — Chatterbox thread pinning.** `generate.py` (both the live
  `/home/vlad/chatterbox_venv` copy and the repo `chatterbox_bridge/` copy) now calls
  `torch.set_num_threads(4)` once at import, before any model load, so the bridge runs with 4
  intra-op CPU threads instead of torch's all-cores default. The bridge is the only place the
  Chatterbox model loads (preview one-shot and `--serve` both spawn it), so this covers every
  Chatterbox path. Verified: copies identical, `py_compile` clean.
- **2026-10-07 — Chatterbox crash-recovery hotfix.** Commit `8ac9bd4` added the Chatterbox bridge
  retry loop to `pipeline.main` but omitted `ChatterboxCancelled` from its `from .tts.chatterbox
  import (...)` line. The loop's first clause is `except ChatterboxCancelled:`; with the name
  undefined, **any** bridge failure (e.g. the `generate.py --serve` subprocess dying mid-chapter →
  `BrokenPipeError` on `stdin.write`, wrapped as `ChatterboxError`) hit that clause and raised
  `NameError`, which escaped `main` — so the intended 3-retry recreate-and-resume path never ran.
  Added the missing import only (the `ChatterboxCancelled`-before-`ChatterboxError` order is already
  correct, since the former subclasses the latter). Verified offline with a stubbed bridge/`soundfile`:
  (a) a first-call `ChatterboxError` is caught, `_recreate_chatterbox_bridge()` runs, the retry
  succeeds and the chapter finishes (`CORE_CHAPTER_FINISHED`, no `CORE_ERROR`); (b) a permanently
  broken bridge gets 4 attempts (1 + 3 retries) and 4 bridge constructions, then posts `CORE_ERROR`
  with no exception escaping. `py_compile` + import smoke clean. Documented in §4.2 and §8.12.
- **2026-10-04 — Smaller AI rewrite requests.** `core/gemini.py` `_AI_REWRITE_MAX_TOKENS` is now
  `8_000` (≈32K chars per request, from `_AI_REWRITE_CHARS_PER_TOKEN = 4`) and
  `_AI_MIN_AGGREGATE_TOKENS` is now `1_000` (small chapters merge with the next sooner, so more
  but smaller requests — the interleaved AI→TTS loop spreads them out). Stale comments/docstrings
  that still said 40K/15K/160K were corrected. `_AI_REWRITE_MAX_CHARS`/`_AI_MIN_AGGREGATE_CHARS`
  are derived, and `pipeline.main` prints the live values, so those follow automatically.
- **2026-10-04 — AI/TTS interleaving + explicit max output.** AI rewriting and TTS now alternate
  (AI→TTS→AI→TTS) instead of rewriting the whole book first. `core.gemini` gained
  `iter_correct_phonetics_ai_chapters()`, a generator that processes one packed request at a time
  and yields each chapter as soon as all its pieces are back; `correct_phonetics_ai_chapters()`
  is now just `dict()` over it. `pipeline.main` pulls one rewrite, synthesizes that chapter, and
  only then pulls the next rewrite — so the next Gemini call is sent after the previous TTS
  finishes, spreading API calls out in time (verified with stubbed AI/TTS: events fire
  AI→TTS→AI→TTS). Also, every `generate_content` call now passes
  `config=types.GenerateContentConfig(max_output_tokens=65536)` via `_ai_generate_config()` to
  explicitly override the SDK/model default. `iter_correct_phonetics_ai_chapters` is exported.
- **2026-10-04 — Cross-chapter AI request packing.** AI rewriting is now planned across chapters
  instead of one request per chapter. `pipeline.main` builds the list of chapters that still need
  synthesis (existing WAVs / <10-char chapters are skipped before any API call, exactly as the
  synth loop does) and hands it to the new `core.gemini.correct_phonetics_ai_chapters()`, which
  returns per-chapter rewritten text. `_ai_pack_units()` packs whole chapters greedily and, when a
  request would close below `_AI_MIN_AGGREGATE_TOKENS`, pulls the leading part of
  the next chapter (`_ai_take_prefix()`, cutting at paragraph → sentence → clause → word) so small
  chapters never become tiny requests; every unit stays ≤ `_AI_REWRITE_MAX_TOKENS`. Units
  spanning chapters are joined with `_AI_PART_MARKER` and the model is asked to preserve it;
  `correct_phonetics_ai_chapters()` splits the rewrite back per chapter and `_ai_rewrite_combined_unit()`
  falls back to one request per chapter if the marker is lost. The cache-key `engine_tag`/`speed_tag`
  computation moved above the synth loop (and into a `_chapter_wav_path()` helper) so the pre-pass
  and the loop agree on which WAVs exist. Verified offline: packing respects the request char cap,
  reconstruction preserves every chapter's text, and a stopped event returns no rewrites.

- **2026-10-04 — AI chunk fallback is now sentence-aware.** `core/gemini.py` no longer hard-slices
  an over-long block at an arbitrary character offset. New `_ai_split_overlong()` breaks at the
  coarsest real boundary — end of a nearby paragraph/newline, then a sentence `.?!`, then a clause
  `,;:`, then word, then a hard split (`_ai_split_hard()` for punctuation-free runs and monster
  tokens). `_ai_split_paragraphs()` packs those pieces. Verified on real EPUBs: *Panzer Leader*
  (no `\n\n`, 173K chars) used to cut mid-sentence and now splits only after `.`; content is
  preserved and every chunk respects the request char cap.

- **2026-10-04 — Memory-bank refresh.** Reconciled §2/§4 module line counts and the `core/__init__`
  export count with the working tree (core grew to 2630 lines: `gemini.py` 473, `tts/chatterbox.py`
  289, `pipeline.py` 382, `__init__.py` 213 exporting 72 names; `ui/preview.py` 197,
  `ui/synthesis.py` 77), and restated §9 to show the UI port staged and the Turbo/tag/bridge edits
  still unstaged. No code change.
- **2026-10-04 — Chatterbox expressive tags in AI rewrite.** When AI is enabled with the
  Chatterbox engine, `_phonetic_rules_for('chatterbox', 'turbo')` appends `_CHATTERBOX_TAG_RULES`
  to `_CHATTERBOX_PHONETIC_RULES`, so one Gemini pass asks for pronunciation rewrites **and**
  insertion of the 19 built-in tags (9 vocal effects, 10 emotion/delivery styles). Gated on
  `chatterbox_model == 'turbo'` — Multilingual V3 has no such tokens and would speak them
  literally. The Chatterbox "no square brackets" rule was narrowed to allow only these tags;
  Kokoro is unchanged. `chatterbox_model` is threaded through `correct_phonetics_ai` /
  `check_phonetic_transcription_ai` / `_ai_rewrite_single_chunk` from `pipeline.main` and the GUI
  preview. Applied to both the rewrite and the read-only check (which shares the ruleset).
- **2026-10-04 — Shared Chatterbox bridge payload.** `core.tts.chatterbox.chatterbox_bridge_request()`
  is now the single builder for the bridge JSON payload, used by `ChatterboxBridge.generate()` and
  the GUI one-shot preview (which previously hand-built the dict and hardcoded `language_id` /
  `t3_model`). Exported as `core.chatterbox_bridge_request`; stops preview and whole-book synthesis
  from drifting apart. UI progress/rewrite ticks reverted to `synth_panel.Layout()` (the full
  `_relayout_right_panel()` remains only for structural changes), avoiding a scrollbar re-measure
  on every progress event.
- **2026-10-04 — Turbo settings.** Turbo gets the sampling knobs it actually honors —
  Temperature / Top P / Top K / Repetition Penalty — in the same panel slot where Multilingual V3
  shows Exaggeration / CFG Weight, saved as `chatterbox_turbo_*` so each model keeps its own values.
  `_clamp_float`/`_clamp_int` + `CHATTERBOX_TURBO_*_RANGE` bounds, `_clamp_turbo_settings()` in
  `load_settings`/`save_settings`, the four `turbo_*` keys plumbed through `ChatterboxBridge`,
  `main()` and the preview payload, and Turbo's cache tag now encodes them. Also reconciled the
  stale `chatterbox_bridge/generate.py` with the live bridge (§9 item closed). Verified end to end:
  Turbo generated a 24 kHz WAV with the new knobs echoed back, and V3 still ignores them.
- **2026-10-04 — Environment switched to voxograph.** Checkout `/home/vlad/audiblez` →
  `/home/vlad/voxograph`; editable reinstall (`pip install -e /home/vlad/voxograph --no-deps
  --force-reinstall`); the stale 1250-line repo `ui.py` replaced by the 17-module `ui/` package
  ported from the installed copy (incl. the `KPipeline` preview-import fix); added `__main__.py`;
  `config.json` and `~/.voxograph/voice_samples/` migrated; `pip uninstall audiblez` plus removal
  of the leftover `site-packages/audiblez/`, its dist-info and the old `bin/audiblez*` scripts.
  Verified `voxograph --help`, `python -m voxograph --help`, the `voxograph.ui` imports and
  headless `MainWindow` construction.
- **2026-10-04 — Rebrand `audiblez` → `voxograph`.** Package `git mv`, case-preserved replacements
  everywhere (incl. `AUDIBLEZ_CHATTERBOX_BRIDGE_DIR` → `VOXOGRAPH_CHATTERBOX_BRIDGE_DIR`), version
  `0.5.0` → `1.0.0`, GitHub repo renamed to `runlevel6/voxograph`, `v1.0.0` tag + release,
  `publish.yml` added, `.gitignore` now ignores `voxograph/config.json`. MIT license retained with
  the original author's line plus `Copyright (c) 2026 Vlad Reshetov`.
- **2026-10-03 — Monoliths split** into `core/` (14 modules) and `ui/` (17 modules), both
  verified by `py_compile`, AST parity against the originals, and headless GUI construction.
- **2026-10-03 — Chatterbox Turbo** added as a second model, with `chatterbox_model` in settings,
  a UI dropdown, and its `_chatterbox_turbo` cache tag.
- **2026-10-03 — Chatterbox voice presets**: each Kokoro voice rendered once by Kokoro and handed
  to Chatterbox as `audio_prompt_path`, so one dropdown serves both engines; `voice_samples_cli.py`
  added for headless building.
- **2026-10-03 — Exaggeration + CFG Weight** replaced the useless Speed box for Chatterbox; cache
  tags now encode those values.
- **2026-10-03 — Full-book Chatterbox synthesis** via a persistent `ChatterboxBridge` and
  sentence-aware 300-char chunking; `CORE_ERROR` event added; AI rewrite made engine-aware.
- **2026-10-03 — Preview fix**: `preview_excerpt()` replaced a blind `text[:300]` slice that cut
  mid-sentence.
- **2026-10-03 — Right panel made scrollable** and the Book Details / Status boxes made to hug
  their content, so the Start button is always reachable.
- **2026-10-03 — `perth` stdout pollution fixed** (see §8.4), which had broken Chatterbox previews
  with `JSONDecodeError`.