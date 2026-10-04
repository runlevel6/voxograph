# Voxograph by runlevel6 — Enhanced Fork

> This repository is an enhanced, feature-rich fork of [santinic/audiblez](https://github.com/santinic/audiblez).

Voxograph generates `.m4b` audiobooks from regular `.epub` e-books using two interchangeable
engines: **Kokoro-82M** and **[Chatterbox](https://github.com/resemble-ai/chatterbox)** (voice
cloning, in Multilingual V3 and Turbo variants).

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) is a recently published text-to-speech model with just 82M params and very natural sounding output.
It's released under Apache licence and it was trained on < 100 hours of audio.
It currently supports these languages: 🇺🇸 🇬🇧 🇪🇸 🇫🇷 🇮🇳 🇮🇹 🇯🇵 🇧🇷 🇨🇳

On a Google Colab's T4 GPU via Cuda, **it takes about 5 minutes to convert "Animal's Farm" by Orwell** (which is about 160,000 characters) to audiobook, at a rate of about 600 characters per second.

On my M2 MacBook Pro, on CPU, it takes about 1 hour, at a rate of about 60 characters per second.


## Acknowledgments & Credits

This project originated as a fork of [Audiblez](https://github.com/santinic/audiblez) created by [Claudio Santini](https://github.com/santinic). 
While it has been extensively refactored with enhanced text sanitization, custom pipeline features, and audio handling improvements, we are grateful to the original author and contributors for laying the foundation.


## Fork Enhancements

This fork adds significant architectural improvements over the original project:

- **Second TTS engine: Chatterbox (voice cloning)** — the GUI can synthesize the whole book with [Chatterbox](https://github.com/resemble-ai/chatterbox) instead of Kokoro, choosing between the **Multilingual V3** and **Turbo** models. Because Chatterbox degrades on long inputs, chapters are split into ~300-character chunks at sentence (then clause, then word) boundaries and stitched back together, and the model is loaded **once per run** through a persistent bridge process, so the multi-second load is paid a single time instead of once per chapter. See [Second TTS engine: Chatterbox](#second-tts-engine-chatterbox).

- **Chatterbox Turbo with native paralinguistic tags** — the **Model** dropdown also offers **Turbo** (350M, faster, English-only). It ignores Exaggeration/CFG Weight and is shaped instead by its own sampling knobs — **Temperature**, **Top P**, **Top K**, **Repetition Penalty** — and its tokenizer understands 19 expressive tags (`[laugh]`, `[sigh]`, `[whispering]`, `[dramatic]`, …). When AI rewriting is enabled, the same Gemini pass that fixes pronunciation also inserts those tags where a real speaker would react. Each model keeps its own settings in `config.json`, so switching back and forth never loses a tuning choice.

- **Named Chatterbox voices via a Kokoro-rendered sample library** — Chatterbox has no named voice list, so voxograph renders every Kokoro voice once into `~/.voxograph/voice_samples/<voice>.wav` (edge silence trimmed, peak normalized) and hands it to Chatterbox as the cloning prompt. The preset name is therefore the Kokoro voice name, and one dropdown serves both engines. **Build Sample** renders the selected voice in the background, **Build All** renders every voice in the dropdown, or switch the source to **Custom WAV** to clone a recording of your own. The same library is available headlessly via `python -m voxograph.voice_samples_cli --list | --voice af_heart | --all`.

- **AI-Assisted Pronunciation Correction (Gemini)** — the flagship feature of this release. When **AI Phonetic Check** is enabled, each chapter is rewritten by Google Gemini *before* synthesis so Kokoro pronounces tricky words correctly. The model expands abbreviations (`Dr.` → `Doctor`, `NASA` → `N A S A`), spells out numbers and dates (`2024` → `twenty twenty four`, `3rd` → `third`), re-spells homophones and silent letters, and **always** rewrites foreign proper nouns — personal names, place names, military units — into an English-friendly spelling or an inline IPA override with stress marks (e.g. `Péronne` → `Peyron`). Plain English respelling is preferred when it is simpler and just as accurate; punctuation is preserved because it shapes prosody. The rewrite rules are engine-aware: Kokoro may receive inline IPA overrides, stress marks and `[[espeak]]` phonemes, while Chatterbox gets plain-English respelling only (it cannot read IPA). When **Turbo** is selected, the same pass also inserts the model's 19 expressive tags.

- **Full-pipeline, chunked AI rewriting** — the AI step runs on the *entire book* during synthesis, not just on a preview snippet. Chapters that already have a WAV file on disk are skipped without any API call, and very long chapters are rewritten in chunks of roughly 300K tokens each, with live per-chunk progress shown in the GUI.

- **AI integration never blocks a conversion** — Gemini calls auto-retry on transient `503 UNAVAILABLE` errors; if retries are exhausted the GUI offers **OK** (continue without AI correction) or **Cancel** (stop the run). Any other failure or malformed AI response silently falls back to the original text, so audiobook generation always proceeds.

- **"Check with AI" analysis + corrected preview** — a GUI button analyzes the selected chapter and lists each flagged word with the reason and its IPA transcription, followed by the exact rewritten text that will be sent to the TTS engine (the analysis and the rewrite share the same phonetic rules, so they can never disagree). The **Preview** button runs the same silent correction on its snippet, so the sample you hear matches the final book.

- **Settings Persistence via `config.json`** — voice, speed, and output folder are saved and loaded automatically via `load_settings()` / `save_settings()` (fix #9), together with the Gemini AI settings (`gemini_enabled`, `gemini_api_key`, `gemini_model`, default model `gemini-3.1-flash-lite`) and the Chatterbox options (`tts_engine`, `chatterbox_model`, device, voice source, reference audio, Exaggeration/CFG Weight and the Turbo sampling knobs). No more configuring options every launch.

- **Cached spaCy NLP Pipeline** — spaCy is loaded once and reused across synthesis runs, eliminating the expensive model load and dramatically speeding up repeated runs (fix #4).

- **Language Code Extraction Helper** — `lang_code_from_voice()` extracts the single-character Kokoro language code directly from the selected voice name, fixing invalid defaults and ensuring proper language matching (fix #5).

- **Text Cleaning Pipeline** — abbreviations are expanded (e.g. "Dr." → "Doctor"), Roman numerals are converted to words (e.g. "Chapter III" → "Chapter Three"), Unicode punctuation is normalized, and whitespace is collapsed. The TTS engine receives clean, unambiguous text.

- **Audio Fades (5 ms)** — each synthesized segment is faded in and out to prevent clicks and pops caused by DC-offset discontinuities at sentence boundaries.

- **Torch Tensor → NumPy Conversion** — audio from Kokoro is converted from torch tensors to NumPy arrays, ensuring standard audio processing operations work reliably.

- **Single-Pass ffmpeg Concat Demuxer** — chapter WAVs are encoded to M4B in one ffmpeg invocation using the concat demuxer, removing intermediate `.tmp.mp4` files and halving total disk I/O.

- **Real-Time Progress Parsing** — ffmpeg's stderr is parsed in real-time to derive accurate progress percentage and ETA from actual encode speed (`speed=Nx` lines).

- **Threaded `stop_event` Support** — long-running operations (synthesis, encoding, spacy loading) check a `threading.Event` so the UI Cancel button and exit flows can abort cleanly without orphaned processes.

- **UI Enhancements** — settings are saved automatically; a **Cancel button** stops synthesis immediately; preview threads run as daemon threads and are pruned automatically; the chapters table is disabled during synthesis.


## How to install the Command Line tool

If you have Python 3 on your computer, you can install it with pip.
You also need `espeak-ng` and `ffmpeg` installed on your machine:

```bash
sudo apt install ffmpeg espeak-ng                   # on Ubuntu/Debian 🐧
pip install voxograph
```

```bash
brew install ffmpeg espeak-ng                       # on Mac 🍏
pip install voxograph
```

Then you can convert an .epub directly with:

```
voxograph book.epub -v af_sky
```

It will first create a bunch of `book_chapter_1.wav`, `book_chapter_2.wav`, etc. files in the same directory,
and at the end it will produce a `book.m4b` file with the whole book you can listen with VLC or any
audiobook player.
It will only produce the `.m4b` file if you have `ffmpeg` installed on your machine.


## How to run the GUI

The GUI is a simple graphical interface to use voxograph.
You need some extra dependencies to run the GUI:

```
sudo apt install ffmpeg espeak-ng 
sudo apt install libgtk-3-dev        # just for Ubuntu/Debian 🐧, Windows/Mac don't need this
  
pip install voxograph pillow wxpython
```

Then you can run the GUI with:
```
voxograph-ui
```


## Second TTS engine: Chatterbox

Besides Kokoro, the GUI can synthesize the book with **Chatterbox**, which clones a voice from a
short reference recording. Select it in the **TTS Engine** radio of the *Audiobook Parameters*
panel; the choice is stored in `config.json` (`tts_engine`) and is also what the command-line tool
uses.

Chatterbox ships as a family, chosen from the **Model** dropdown (`chatterbox_model`):

- **Multilingual V3** (default, 500M) — the highest-quality model. It is English-locked in this
  configuration (23 languages are available upstream) and its delivery is shaped by
  **Exaggeration** (expressiveness; higher also speeds pacing) and **CFG Weight** (adherence to the
  reference clip's style/pacing).
- **Turbo** (350M) — smaller, faster and lighter, but English-only. It ignores Exaggeration/CFG
  Weight and is shaped instead by its sampling knobs: **Temperature**, **Top P**, **Top K** and
  **Repetition Penalty**. Turbo also understands 19 paralinguistic tags (`[laugh]`, `[sigh]`,
  `[whispering]`, `[dramatic]`, …), which voxograph inserts automatically when AI rewriting is on.

Each model keeps its own settings in `config.json` (`chatterbox_*` vs `chatterbox_turbo_*`), so
switching models never loses a tuning choice.

Chatterbox is installed in a **separate virtualenv**, because it pins `torch==2.6.0` while voxograph
needs a different build, and voxograph talks to it over a small bridge script:

```bash
python3 -m venv ~/chatterbox_venv
~/chatterbox_venv/bin/pip install torch==2.6.0 torchaudio==2.6.0 chatterbox-tts==0.1.7
```

The bridge script lives at `<venv>/generate.py` and is shipped in this repo as
[`chatterbox_bridge/generate.py`](chatterbox_bridge/generate.py) — copy it there:

```bash
cp chatterbox_bridge/generate.py ~/chatterbox_venv/generate.py
```

It reads one JSON request on stdin and writes one JSON result on stdout; in `--serve` mode it loops
over newline-delimited requests so a whole book is synthesized with a single model load. Its location
is baked into `voxograph/core/tts/chatterbox.py` (`CHATTERBOX_BRIDGE_DIR`) and can be overridden with
the `VOXOGRAPH_CHATTERBOX_BRIDGE_DIR` environment variable.

Two patches are needed on top of stock `chatterbox-tts==0.1.7`: Multilingual V3 needs a
`t3_model="t3_mtl23ls_v3.safetensors"` argument on `ChatterboxMultilingualTTS.from_pretrained()`
(thread it through `from_local()` too), and Turbo needs the `chatterbox.tts_turbo` module
(`ChatterboxTurboTTS`), which may require a newer build.

Then pick the voice to clone:

- **Voice Preset** — clones `~/.voxograph/voice_samples/<voice>.wav`, a Kokoro rendering of the voice
  selected in the dropdown. Missing presets are built automatically (or with **Build Sample** /
  **Build All**, or `python -m voxograph.voice_samples_cli --all`).
- **Custom WAV** — clones a recording you supply.

Both engines then share the rest of the pipeline: chapter WAV caching, normalization, single-pass
ffmpeg M4B encoding and AI phonetic correction. The AI rewrite rules are engine-aware — Kokoro gets
inline IPA overrides, stress marks and espeak phonemes; Chatterbox gets plain-English respelling only
(since it does not read IPA), plus the expressive tags for Turbo. Chapter WAVs are tagged per engine
and model, so switching engines, models or style/sampling settings never reuses stale audio.

Chatterbox is English-locked in this configuration (Multilingual V3 and Turbo alike), and it is
noticeably slower than Kokoro — expect a fraction of Kokoro's characters-per-second.


## How to run on Windows

After many trials, on Windows we recommend to install voxograph in a Python venv:

1. Open a Windows terminal
2. Create anew folder: `mkdir voxograph`
3. Enter the folder: `cd voxograph`
4. Create a venv: `python -m venv venv`
5. Activate the venv: `.\venv\Scripts\Activate.ps1`
6. Install the dependencies: `pip install voxograph pillow wxpython`
7. Now you can run `voxograph` or `voxograph-ui`
8. For Cuda support, you need to install Pytorch accordingly: https://pytorch.org/get-started/locally/


## Speed

By default the audio is generated using a normal speed, but you can make it up to twice slower or faster by specifying a speed argument between 0.5 to 2.0:

```
voxograph book.epub -v af_sky -s 1.5
```


## Supported Voices

Use `-v` option to specify the voice to use. Available voices are listed here.
The first letter is the language code and the second is the gender of the speaker e.g. `im_nicola` is an italian male voice.

[For hearing samples of Kokoro-82M voices, go here](https://claudio.uk/posts/audiblez-v4.html)

| Language                  | Voices                                                                                                                                                                                                                                     |
|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 🇺🇸 American English     | `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` |
| 🇬🇧 British English      | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis`                                                                                                                                          |
| 🇪🇸 Spanish              | `ef_dora`, `em_alex`, `em_santa`                                                                                                                                                                                                           |
| 🇫🇷 French               | `ff_siwis`                                                                                                                                                                                                                                 |
| 🇮🇳 Hindi                | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`                                                                                                                                                                                                |
| 🇮🇹 Italian              | `if_sara`, `im_nicola`                                                                                                                                                                                                                     |
| 🇯🇵 Japanese             | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo`                                                                                                                                                                         |
| 🇧🇷 Brazilian Portuguese | `pf_dora`, `pm_alex`, `pm_santa`                                                                                                                                                                                                           |
| 🇨🇳 Mandarin Chinese     | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang`                                                                                                                                 |

For more detaila about voice quality, check this document: [Kokoro-82M voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)


## How to run on GPU

By default, voxograph runs on CPU. If you pass the option `--cuda` it will try to use the Cuda device via Torch.

Check out this example: [Audiblez running on a Google Colab Notebook with Cuda ](https://colab.research.google.com/drive/164PQLowogprWQpRjKk33e-8IORAvqXKI?usp=sharing]).

We don't currently support Apple Silicon, as there is not yet a Kokoro implementation in MLX. As soon as it will be available, we will support it.


## Manually pick chapters to convert

Sometimes you want to manually select which chapters/sections in the e-book to read out loud.
To do so, you can use `--pick` to interactively choose the chapters to convert (without running the GUI).


## Help page

For all the options available, you can check the help page `voxograph --help`:

```
usage: voxograph [-h] [-v VOICE] [-p] [-s SPEED] [-c] [-o FOLDER] epub_file_path

positional arguments:
  epub_file_path        Path to the epub file

options:
  -h, --help            show this help message and exit
  -v VOICE, --voice VOICE
                        Choose narrating voice: a, b, e, f, h, i, j, p, z
  -p, --pick            Interactively select which chapters to read in the audiobook
  -s SPEED, --speed SPEED
                        Set speed from 0.5 to 2.0
  -c, --cuda            Use GPU via Cuda in Torch if available
  -o FOLDER, --output FOLDER
                        Output folder for the audiobook and temporary files

example:
  voxograph book.epub -l en-us -v af_sky

to use the GUI, run:
  voxograph-ui
```


## Credits

This project was originally created by [Claudio Santini](https://claudio.uk) in 2025.

Related Article: [Audiblez v4: Generate Audiobooks from E-books](https://claudio.uk/posts/audiblez-v4.html)

This fork is maintained by **runlevel6**. All MIT licence terms and attribution to the original author are preserved.


## License

MIT — see LICENSE file for full text.
