"""audiblez.core.nlp - spaCy model caching, espeak-ng setup, and Kokoro lang-code helper."""


import os
import traceback
from glob import glob
import spacy
import subprocess
import platform
from pathlib import Path


# ---------------------------------------------------------------------------
# spaCy — load once, reuse everywhere  (fix #4)
# ---------------------------------------------------------------------------

_nlp = None


def get_nlp():
    """Return a cached spaCy nlp instance, loading it on first call."""
    global _nlp
    if _nlp is None:
        load_spacy()          # downloads model if absent
        _nlp = spacy.load('xx_ent_wiki_sm')
        _nlp.add_pipe('sentencizer')
    return _nlp


def load_spacy():
    if not spacy.util.is_package("xx_ent_wiki_sm"):
        print("Downloading Spacy model xx_ent_wiki_sm...")
        spacy.cli.download("xx_ent_wiki_sm")


def set_espeak_library():
    """Find and register the espeak-ng library path."""
    try:
        if os.environ.get('ESPEAK_LIBRARY'):
            library = os.environ['ESPEAK_LIBRARY']
        elif platform.system() == 'Darwin':
            from subprocess import check_output
            try:
                cellar = Path(check_output(["brew", "--cellar"], text=True).strip())
                pattern = cellar / "espeak-ng" / "*" / "lib" / "*.dylib"
                if not (library := next(iter(glob(str(pattern))), None)):
                    raise RuntimeError("No espeak-ng library found; please set the path manually")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError("Cannot locate Homebrew Cellar. Is 'brew' installed and in PATH?") from e
        elif platform.system() == 'Linux':
            library = glob('/usr/lib/*/libespeak-ng*')[0]
        elif platform.system() == 'Windows':
            library = 'C:\\Program Files*\\eSpeak NG\\libespeak-ng.dll'
        else:
            print('Unsupported OS, please set the espeak library path manually')
            return
        print('Using espeak library:', library)
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(library)
    except Exception:
        traceback.print_exc()
        print("Error finding espeak-ng library:")
        print("Probably you haven't installed espeak-ng.")
        print("On Mac: brew install espeak-ng")
        print("On Linux: sudo apt install espeak-ng")


# ---------------------------------------------------------------------------
# Lang-code helper  (fix #5)
# ---------------------------------------------------------------------------

# Kokoro lang codes are a single character prefix of the voice name,
# e.g. "af_heart" → "a", "bf_emma" → "b".  The old settings default of
# "en_US" would have produced "e" which is invalid.

def lang_code_from_voice(voice: str) -> str:
    """
    Extract the single-character Kokoro language code from a voice name.
    e.g. "af_heart" → "a",  "bf_emma" → "b".
    Falls back to 'a' (American English) if the voice string is unexpected.
    """
    if voice and len(voice) >= 1 and voice[1:2] in ('f', 'm', '_'):
        return voice[0]
    # Unrecognised format — default to American English
    print(f"Warning: could not determine lang code from voice '{voice}', defaulting to 'a'.")
    return 'a'
