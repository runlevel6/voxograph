"""voxograph.core.constants - Global constants shared across the package.

This module is the single source of truth for every named constant in the
project. Modules import their constants from here instead of defining their own
module-level constants, so each value lives in exactly one place.

The only constant that cannot move here is ``voxograph.ui.events.EVENTS``
(wxPython event objects), because the core package must stay importable without
a GUI toolchain; UI *layout* values are plain ints and do live here.
"""


import os
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Audio / timing
# ---------------------------------------------------------------------------

sample_rate = 24000

_SPOKEN_CHARS_PER_SEC = 12.5  # ~150 wpm * 5 chars / 60 s
_AAC_ENCODE_RT_FACTOR = 50    # ffmpeg aac runs ~50x realtime

# Silence (ms) inserted between synthesized chunks so the join does not sound
# like one breathless run-on sentence. 400 ms is a natural short-pause beat.
CHUNK_SILENCE_MS = 400

# Fix #10: use path relative to this file so it resolves consistently
# regardless of where the process is launched from.
CONFIG_FILE = Path(__file__).resolve().parent.parent / 'config.json'

DEFAULT_VOICE = 'af_heart'


# ---------------------------------------------------------------------------
# Voice samples (Kokoro-rendered presets that Chatterbox clones from)
# ---------------------------------------------------------------------------

# Default home for the rendered voice samples (one WAV per Kokoro voice) that
# Chatterbox clones from. Kept outside the package directory on purpose: a
# pip reinstall of voxograph wipes site-packages but must not wipe samples.
DEFAULT_VOICE_SAMPLES_DIR = Path.home() / '.voxograph' / 'voice_samples'

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


# ---------------------------------------------------------------------------
# Kokoro voice table (keyed by single-character language code) + flag emoji
# ---------------------------------------------------------------------------

flags = {'a': '🇺🇸', 'b': '🇬🇧', 'e': '🇪🇸', 'f': '🇫🇷', 'h': '🇮🇳', 'i': '🇮🇹', 'j': '🇯🇵', 'p': '🇧🇷', 'z': '🇨🇳'}

flags_win = {'a': 'american', 'b': 'british', 'e': 'spanish', 'f': 'french', 'h': 'hindi', 'i': 'italian',
             'j': 'japanese', 'p': 'portuguese', 'z': 'chinese'}

voices = {
    'a': ['af_alloy', 'af_aoede', 'af_bella', 'af_heart', 'af_jessica', 'af_kore', 'af_nicole', 'af_nova',
          'af_river', 'af_sarah', 'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir', 'am_liam',
          'am_michael', 'am_onyx', 'am_puck', 'am_santa'],
    'b': ['bf_alice', 'bf_emma', 'bf_isabella', 'bf_lily', 'bm_daniel', 'bm_fable', 'bm_george', 'bm_lewis'],
    'e': ['ef_dora', 'em_alex', 'em_santa'],
    'f': ['ff_siwis'],
    'h': ['hf_alpha', 'hf_beta', 'hm_omega', 'hm_psi'],
    'i': ['if_sara', 'im_nicola'],
    'j': ['jf_alpha', 'jf_gongitsune', 'jf_nezumi', 'jf_tebukuro', 'jm_kumo'],
    'p': ['pf_dora', 'pm_alex', 'pm_santa'],
    'z': ['zf_xiaobei', 'zf_xiaoni', 'zf_xiaoxiao', 'zf_xiaoyi', 'zm_yunjian', 'zm_yunxi', 'zm_yunxia',
          'zm_yunyang']
}


# ---------------------------------------------------------------------------
# Chatterbox engine / bridge
# ---------------------------------------------------------------------------

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

# Chatterbox degrades/hallucinates on input longer than this, so whole-book
# synthesis packs sentences into chunks that approach (but never exceed) it.
CHATTERBOX_MAX_CHUNK_CHARS = 300

# How many times a failed fragment is retried (recreating the bridge each time)
# before the run is ended with CORE_ERROR.
CHATTERBOX_MAX_RETRIES = 3

CHATTERBOX_BRIDGE_DIR = Path(
    os.environ.get('VOXOGRAPH_CHATTERBOX_BRIDGE_DIR', '/home/vlad/chatterbox_venv'))
CHATTERBOX_BRIDGE_PYTHON = CHATTERBOX_BRIDGE_DIR / 'bin' / 'python3'
CHATTERBOX_BRIDGE_SCRIPT = CHATTERBOX_BRIDGE_DIR / 'generate.py'
CHATTERBOX_T3_MODEL = 't3_mtl23ls_v3.safetensors'
CHATTERBOX_LANGUAGE_ID = 'en'
_CHATTERBOX_NOISE_MARKERS = ('it/s', '?it/s', 'Sampling:', 'Fetching', '%|')

# Sentence/clause boundaries used by the Chatterbox chunker.
_SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?\u2026])\s+')
_CLAUSE_BOUNDARY_RE = re.compile(r'(?<=[,;:])\s+')


# ---------------------------------------------------------------------------
# Text cleaning — abbreviations and Roman numerals
# ---------------------------------------------------------------------------

# Fix #1: Street pattern MUST come before the generic Saint pattern so that
# "St. James" → "Street James" does not accidentally fire first.
ABBREVIATIONS = [
    # Titles
    (r'\bMrs\.', 'Missus'),
    (r'\bMr\.', 'Mister'),
    (r'\bMs\.', 'Miz'),
    (r'\bMiss\.', 'Miss'),
    (r'\bDr\.', 'Doctor'),
    (r'\bProf\.', 'Professor'),
    (r'\bRev\.', 'Reverend'),
    (r'\bHon\.', 'Honorable'),
    # Street before Saint so the lookahead fires first
    (r'\bSt\.(?=\s+[A-Z])', 'Street'),
    (r'\bSt\.', 'Saint'),
    # Military / professional ranks
    (r'\bGen\.', 'General'),
    (r'\bCol\.', 'Colonel'),
    (r'\bMaj\.', 'Major'),
    (r'\bCapt\.', 'Captain'),
    (r'\bLt\.', 'Lieutenant'),
    (r'\bSgt\.', 'Sergeant'),
    (r'\bCpl\.', 'Corporal'),
    (r'\bPvt\.', 'Private'),
    (r'\bAdm\.', 'Admiral'),
    # Name suffixes
    (r'\bJr\.', 'Junior'),
    (r'\bSr\.', 'Senior'),
    (r'\bEsq\.', 'Esquire'),
    # Common Latin / general abbreviations
    (r'\betc\.', 'et cetera'),
    (r'\bvs\.', 'versus'),
    (r'\bVs\.', 'Versus'),
    (r'\bapprox\.', 'approximately'),
    (r'\bcf\.', 'compare'),
    (r'\be\.g\.', 'for example'),
    (r'\bi\.e\.', 'that is'),
    (r'\bviz\.', 'namely'),
    (r'\bib\.', 'in the same place'),
    (r'\bibid\.', 'in the same place'),
    (r'\bop\. cit\.', 'in the work cited'),
    (r'\bno\.?\s*(?=\d)', 'number '),
    (r'\bNo\.?\s*(?=\d)', 'Number '),
    (r'\bvol\.', 'volume'),
    (r'\bVol\.', 'Volume'),
    (r'\bch\.', 'chapter'),
    (r'\bCh\.', 'Chapter'),
    (r'\bfig\.', 'figure'),
    (r'\bFig\.', 'Figure'),
    (r'\bp\.', 'page'),
    (r'\bpp\.', 'pages'),
    # Addresses
    (r'\bAve\.', 'Avenue'),
    (r'\bBlvd\.', 'Boulevard'),
    (r'\bRd\.', 'Road'),
    (r'\bDept\.', 'Department'),
    (r'\bGovt\.', 'Government'),
]

# Compile once for efficiency
_ABBREV_PATTERNS = [(re.compile(pat), repl) for pat, repl in ABBREVIATIONS]

# Matches a valid Roman numeral token (1–3999).  The alternation structure
# ensures only well-formed sequences match — e.g. "IIII" won't match because
# there is no four-I combination in standard notation.
_ROMAN_RE_SRC = (
    r'M{0,4}'           # thousands: 0–4000
    r'(?:CM|CD|D?C{0,3})'   # hundreds: 900, 400, 0–300, 500–800
    r'(?:XC|XL|L?X{0,3})'   # tens: 90, 40, 0–30, 50–80
    r'(?:IX|IV|V?I{0,3})'   # ones: 9, 4, 0–3, 5–8
)

# Case 1 — standalone heading: a line that is *only* a Roman numeral,
# optionally followed by a period, colon, or dash (and nothing else).
# Examples:  "III."   "XIV"   "IV:"   "  ii.  "
_ROMAN_HEADING_RE = re.compile(
    r'^(' + _ROMAN_RE_SRC + r')([.:\-]?)$',
    re.IGNORECASE | re.MULTILINE,
)

# Case 2 — inline after a structural keyword.
# Examples:  "Chapter III"   "Part IV:"   "Book II,"   "Act V, Scene i"
_ROMAN_KEYWORD_RE = re.compile(
    r'\b(Chapter|Part|Section|Book|Volume|Vol|Act|Scene|Article|Appendix|Canto)'
    r'(\s+)(' + _ROMAN_RE_SRC + r')\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Gemini AI rewriting
# ---------------------------------------------------------------------------

# Explicitly request the largest output budget Gemini accepts, overriding the
# SDK/model default, so long phonetic rewrites are never silently truncated.
# Applied as GenerateContentConfig(max_output_tokens=...) on every API call.
_AI_MAX_OUTPUT_TOKENS = 65536

# Backoff schedule requested: retry after 1 minute, then 2, then 3, then 4
# (each pause is the previous pause + 1 minute). That's 4 retries (5 total
# attempts) before giving up.
_GEMINI_RETRY_DELAYS_SECS = [60, 120, 180, 240]

# Errors in this set are permanent (bad model name, bad key, no permission)
# and will never succeed no matter how many times we retry, so we bail out
# immediately instead of making the caller wait up to 10 minutes for nothing.
_NON_RETRYABLE_ERROR_MARKERS = ('NOT_FOUND', 'PERMISSION_DENIED', 'INVALID_ARGUMENT', 'UNAUTHENTICATED')

# Hard ceiling on a single Gemini request. A watchdog thread prints a live
# countdown while the call is in flight; if no response arrives in this window
# the model is abandoned and the next one is tried. The SDK client is also
# built with the same HTTP timeout (HttpOptions.timeout is in milliseconds)
# so the hung request is actually aborted on the client side rather than
# lingering until the OS-level TCP timeout.
_GEMINI_REQUEST_TIMEOUT_SECS = 180
_GEMINI_REQUEST_TIMEOUT_MS = _GEMINI_REQUEST_TIMEOUT_SECS * 1000

# Ordered list of Gemini models to try sequentially until one succeeds.
# Newer models first; the first available/working model will be used.
GEMINI_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
)

# ---------------------------------------------------------------------------
# Shared phonetic rewrite rules — single source of truth used by BOTH the
# silent rewrite path (_ai_rewrite_single_chunk) and the human-readable
# analysis path (check_phonetic_transcription_ai), so the two can never
# describe/apply different rules for what counts as "needs fixing".
#
# Note: espeak-ng's [[...]] phoneme-override syntax is intentionally NOT
# offered here. That syntax expects espeak's own Kirshenbaum-based ASCII
# mnemonics (e.g. "w3:ld"), not standard Unicode IPA — Gemini reliably
# produces the latter but not the former, so asking it to emit [[...]]
# phonemes silently produces garbled/unrecognized input to espeak-ng.
# Kokoro's own inline IPA override (plain Unicode IPA, no brackets) is the
# only phonetic-override path offered, since Gemini can produce valid IPA.
# ---------------------------------------------------------------------------
_PHONETIC_RULES = (
    "Only flag/change words or formatting that affect pronunciation, such as:\n"
    "- Expand abbreviations (Dr. -> Doctor, St. -> Saint only when it is a name, "
    "  NASA -> N A S A, FBI -> F B I, etc.)\n"
    "- Spell out numbers and dates in words (2024 -> twenty twenty four, 3rd -> third)\n"
    "- Fix homophones, silent letters, or unusual stress by re-spelling the word\n\n"
    "FOREIGN PROPER NOUNS (mandatory, not optional):\n"
    "Any proper noun using non-English orthography or spelling conventions — accented "
    "letters (é, è, ç, œ, ü, etc.), unusual consonant clusters, silent letters, or "
    "foreign name endings — MUST be rewritten in some form. This applies to place names, "
    "personal names, military unit names, and any other proper noun. Do not leave a "
    "French, German, or other foreign-language name unchanged, even if you are uncertain "
    "of the exact native pronunciation — an approximate English rendering is always better "
    "than leaving the raw spelling for the default G2P to butcher.\n"
    "  Example: 'Amiens' -> 'Amyen' or ˈɑːmiˌæn\n"
    "  Example: 'Péronne' -> 'Peyron' or peɪˈrɒn\n"
    "  Example: 'Flixécourt' -> 'Flixaycoor'\n"
    "  Example: 'GUDERIAN' -> 'Guderian' (normalize casing)\n\n"
    "You MAY use the Kokoro inline phonetic override: replace a word directly with its "
    "standard Unicode IPA phonetic spelling (no brackets needed), using stress marks ˈ "
    "(primary) and ˌ (secondary). Example: Kokoro or Paris rewritten as kˈOkəɹO or ˈpæɹɪs. "
    "Do not use espeak-style double-bracket phoneme syntax (e.g. [[...]]) — it is not "
    "supported here and will not be pronounced correctly.\n\n"
    "Prosody: existing punctuation already controls intonation — "
    "; : , . ! ? — … \" ( ) \u201c \u201d all shape phrasing and pitch. "
    "Do not remove or alter punctuation; it is meaningful for Kokoro.\n\n"
    "- Prefer the simplest fix: only use inline IPA replacements for words that "
    "  the default G2P would misread (names, loanwords, acronyms). Plain English "
    "  respelling is preferred when it's simpler and equally accurate."
)

# Chatterbox has no inline phonetic-override syntax: IPA symbols, stress marks
# and [[...]] would be read literally (or dropped). Chatterbox rewrites must
# therefore use ordinary English letter respelling only.
_CHATTERBOX_PHONETIC_RULES = (
    "Only flag/change words or formatting that affect pronunciation, such as:\n"
    "- Expand abbreviations (Dr. -> Doctor, St. -> Saint only when it is a name, "
    "  NASA -> N A S A, FBI -> F B I, etc.)\n"
    "- Spell out numbers and dates in words (2024 -> twenty twenty four, 3rd -> third)\n"
    "- Respell words with unusual pronunciation using ordinary English letters\n\n"
    "FOREIGN PROPER NOUNS (mandatory, not optional):\n"
    "Any proper noun using non-English orthography or spelling conventions — accented "
    "letters (é, è, ç, œ, ü, etc.), unusual consonant clusters, silent letters, or "
    "foreign name endings — MUST be rewritten using ordinary English letters. This applies "
    "to place names, personal names, military unit names, and any other proper noun. Do not "
    "leave a French, German, or other foreign-language name unchanged, even if you are "
    "uncertain of the exact native pronunciation — an approximate English rendering is "
    "always better than leaving the raw spelling for the default G2P to butcher.\n"
    "  Example: 'Amiens' -> 'Amyen'\n"
    "  Example: 'Péronne' -> 'Peyron'\n"
    "  Example: 'Flixécourt' -> 'Flixaycoor'\n"
    "  Example: 'GUDERIAN' -> 'Guderian' (normalize casing)\n\n"
    "IMPORTANT: The Chatterbox TTS engine does NOT understand phonetic alphabets. "
    "Do NOT output IPA symbols, stress marks, slashes, or espeak [[...]] syntax. "
    "Use only ordinary English letters and normal punctuation — the ONLY square "
    "brackets you may ever output are the expressive tags listed at the end of "
    "these rules.\n\n"
    "Prosody: existing punctuation already controls intonation — "
    "; : , . ! ? — … \" ( ) \u201c \u201d all shape phrasing and pitch. "
    "Do not remove or alter punctuation; it is meaningful for Chatterbox."
)

# Chatterbox (Turbo) ships 19 expressive tokens trained natively into its
# tokenizer vocabulary: 9 non-speech vocal effects and 10 emotion/delivery
# styles. They are the only way to steer delivery from inside the text, so
# when AI rewriting runs for a Chatterbox engine we ask for these tags on top
# of the ordinary pronunciation fixes. Tags are lowercase and bracketed, and
# they are the sole bracketed syntax allowed for Chatterbox.
_CHATTERBOX_TAG_RULES = (
    "CHATTERBOX EXPRESSIVE TAGS (apply IN ADDITION to the pronunciation rewrites above):\n"
    "The selected Chatterbox model has 19 built-in special tokens that make the speech "
    "more natural, conversational, and expressive. Use ONLY these exact lowercase, "
    "square-bracketed tags, spelled exactly as shown:\n\n"
    "1) Vocal Sound Effects — insert inline or mid-sentence where a real speaker would "
    "physically react:\n"
    "   [laugh] [chuckle] [gasp] [sigh] [groan] [sniff] [cough] [clear throat] [shush]\n\n"
    "2) Emotion & Delivery Styles — place ONLY at the very beginning of a sentence or a "
    "distinct line, to set the tone for the words that follow:\n"
    "   [happy] [crying] [angry] [fear] [surprised] [whispering] [sarcastic] [dramatic] "
    "[narration] [advertisement]\n\n"
    "Tag instructions:\n"
    "- Analyze the context, emotion, and punctuation, then insert a tag only where a real "
    "human would naturally breathe, laugh, shift emotion, or react physically. Do NOT "
    "overuse them — not every sentence needs a tag.\n"
    "- Tags are always lowercase and enclosed in square brackets. Never translate, "
    "capitalize, alter, or invent a tag, and never emit any other bracketed syntax.\n"
    "- An emotion tag at the start of a line naturally bleeds into the words that follow, "
    "until a new tag or punctuation resets the pacing.\n"
    "- Do not change, add, or remove any of the author's words or punctuation while "
    "adding tags; the tags are the only insertions you may make.\n"
    "- The first tag may only be inserted where the input actually begins; never prepend "
    "a tag before the very first word of the supplied text."
)

# Rough char-to-token ratio used to size Gemini requests for the
# phonetic rewriter. ~4 chars/token is a defensible average for English
# prose, so 8K tokens ~= 32K characters of payload per request.
_AI_REWRITE_MAX_TOKENS = 8_000
_AI_REWRITE_CHARS_PER_TOKEN = 4
_AI_REWRITE_MAX_CHARS = _AI_REWRITE_MAX_TOKENS * _AI_REWRITE_CHARS_PER_TOKEN

# Break priority used when a stretch of text is too long for one request:
# end of a nearby paragraph -> end of sentence (.?!) -> clause (,;:) -> word.
_AI_PARAGRAPH_BREAK_RE = re.compile(r'\n+')
_AI_SENTENCE_BREAK_RE = re.compile(r'(?<=[.!?])\s+')
_AI_CLAUSE_BREAK_RE = re.compile(r'(?<=[,;:])\s+')

# A chapter below this size (in rough tokens) is never sent to Gemini as its
# own request: it is merged with the leading part of the next chapter so a
# request is never a tiny, context-poor snippet.
_AI_MIN_AGGREGATE_TOKENS = 1_000
_AI_MIN_AGGREGATE_CHARS = _AI_MIN_AGGREGATE_TOKENS * _AI_REWRITE_CHARS_PER_TOKEN

# Marker used to join several chapters into one request. The model is asked to
# preserve it verbatim; the response is split back on it, and if it is not
# preserved the unit falls back to one request per piece. The token itself is
# bracketed with newlines when joining, but only the bare token is searched for
# on the way back, so a dropped blank line still splits correctly.
_AI_PART_MARKER = '<VX_PART_BREAK>'
_AI_PART_JOIN = '\n\n' + _AI_PART_MARKER + '\n\n'

# Below this many characters of slack it is not worth splitting the next
# chapter just to top up a small unit; close the unit as-is instead.
_AI_MIN_PREFIX_CHARS = 200


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

border = 5

# Long titles/authors wrap at this width so the Book Details box (and with it
# the whole right column) stays compact instead of growing to the text length.
BOOK_DETAILS_WRAP_WIDTH = 260
