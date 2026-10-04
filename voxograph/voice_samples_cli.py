#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless builder for the Kokoro voice sample library used by Chatterbox.

Run with the voxograph venv (Kokoro must be importable):

    python voice_samples_cli.py --list
    python voice_samples_cli.py --voice af_heart --force
    python voice_samples_cli.py --all
    python voice_samples_cli.py --all --lang a --dir ~/my_samples

Each voice is rendered by Kokoro into a 24 kHz WAV that Chatterbox can clone,
so the presets are named exactly like the Kokoro voices in the GUI dropdown.
"""
import argparse
import sys
import threading

import torch.cuda

import voxograph.core as core
from voxograph.voices import voices


def all_voices(lang_codes):
    selected = []
    for lang_code, lang_voices in voices.items():
        if not lang_codes or lang_code in lang_codes:
            selected.extend(lang_voices)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--voice', action='append', default=[],
                        help='Kokoro voice code to build (repeatable), e.g. af_heart')
    parser.add_argument('--all', action='store_true', help='build every voice that is missing')
    parser.add_argument('--list', action='store_true', help='show which samples are built, then exit')
    parser.add_argument('--lang', action='append', default=[],
                        help='build every voice of these language codes (a, b, e, f, h, i, j, p, z)')
    parser.add_argument('--dir', help=f'samples directory (default: {core.DEFAULT_VOICE_SAMPLES_DIR})')
    parser.add_argument('--force', action='store_true', help='rebuild samples that already exist')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=('cuda', 'cpu'))
    args = parser.parse_args()

    settings = core.load_settings()
    if args.dir:
        settings['voice_samples_dir'] = args.dir

    targets = args.voice or (all_voices(args.lang) if (args.all or args.lang) else [])
    if args.list or not targets:
        _list_samples(settings)
        if args.list or targets:
            return 0
        print('\nNothing to do. Pass --voice CODE, --all, or --list.')
        return 1

    unknown = [v for v in targets if v not in all_voices(None)]
    if unknown:
        print(f"Unknown voice code(s): {', '.join(unknown)}", file=sys.stderr)
        return 2

    if args.device == 'cuda' and torch.cuda.is_available():
        torch.set_default_device('cuda')

    print(f'Samples directory: {core.voice_samples_dir(settings)}')
    print(f'Device: {args.device}')
    missing = len(core.missing_voice_samples(targets, settings))
    if not args.force and not missing:
        print('All requested samples are already built. Use --force to rebuild.')
        _list_samples(settings)
        return 0

    stop_event = threading.Event()
    _stop_on_sigint(stop_event)
    written = core.generate_voice_samples(
        targets, overwrite=args.force, stop_event=stop_event, settings=settings,
        progress=_report)

    print(f'\nBuilt {len(written)} sample(s).')
    return 0


def _report(voice, path, done, total):
    status = 'ok' if path else 'FAILED'
    print(f'[{done}/{total}] {voice}: {status}')


def _stop_on_sigint(stop_event):
    import signal

    def handler(_sig, _frame):
        print('\nStopping after the current voice…')
        stop_event.set()

    signal.signal(signal.SIGINT, handler)


def _list_samples(settings=None):
    settings = settings if settings is not None else core.load_settings()
    print(f'\nSample directory: {core.voice_samples_dir(settings)}')
    for lang_code, lang_voices in voices.items():
        built = [v for v in lang_voices if core.voice_sample_exists(v, settings)]
        missing = [v for v in lang_voices if v not in built]
        print(f'  {lang_code}: {len(built)}/{len(lang_voices)} built'
              + (f' — missing: {", ".join(missing)}' if missing else ''))


if __name__ == '__main__':
    sys.exit(main())