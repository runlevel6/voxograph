"""voxograph.ui.bridge - helpers for talking to the Chatterbox bridge."""
import json


def _extract_bridge_json(stdout):
    """Return the JSON object emitted by the Chatterbox bridge.

    The bridge's stdout can carry stray prints from third-party libraries
    (e.g. perth's "loaded PerthNet" line), so scan the lines and return the
    last one that parses as a JSON object rather than assuming clean output.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise RuntimeError(
        f"Chatterbox bridge produced no JSON result. stdout={stdout!r}"
    )
