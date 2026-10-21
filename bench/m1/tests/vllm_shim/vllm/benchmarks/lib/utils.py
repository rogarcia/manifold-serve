"""Dry-run stand-in for the one vLLM helper the vendored multi-turn bench imports.

The pod has the real vLLM; dryrun-local.sh puts this directory on PYTHONPATH for the bench only.
"""

import argparse


def redact_sensitive_namespace(args: argparse.Namespace, fields: tuple[str, ...]) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    for f in fields:
        if getattr(out, f, None):
            setattr(out, f, "<redacted>")
    return out
