"""Merge one KD configuration's student blocks into a per-fold results file.

The gated-KD runs (Task 3 of the 2026-11-17 resubmission plan) produce their
own ``results.json`` per fold.  This script adds their ``students`` entries to
the existing ``runs/combined/results_fold{N}.json`` without touching the
``teacher``/``baseline`` blocks or any student key already present.

    python scripts/merge_results.py --base runs/combined/results_fold0.json \
        --extra runs/combined/results_gated_fold0.json --fold 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_results(base: dict, extra: dict, fold: int) -> dict:
    """Return ``base`` with ``extra[f"fold_{fold}"]["students"]`` added.

    Raises ``KeyError`` if a student key already exists in ``base`` (results
    are never silently overwritten) or if either file lacks the fold.
    """
    key = f"fold_{fold}"
    if key not in base or key not in extra:
        raise KeyError(f"{key} missing from base or extra results")
    dest = base[key].setdefault("students", {})
    for name, block in extra[key].get("students", {}).items():
        if name in dest:
            raise KeyError(f"{key}/students/{name} already present in base; refusing to overwrite")
        dest[name] = block
    return base


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, help="results_fold{N}.json to update in place")
    ap.add_argument("--extra", required=True, help="results.json produced by the new run")
    ap.add_argument("--fold", type=int, required=True)
    args = ap.parse_args(argv)
    base_path = Path(args.base)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    extra = json.loads(Path(args.extra).read_text(encoding="utf-8"))
    merged = merge_results(base, extra, args.fold)
    base_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    added = sorted(extra[f"fold_{args.fold}"].get("students", {}))
    print(f"merged {len(added)} student block(s) into {base_path}: {', '.join(added)}")


if __name__ == "__main__":
    main()
