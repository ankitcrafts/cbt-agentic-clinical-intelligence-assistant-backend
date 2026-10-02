#!/usr/bin/env python3
"""Lightweight CI checks that avoid loading HuggingFace embeddings."""

from pathlib import Path
import compileall
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ok = compileall.compile_dir(str(ROOT / "app"), quiet=1, force=True)
    if not ok:
        print("compileall failed")
        return 1

    from app.rag.guideline_catalog import GUIDELINE_CATALOG, metadata_for_filename
    from app.services.followup_service import (
        CONFIDENCE_FOLLOW_UP_THRESHOLD,
        MAX_FOLLOW_UP_ROUNDS,
    )

    assert len(GUIDELINE_CATALOG) >= 5, "expected multi-disease guideline catalog"
    assert CONFIDENCE_FOLLOW_UP_THRESHOLD == 0.70
    assert MAX_FOLLOW_UP_ROUNDS == 1

    sample = metadata_for_filename("NG109_lower_uti.pdf")
    assert sample["doc_id"] == "nice_ng109_uti"

    print("CI checks passed")
    print(f"  guidelines: {len(GUIDELINE_CATALOG)}")
    print(f"  follow_up_threshold: {CONFIDENCE_FOLLOW_UP_THRESHOLD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
