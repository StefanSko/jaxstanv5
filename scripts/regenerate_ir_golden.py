"""Regenerate the golden IR v1 documents, canonical hashes, and tag spec.

Run from the repository root:

    uv run scripts/regenerate_ir_golden.py

Golden files pin the wire format. Review every diff this produces and decide
whether the change requires a format version bump before committing it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from integration._ir_golden_models import golden_ir_cases  # noqa: E402
from jaxstanv5.ir import canonical_bytes, meta_to_dict, render_ir_v1_tag_spec  # noqa: E402

GOLDEN_DIR = REPO_ROOT / "tests" / "golden_ir"
TAG_SPEC_PATH = REPO_ROOT / "docs" / "ir-v1-tags.md"


def main() -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    for case in golden_ir_cases():
        document = meta_to_dict(case.meta)
        path = GOLDEN_DIR / f"{case.name}.json"
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        hashes[case.name] = hashlib.sha256(canonical_bytes(case.meta)).hexdigest()
        print(f"wrote {path.relative_to(REPO_ROOT)}")

    hashes_path = GOLDEN_DIR / "hashes.json"
    hashes_path.write_text(
        json.dumps(hashes, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {hashes_path.relative_to(REPO_ROOT)}")

    TAG_SPEC_PATH.write_text(render_ir_v1_tag_spec(), encoding="utf-8")
    print(f"wrote {TAG_SPEC_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
