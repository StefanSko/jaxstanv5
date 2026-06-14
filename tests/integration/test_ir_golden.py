"""Golden-file tests pinning the IR v1 wire format."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from _ir_golden_models import GoldenIRCase, golden_ir_cases

from jaxstanv5.ir import canonical_bytes, meta_from_dict, meta_to_dict

GOLDEN_DIR = Path(__file__).parent.parent / "golden_ir"

REGENERATE_HINT = (
    "Golden IR files pin the v1 wire format. If this change is deliberate, run "
    "scripts/regenerate_ir_golden.py, review the diff, and decide whether the "
    "format version must change."
)


def _case_ids() -> list[str]:
    return [case.name for case in golden_ir_cases()]


@pytest.mark.parametrize("case", golden_ir_cases(), ids=_case_ids())
def test_golden_document_matches_current_encoding(case: GoldenIRCase) -> None:
    golden_path = GOLDEN_DIR / f"{case.name}.json"

    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    assert meta_to_dict(case.meta) == golden, REGENERATE_HINT


@pytest.mark.parametrize("case", golden_ir_cases(), ids=_case_ids())
def test_golden_document_decodes_to_current_metadata(case: GoldenIRCase) -> None:
    golden_path = GOLDEN_DIR / f"{case.name}.json"

    golden = json.loads(golden_path.read_text(encoding="utf-8"))

    assert meta_from_dict(golden) == case.meta, REGENERATE_HINT


def test_golden_canonical_hashes_are_pinned() -> None:
    recorded = json.loads((GOLDEN_DIR / "hashes.json").read_text(encoding="utf-8"))

    current = {
        case.name: hashlib.sha256(canonical_bytes(case.meta)).hexdigest()
        for case in golden_ir_cases()
    }

    assert current == recorded, REGENERATE_HINT
