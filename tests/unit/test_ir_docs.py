"""Tests enforcing the generated IR tag/field spec document."""

from __future__ import annotations

from pathlib import Path

from jaxstanv5.ir import render_ir_v1_tag_spec

TAG_SPEC_PATH = Path(__file__).parent.parent.parent / "docs" / "ir-v1-tags.md"


def test_tag_spec_document_matches_registry() -> None:
    assert TAG_SPEC_PATH.read_text(encoding="utf-8") == render_ir_v1_tag_spec(), (
        "docs/ir-v1-tags.md is out of date. Run scripts/regenerate_ir_golden.py "
        "and review the diff: any tag or field change is a wire-format change."
    )
