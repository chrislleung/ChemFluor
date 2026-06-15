from __future__ import annotations

from pathlib import Path

import pytest

from src import config


def touch_csv(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder\n", encoding="utf-8")
    return path


def test_resolve_existing_path_returns_explicit_existing_path(tmp_path: Path) -> None:
    explicit = touch_csv(tmp_path / "custom" / "chemfluor.csv")
    fallback = touch_csv(tmp_path / "data" / "chemfluor_data.csv")

    resolved = config.resolve_existing_path(
        explicit_path=explicit,
        fallback_paths=[fallback],
        description="ChemFluor dataset",
    )

    assert resolved == explicit


def test_resolve_existing_path_raises_for_explicit_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    fallback = touch_csv(tmp_path / "data" / "chemfluor_data.csv")

    with pytest.raises(FileNotFoundError, match="explicit path"):
        config.resolve_existing_path(
            explicit_path=missing,
            fallback_paths=[fallback],
            description="ChemFluor dataset",
        )


def test_resolve_existing_path_prefers_data_layout_over_root_layout(tmp_path: Path) -> None:
    data_path = touch_csv(tmp_path / "data" / "chemfluor_data.csv")
    root_path = touch_csv(tmp_path / "chemfluor_data.csv")

    resolved = config.resolve_existing_path(
        explicit_path=None,
        fallback_paths=[data_path, root_path],
        description="ChemFluor dataset",
    )

    assert resolved == data_path


def test_resolve_existing_path_falls_back_to_root_layout(tmp_path: Path) -> None:
    data_path = tmp_path / "data" / "chemfluor_data.csv"
    root_path = touch_csv(tmp_path / "chemfluor_data.csv")

    resolved = config.resolve_existing_path(
        explicit_path=None,
        fallback_paths=[data_path, root_path],
        description="ChemFluor dataset",
    )

    assert resolved == root_path


def test_solvent_descriptor_resolution_uses_same_priority(tmp_path: Path) -> None:
    data_path = touch_csv(tmp_path / "data" / "solvent_descriptors.csv")
    root_path = touch_csv(tmp_path / "solvent_descriptors.csv")

    resolved = config.resolve_existing_path(
        explicit_path=None,
        fallback_paths=[data_path, root_path],
        description="Solvent descriptor file",
    )

    assert resolved == data_path
