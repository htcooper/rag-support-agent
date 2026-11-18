from __future__ import annotations

from pathlib import Path

import pytest

from rag_support.extract import ExtractionError, ExtractLimits, safe_extract
from tests.conftest import ROOT, ArchiveFactory

DOC = f"{ROOT}/docs/en/docs"


def test_extracts_only_allowlisted_files(
    make_archive: ArchiveFactory, tmp_path: Path
) -> None:
    archive = make_archive(
        [
            (f"{DOC}/index.md", b"# Home\n"),
            (f"{DOC}/tutorial/first-steps.md", b"# First steps\n"),
            (f"{DOC}/img/logo.png", b"\x89PNG"),
            (f"{DOC}/js/custom.js", b"alert(1)"),
            (f"{DOC}/helper.py", b"import os"),
            (f"{ROOT}/docs_src/first_steps/tutorial001.py", b"app = 1\n"),
            (f"{ROOT}/docs_src/templates/item.html", b"<html></html>"),
            (f"{ROOT}/docs/es/docs/index.md", b"# Inicio\n"),
            (f"{ROOT}/setup.py", b"import os"),
        ]
    )
    files = safe_extract(archive, tmp_path / "raw")
    assert sorted(files) == [
        "docs/en/docs/index.md",
        "docs/en/docs/tutorial/first-steps.md",
        "docs_src/first_steps/tutorial001.py",
    ]
    assert files["docs/en/docs/index.md"].read_bytes() == b"# Home\n"
    extracted = [p for p in (tmp_path / "raw").rglob("*") if p.is_file()]
    assert len(extracted) == 3


def test_rejects_symlink_in_code_examples(
    make_archive: ArchiveFactory, tmp_path: Path
) -> None:
    archive = make_archive([(f"{ROOT}/docs_src/x/evil.py", None, "../../../.env")])
    with pytest.raises(ExtractionError, match="link or special file"):
        safe_extract(archive, tmp_path / "raw")


def test_rejects_path_traversal(make_archive: ArchiveFactory, tmp_path: Path) -> None:
    archive = make_archive([(f"{DOC}/../../../../evil.md", b"x")])
    with pytest.raises(ExtractionError, match="unsafe path"):
        safe_extract(archive, tmp_path / "raw")
    assert not (tmp_path / "evil.md").exists()


def test_rejects_absolute_path(make_archive: ArchiveFactory, tmp_path: Path) -> None:
    archive = make_archive([("/etc/evil.md", b"x")])
    with pytest.raises(ExtractionError, match="unsafe path"):
        safe_extract(archive, tmp_path / "raw")


def test_rejects_symlink_in_docs(make_archive: ArchiveFactory, tmp_path: Path) -> None:
    archive = make_archive([(f"{DOC}/link.md", None, "/etc/passwd")])
    with pytest.raises(ExtractionError, match="link or special file"):
        safe_extract(archive, tmp_path / "raw")


def test_rejects_oversize_file(make_archive: ArchiveFactory, tmp_path: Path) -> None:
    archive = make_archive([(f"{DOC}/big.md", b"x" * 101)])
    with pytest.raises(ExtractionError, match="size cap"):
        safe_extract(
            archive, tmp_path / "raw", limits=ExtractLimits(max_file_bytes=100)
        )


def test_rejects_total_over_cap(make_archive: ArchiveFactory, tmp_path: Path) -> None:
    archive = make_archive([(f"{DOC}/a.md", b"x" * 60), (f"{DOC}/b.md", b"x" * 60)])
    with pytest.raises(ExtractionError, match="total-size cap"):
        safe_extract(
            archive, tmp_path / "raw", limits=ExtractLimits(max_total_bytes=100)
        )


def test_rejects_too_many_files(make_archive: ArchiveFactory, tmp_path: Path) -> None:
    archive = make_archive([(f"{DOC}/{i}.md", b"x") for i in range(3)])
    with pytest.raises(ExtractionError, match="file-count cap"):
        safe_extract(archive, tmp_path / "raw", limits=ExtractLimits(max_files=2))


def test_rejects_archive_over_scanned_cap(
    make_archive: ArchiveFactory, tmp_path: Path
) -> None:
    archive = make_archive([(f"{ROOT}/blob.bin", b"x" * 200)])
    with pytest.raises(ExtractionError, match="scanned-size cap"):
        safe_extract(
            archive, tmp_path / "raw", limits=ExtractLimits(max_scanned_bytes=100)
        )


def test_refuses_non_empty_destination(
    make_archive: ArchiveFactory, tmp_path: Path
) -> None:
    archive = make_archive([(f"{DOC}/index.md", b"# Home\n")])
    dest = tmp_path / "raw"
    dest.mkdir()
    (dest / "leftover.md").write_text("old")
    with pytest.raises(ExtractionError, match="not empty"):
        safe_extract(archive, dest)
