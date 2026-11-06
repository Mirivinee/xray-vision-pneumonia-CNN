"""Verify prepared image files against the project dataset manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "dataset_manifest.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "data" / "dataset_manifest_summary.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(split_dir: Path, manifest_path: Path, summary_path: Path) -> None:
    with summary_path.open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if sha256_file(manifest_path) != summary["manifest_sha256"]:
        raise ValueError(f"Manifest SHA-256 differs from {summary_path}")

    counts: Counter[tuple[str, str]] = Counter()
    hashes: set[str] = set()
    listed_paths: set[str] = set()
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            relative_path = Path(row["relative_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Manifest path must stay inside the split root: {relative_path}")
            if len(relative_path.parts) < 3:
                raise ValueError(f"Expected split/class/file path, got: {relative_path}")

            split, class_name = relative_path.parts[:2]
            image_path = split_dir / relative_path
            normalized_path = relative_path.as_posix()
            if normalized_path in listed_paths:
                raise ValueError(f"Duplicate manifest path: {relative_path}")
            listed_paths.add(normalized_path)
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image is missing: {image_path}")
            actual_hash = sha256_file(image_path)
            if actual_hash != row["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {image_path}")
            if actual_hash in hashes:
                raise ValueError(f"Duplicate image content found: {image_path}")
            hashes.add(actual_hash)
            counts[(split, class_name)] += 1

    actual_paths = {
        path.relative_to(split_dir).as_posix()
        for split in summary["splits"]
        for class_name in summary["class_names"]
        for path in (split_dir / split / class_name).rglob("*")
        if path.is_file()
    }
    if actual_paths != listed_paths:
        raise ValueError(
            f"Prepared image paths differ from the manifest: "
            f"{len(actual_paths - listed_paths)} unlisted, "
            f"{len(listed_paths - actual_paths)} missing."
        )

    expected = summary["counts"]
    actual = {
        split: {class_name: counts[(split, class_name)] for class_name in summary["class_names"]}
        for split in summary["splits"]
    }
    if actual != expected:
        raise ValueError(f"Split counts differ from the manifest. Expected {expected}; found {actual}")
    print(f"Verified {len(hashes):,} files against {manifest_path} (SHA-256 {summary['manifest_sha256']})")
    print(json.dumps(actual, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True, help="Prepared root containing train/, val/, and test/.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    arguments = parser.parse_args()
    validate_dataset(arguments.split_dir.resolve(), arguments.manifest.resolve(), arguments.summary.resolve())


if __name__ == "__main__":
    main()
