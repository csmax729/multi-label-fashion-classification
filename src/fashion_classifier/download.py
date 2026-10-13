"""Fetch a public Kaggle dataset over verified HTTPS, without embedding credentials."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath


def safe_extract(archive, output_dir, max_uncompressed_bytes=8 * 1024**3):
    output_dir = Path(output_dir)
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if (
            len(members) > 200_000
            or sum(item.file_size for item in members) > max_uncompressed_bytes
        ):
            raise ValueError("Archive exceeds safe extraction limits")
        for member in members:
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if (
                member.file_size
                and member.compress_size
                and member.file_size / member.compress_size > 5000
            ):
                raise ValueError("Archive member exceeds compression-ratio limit")
        normalized_names = [str(PurePosixPath(member.filename)) for member in members]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("Archive contains duplicate destination paths")
        missing = []
        verified = 0
        # A completed extraction from an interrupted previous attempt is reusable
        # only after every existing file matches its archive size and CRC.
        for member in members:
            if member.is_dir():
                continue
            destination = output_dir / member.filename
            if not destination.resolve().is_relative_to(output_dir.resolve()):
                raise ValueError("Existing destination symlink escapes extraction directory")
            if destination.is_symlink():
                raise ValueError("Refusing to reuse an extracted file symlink")
            if destination.exists():
                if not destination.is_file() or destination.stat().st_size != member.file_size:
                    raise FileExistsError(f"Existing file differs from the archive: {destination}")
                crc = 0
                with destination.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                if crc != member.CRC:
                    raise FileExistsError(
                        f"Existing file checksum differs from the archive: {destination}"
                    )
                verified += 1
            else:
                missing.append(member)
        if not missing:
            return {"verified_existing_files": verified, "extracted_new_files": 0}
        output_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=output_dir))
        try:
            # All entries have been checked before any are written. ZipFile checks CRCs.
            for member in missing:
                source.extract(member, staging)
            files = [path for path in staging.rglob("*") if path.is_file()]
            for path in files:
                destination = output_dir / path.relative_to(staging)
                if not destination.resolve().is_relative_to(output_dir.resolve()):
                    raise ValueError("Existing destination symlink escapes extraction directory")
                if destination.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite existing dataset file: {destination}"
                    )
            for path in files:
                destination = output_dir / path.relative_to(staging)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), destination)
        finally:
            shutil.rmtree(staging)
        return {"verified_existing_files": verified, "extracted_new_files": len(missing)}


def _select_dataset_layout(archive, output_dir):
    """The public ZIP repeats the same dataset at root and under myntradataset/."""
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        style_members = [
            member for member in members if PurePosixPath(member.filename).name == "styles.csv"
        ]
        candidates = [Path(output_dir) / member.filename for member in style_members]
        if not candidates or any(not (path.parent / "images").is_dir() for path in candidates):
            raise ValueError("Archive must contain styles.csv with a sibling images directory")
        candidates.sort(key=lambda path: (len(path.relative_to(output_dir).parts), str(path)))
        selected = candidates[0]
        csv_digest = hashlib.sha256(selected.read_bytes()).digest()
        if any(hashlib.sha256(path.read_bytes()).digest() != csv_digest for path in candidates[1:]):
            raise ValueError(
                "Archive contains conflicting styles.csv layouts; choose a source explicitly"
            )
        image_indexes = []
        for style in candidates:
            relative = style.parent.relative_to(output_dir).as_posix()
            prefix = "images/" if relative == "." else f"{relative}/images/"
            image_indexes.append(
                {
                    member.filename[len(prefix) :]: (member.file_size, member.CRC)
                    for member in members
                    if not member.is_dir() and member.filename.startswith(prefix)
                }
            )
        if any(index != image_indexes[0] for index in image_indexes[1:]):
            raise ValueError("Archive contains conflicting duplicated image layouts")
        return selected, {
            "selected_layout": selected.parent.relative_to(output_dir).as_posix(),
            "ignored_duplicate_layouts": [
                path.parent.relative_to(output_dir).as_posix() for path in candidates[1:]
            ],
            "duplicate_styles_identical": len(candidates) > 1,
            "duplicate_images_identical_crc_and_size": len(candidates) > 1,
            "images_in_selected_layout": len(image_indexes[0]),
        }


def download_dataset(
    output_dir, dataset="paramaggarwal/fashion-product-images-small", max_bytes=2 * 1024**3
):
    parts = dataset.split("/")
    if len(parts) != 2 or any(
        not part or not all(char.isalnum() or char in "_-" for char in part) for part in parts
    ):
        raise ValueError("Dataset must be a public owner/name slug")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "dataset.zip"
    partial = output_dir / "dataset.zip.part"
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"
    if not archive.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "fashion-classifier/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                resume = response.status == 206
                if resume and not response.headers.get("Content-Range", "").startswith(
                    f"bytes {offset}-"
                ):
                    raise ValueError("Server returned an unexpected resume offset")
                received = offset if resume else 0
                with partial.open("ab" if resume else "wb") as handle:
                    while block := response.read(1024 * 1024):
                        received += len(block)
                        if received > max_bytes:
                            raise ValueError("Download exceeds maximum archive size")
                        handle.write(block)
            if not zipfile.is_zipfile(partial):
                raise ValueError(
                    "Kaggle returned a non-ZIP response; public download may require authenticated access. Supply the dataset through environment settings or download it manually, then prepare it."
                )
            os.replace(partial, archive)
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Public Kaggle download failed: {error}. Required destinations: www.kaggle.com and its download redirect storage host. Existing partial download is retained; no credentials are required for this attempted public endpoint."
            ) from error
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    extraction = safe_extract(archive, output_dir)
    styles, layout = _select_dataset_layout(archive, output_dir)
    metadata = {
        "dataset_kind": "public",
        "source": dataset,
        "url": url,
        "archive_sha256": digest.hexdigest(),
        "integrity": "Verified HTTPS transport and ZIP CRC verification; recorded SHA256 is a local provenance digest, not a publisher-supplied checksum.",
        "layout": layout,
        "extraction": extraction,
    }
    marker = styles.parent / "download_metadata.json"
    if marker.exists():
        prior = json.loads(marker.read_text())
        if prior.get("source") != dataset or prior.get("archive_sha256") != digest.hexdigest():
            raise FileExistsError(
                "Existing download metadata belongs to a different archive/source"
            )
    marker.write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        **metadata,
        "styles": str(styles.resolve()),
        "images_dir": str((styles.parent / "images").resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw", type=Path)
    parser.add_argument("--dataset", default="paramaggarwal/fashion-product-images-small")
    args = parser.parse_args()
    try:
        print(json.dumps(download_dataset(args.output_dir, args.dataset), indent=2))
    except (RuntimeError, ValueError, FileExistsError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Dataset download blocked: {error}\n")


if __name__ == "__main__":
    main()
