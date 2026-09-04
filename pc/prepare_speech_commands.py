#!/usr/bin/env python3
"""Download and extract Google Speech Commands v2 outside Git."""
import argparse
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with out.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out.name) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/speech_commands", help="Extraction directory")
    parser.add_argument("--archive", default="data/speech_commands_v0.02.tar.gz")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    archive = Path(args.archive)
    if not args.skip_download and not archive.exists():
        download(URL, archive)

    root.mkdir(parents=True, exist_ok=True)
    marker = root / ".extracted"
    if marker.exists():
        print(f"Already extracted: {root}")
        return

    print(f"Extracting {archive} -> {root}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    marker.write_text("ok\n", encoding="utf-8")
    print(f"Ready: {root}")


if __name__ == "__main__":
    main()
