"""Fetch the pinned face detector into models/ and verify its bytes.

    python -m app.fetch_models

The app imports nothing from the internet at request time. This command is the
one place a weight file is allowed to arrive, and it refuses to leave a file on
disk unless the SHA-256 matches the pin in ``app._detector``. Running it twice
is a no-op; running it after a partial download repairs the file.

Licence: this blob is Apache-2.0 (OpenCV Zoo). The matching licence text is
written next to it so the shipped artefact carries its own terms.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from app._detector import MODEL_FILE, MODEL_SHA256, MODELS_DIR

SOURCE = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    f"models/face_detection_yunet/{MODEL_FILE}"
)
LICENCE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/LICENSE"
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stage(data: bytes, target: Path) -> None:
    """Write via .partial then rename: a killed download leaves no half model."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")
    partial.write_bytes(data)
    partial.replace(target)


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def main() -> int:
    target = MODELS_DIR / MODEL_FILE
    if target.is_file() and sha256_of(target) == MODEL_SHA256:
        print(f"OK   {target.name} already matches {MODEL_SHA256[:16]}...")
    else:
        print(f"GET  {SOURCE}")
        data = download(SOURCE)
        digest = hashlib.sha256(data).hexdigest()
        if digest != MODEL_SHA256:
            print(f"FAIL downloaded {len(data)} bytes")
            print(f"     sha256 got      {digest}")
            print(f"     sha256 expected {MODEL_SHA256}")
            return 1
        stage(data, target)
        print(f"OK   wrote {target} ({len(data)} bytes, sha256 verified)")

    licence = MODELS_DIR / "LICENSE-yunet.txt"
    if not licence.is_file():
        stage(download(LICENCE_URL), licence)
        print(f"OK   wrote {licence} (Apache-2.0)")

    print()
    print("License summary for files in", MODELS_DIR)
    print(f"  {MODEL_FILE}  Apache-2.0 (OpenCV Zoo / Shiqi Yu et al.)")
    print("  Weights by Shiqi Yu et al., 'YuNet: A Tiny Millisecond-level Face Detector'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
