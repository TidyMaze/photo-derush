"""Scan images in a directory and output first detected face per image.

Usage (module):
  python -m ml.scan_first_faces /path/to/images [--ext jpg png] [--stop-first]

Outputs one JSON line per image with keys:
  path, face (or null), duration_ms

Face format: {"x": int, "y": int, "w": int, "h": int, "score": float}

--stop-first: terminate after printing the first image that has a detected face.

Returns exit code 0 always (non-fatal) so it can be piped.
"""
from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Iterable, List

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

from .face_detection import detect_faces

DEFAULT_EXTS = {"jpg", "jpeg", "png", "bmp"}


def iter_images(root: Path, exts: Iterable[str]) -> Iterable[Path]:
    exts_lower = {e.lower().lstrip('.') for e in exts}
    for p in sorted(root.rglob('*')):
        if not p.is_dir() and p.suffix.lower().lstrip('.') in exts_lower:
            yield p


def first_face_for_image(path: Path):
    if cv2 is None:
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    faces = detect_faces(img)
    if not faces:
        return None
    return faces[0]


def main(argv: List[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        sys.stderr.write(__doc__ + '\n')
        return 0
    stop_first = False
    if "--stop-first" in argv:
        stop_first = True
        argv.remove("--stop-first")
    root = Path(argv[0])
    if not root.exists():
        sys.stderr.write(f"Directory not found: {root}\n")
        return 0
    # parse optional extensions
    exts = DEFAULT_EXTS
    if "--ext" in argv:
        idx = argv.index("--ext")
        custom: List[str] = []
        for tok in argv[idx + 1:]:
            if tok.startswith('--'):
                break
            custom.append(tok)
        if custom:
            exts = {e.lower().lstrip('.') for e in custom}
    for img_path in iter_images(root, exts):
        t0 = time.perf_counter()
        face = first_face_for_image(img_path)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        out = {
            "path": str(img_path),
            "face": face,
            "duration_ms": round(dt_ms, 2),
        }
        print(json.dumps(out, ensure_ascii=False))
        if stop_first and face is not None:
            break
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
