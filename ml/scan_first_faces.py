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


def first_face_for_image(path: Path, min_conf: float, model: int, auto_model: bool):
    if cv2 is None:
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    faces = detect_faces(img, min_conf=min_conf, model_selection=model)
    if faces:
        return faces[0]
    if auto_model:
        alt_model = 1 - model  # switch between 0 and 1
        faces_alt = detect_faces(img, min_conf=min_conf, model_selection=alt_model)
        if faces_alt:
            return faces_alt[0]
    return None


def main(argv: List[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        sys.stderr.write(__doc__ + '\n')
        return 0
    stop_first = False
    if "--stop-first" in argv:
        stop_first = True
        argv.remove("--stop-first")
    # Defaults
    min_conf = 0.5
    model = 0
    auto_model = False
    max_images = None
    # Parse numeric / flag args (simple manual parse)
    def _extract_value(flag: str, cast):
        if flag in argv:
            idx = argv.index(flag)
            try:
                val = cast(argv[idx + 1])
                del argv[idx:idx + 2]
                return val
            except Exception:
                del argv[idx]
        return None
    v = _extract_value("--min-conf", float)
    if v is not None:
        min_conf = max(0.0, min(1.0, v))
    v = _extract_value("--model", int)
    if v is not None and v in (0, 1):
        model = v
    v = _extract_value("--max-images", int)
    if v is not None and v > 0:
        max_images = v
    if "--auto-model" in argv:
        auto_model = True
        argv.remove("--auto-model")
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
    count = 0
    for img_path in iter_images(root, exts):
        if max_images is not None and count >= max_images:
            break
        t0 = time.perf_counter()
        face = first_face_for_image(img_path, min_conf=min_conf, model=model, auto_model=auto_model)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        out = {
            "path": str(img_path),
            "face": face,
            "duration_ms": round(dt_ms, 2),
        }
        if face is not None:
            out["used_model"] = model if not auto_model or face else model
        print(json.dumps(out, ensure_ascii=False))
        count += 1
        if stop_first and face is not None:
            break
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
