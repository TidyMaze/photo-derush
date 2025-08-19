"""Annotate faces in images and export copies with rectangles.

Usage:
  python -m ml.annotate_faces /path/to/images [--ext jpg png] [--min-conf 0.5] [--model 0|1] [--overwrite]

Rules:
  - For each image with >=1 faces, create <original_name>_faces<extension> alongside original.
  - Existing output is skipped unless --overwrite provided.
  - Uses strict face_detection.detect_faces (will raise if dependencies missing).
"""
from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from typing import Iterable, List, Sequence
import cv2  # type: ignore

from .face_detection import detect_faces

DEFAULT_EXTS = {"jpg", "jpeg", "png", "bmp"}


def iter_images(root: Path, exts: Sequence[str]) -> Iterable[Path]:
    exts_lower = {e.lower().lstrip('.') for e in exts}
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower().lstrip('.') in exts_lower:
            yield p


def annotate_image(path: Path, faces, overwrite: bool) -> bool:
    if not faces:
        return False
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    h, w = img.shape[:2]
    thickness = max(2, min(h, w) // 300)
    color = (0, 255, 0)  # BGR green
    for f in faces:
        x, y, bw, bh = f["x"], f["y"], f["w"], f["h"]
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, thickness)
    out_path = path.with_name(f"{path.stem}_faces{path.suffix}")
    if out_path.exists() and not overwrite:
        return False
    if not cv2.imwrite(str(out_path), img):
        raise RuntimeError(f"Failed to write annotated image: {out_path}")
    return True


def main(argv: List[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    if not argv or argv[0] in {"-h", "--help"}:
        sys.stderr.write(__doc__ + "\n")
        return 0
    overwrite = False
    if "--overwrite" in argv:
        overwrite = True
        argv.remove("--overwrite")
    # parameters
    def _extract(flag: str, cast, default):
        if flag in argv:
            idx = argv.index(flag)
            try:
                val = cast(argv[idx + 1])
                del argv[idx:idx + 2]
                return val
            except Exception:
                del argv[idx]
        return default
    min_conf = _extract("--min-conf", float, 0.5)
    if min_conf < 0: min_conf = 0.0
    if min_conf > 1: min_conf = 1.0
    model = _extract("--model", int, 0)
    if model not in (0, 1):
        model = 0
    root = Path(argv[0])
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    exts = DEFAULT_EXTS
    if "--ext" in argv:
        idx = argv.index("--ext")
        custom: List[str] = []
        for tok in argv[idx + 1:]:
            if tok.startswith('--'):
                break
            custom.append(tok)
        if custom:
            exts = {c.lower().lstrip('.') for c in custom}
    total = 0
    annotated = 0
    t_start = time.perf_counter()
    for img_path in iter_images(root, sorted(exts)):
        total += 1
        t0 = time.perf_counter()
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        faces = detect_faces(img, min_conf=min_conf, model_selection=model)
        wrote = annotate_image(img_path, faces, overwrite=overwrite)
        if wrote:
            annotated += 1
        line = {
            "path": str(img_path),
            "faces": len(faces),
            "written": wrote,
            "duration_ms": round((time.perf_counter() - t0)*1000, 2),
        }
        if faces:
            line["scores"] = [round(f["score"], 4) for f in faces]
        print(json.dumps(line, ensure_ascii=False))
    summary = {
        "summary": True,
        "images": total,
        "annotated": annotated,
        "min_conf": min_conf,
        "model": model,
        "duration_s": round(time.perf_counter() - t_start, 2)
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

