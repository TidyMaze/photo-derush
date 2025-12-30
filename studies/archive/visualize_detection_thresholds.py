"""Visualize detection sensitivity to confidence and area thresholds.

Usage: python scripts/visualize_detection_thresholds.py /path/to/image.jpg

This script calls `src.object_detection.detect_objects` across a grid of
confidence thresholds and area ratios, then plots number of kept detections
and shows sample overlay images.
"""
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.object_detection import detect_objects


def draw_boxes_on_image(img: Image.Image, detections, box_width: int = 8, font_scale: float = 1.0, img_scale: float = 1.0):
    # KISS: draw fixed-thickness bounding boxes and fixed-size text with
    # exact text background matching the text bbox. No padding, no scaling.
    out = img.copy()
    draw = ImageDraw.Draw(out)
    # Fixed parameters (even larger font, fail fast)
    stroke = max(2, int(box_width) * 2)
    fixed_font_size = 32
    # Cross-platform Arial font discovery
    import platform
    def find_arial_font():
        candidates = []
        sys_plat = platform.system()
        if sys_plat == "Darwin":  # macOS
            candidates = [
                "/System/Library/Fonts/Supplemental/Arial.ttf",
                "/Library/Fonts/Arial.ttf",
                os.path.expanduser("~/Library/Fonts/Arial.ttf"),
            ]
        elif sys_plat == "Windows":
            candidates = [
                os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "arial.ttf"),
                os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "Arial.ttf"),
            ]
        else:  # Linux and others
            candidates = [
                "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
                "/usr/share/fonts/truetype/msttcorefonts/Arial/Arial.ttf",
                "/usr/share/fonts/truetype/arial.ttf",
                "/usr/share/fonts/TTF/Arial.ttf",
                "/usr/share/fonts/TTF/arial.ttf",
                os.path.expanduser("~/.fonts/Arial.ttf"),
                os.path.expanduser("~/.fonts/arial.ttf"),
            ]
        for path in candidates:
            if os.path.isfile(path):
                print(f"Found Arial font at: {path}")
                return path
        raise RuntimeError("Arial.ttf not found in standard font locations. Please install Arial or edit the script to use another font.")

    font_path = find_arial_font()
    font = ImageFont.truetype(font_path, fixed_font_size)

    for d in detections:
        box = d.get('bbox')
        if not box:
            continue
        x1, y1, x2, y2 = box
        # Scale coords if detection included det_w/det_h
        if 'det_w' in d and 'det_h' in d:
            det_w, det_h = d['det_w'], d['det_h']
            scale_x = img.width / det_w if det_w else 1.0
            scale_y = img.height / det_h if det_h else 1.0
            x1 *= scale_x; x2 *= scale_x; y1 *= scale_y; y2 *= scale_y

        # Draw fixed-thickness rectangle
        draw.rectangle((x1, y1, x2, y2), outline=(255, 165, 0), width=stroke)

        label = f"{d.get('class','?')} {d.get('confidence',0.0):.2f}"
        # Measure text bbox tightly
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w, text_h = font.getsize(label) if hasattr(font, 'getsize') else (len(label)*8, 16)

        # Draw background exactly the text bbox at (x1, y1)
        tx1 = int(x1)
        ty1 = int(y1)
        tx2 = int(tx1 + text_w)
        ty2 = int(ty1 + text_h)
        draw.rectangle((tx1, ty1, tx2, ty2), fill=(255, 165, 0))
        # Draw text at tx1,ty1
        draw.text((tx1, ty1), label, fill='black', font=font)

    return out


def visualize(image_path: str, confs=None, areas=None, out_dir: str = None, dpi: int = 150, grid_scale: float = 1.0, box_width: int = 8, font_scale: float = 1.0):
    if confs is None:
        confs = [0.6, 0.7, 0.8, 0.9]
    if areas is None:
        areas = [0.01, 1e-3, 5e-4, 1e-4]

    # If we will save to files, ensure a non-interactive backend
    if out_dir:
        try:
            import matplotlib
            matplotlib.use('Agg')
        except Exception:
            pass

    img = Image.open(image_path).convert('RGB')

    counts = []
    overlays = []
    params = []
    # Determine image scaling for overlays so fonts/strokes scale with output size
    img_scale = float(grid_scale) * (float(dpi) / 150.0)
    for c in confs:
        row = []
        row_overlays = []
        for a in areas:
            dets = detect_objects(image_path, confidence_threshold=c, min_area_ratio=a)
            row.append(len(dets))
            row_overlays.append(draw_boxes_on_image(img, dets, box_width=box_width, font_scale=font_scale, img_scale=img_scale))
            params.append((c, a))
        counts.append(row)
        overlays.append(row_overlays)

    # Plot heatmap of counts
    fig, ax = plt.subplots(figsize=(8 * grid_scale, 6 * grid_scale), dpi=dpi)
    im = ax.imshow(counts, cmap='viridis')
    ax.set_xticks(range(len(areas)))
    ax.set_xticklabels([f"{x:.0e}" for x in areas])
    ax.set_yticks(range(len(confs)))
    ax.set_yticklabels([f"{x:.2f}" for x in confs])
    ax.set_xlabel('min_area_ratio')
    ax.set_ylabel('confidence_threshold')
    ax.set_title('Number of detections kept')
    for i in range(len(confs)):
        for j in range(len(areas)):
            ax.text(j, i, str(counts[i][j]), ha='center', va='center', color='white')
    fig.colorbar(im, ax=ax)

    # Show all combinations of overlay thumbnails in a grid where rows
    # correspond to confidence thresholds and columns to area thresholds.
    rows = len(confs)
    cols = len(areas)
    fig2 = plt.figure(figsize=(max(6, cols * 3) * grid_scale, max(3, rows * 2.5) * grid_scale), dpi=dpi)
    for r in range(rows):
        for c in range(cols):
            ax2 = fig2.add_subplot(rows, cols, r * cols + c + 1)
            ax2.imshow(overlays[r][c])
            ax2.set_title(f"conf={confs[r]:.2f}, area={areas[c]:.0e}", fontsize=8)
            # Keep axes frame but remove tick marks for a cleaner look
            ax2.set_xticks([])
            ax2.set_yticks([])
            # Label left-most column with confidence (row) and bottom row with area (column)
            if c == 0:
                ax2.set_ylabel(f"conf={confs[r]:.2f}", fontsize=8)
            if r == rows - 1:
                ax2.set_xlabel(f"area={areas[c]:.0e}", fontsize=8)
    fig2.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base = Path(image_path).stem
        heatmap_path = Path(out_dir) / f"{base}.heatmap.png"
        grid_path = Path(out_dir) / f"{base}.grid.png"
        try:
            fig.savefig(str(heatmap_path), bbox_inches='tight', dpi=dpi)
            fig2.savefig(str(grid_path), bbox_inches='tight', dpi=dpi)
            print(f"Saved heatmap -> {heatmap_path}")
            print(f"Saved grid -> {grid_path}")
        except Exception as exc:
            print(f"Failed to save figures: {exc}")
    else:
        plt.show()


def save_single_overlay(image_path: str, out_path: str, scale: float = 2.0, box_width: int = 12, font_scale: float = 2.0):
    """Render a single upscaled overlay and save it. This avoids building huge Matplotlib
    figures when you only need a large overlay of the source image.
    """
    img = Image.open(image_path).convert('RGB')
    # Compute an img_scale to pass to draw_boxes_on_image so fonts scale
    img_scale = float(scale)
    # Use detect_objects once at the discovered default confidence/area to get boxes
    dets = detect_objects(image_path, confidence_threshold=0.6, min_area_ratio=0.01)
    out = draw_boxes_on_image(img, dets, box_width=box_width, font_scale=font_scale, img_scale=img_scale)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)
    print(f"Saved single overlay -> {out_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize detection thresholds')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--out-dir', '-o', help='Directory to save figures (PNG). If omitted, displays interactively.', default=None)
    parser.add_argument('--dpi', type=int, default=150, help='DPI for saved figures')
    parser.add_argument('--grid-scale', type=float, default=1.0, help='Scale factor for figure sizes')
    parser.add_argument('--box-width', type=int, default=8, help='Bounding box stroke width in pixels')
    parser.add_argument('--font-scale', type=float, default=1.0, help='Scale factor for label font size')
    parser.add_argument('--single-overlay', '-s', dest='single_overlay', help='Path to save a single upscaled overlay PNG (memory-friendly)')
    parser.add_argument('--overlay-scale', type=float, default=2.0, help='Upscale factor for single overlay')
    parser.add_argument('--overlay-box-width', type=int, default=12, help='Box width for single overlay')
    parser.add_argument('--overlay-font-scale', type=float, default=2.0, help='Font scale for single overlay')
    args = parser.parse_args()
    
    if args.single_overlay:
        save_single_overlay(args.image, args.single_overlay, scale=args.overlay_scale, box_width=args.overlay_box_width, font_scale=args.overlay_font_scale)
    else:
        visualize(args.image, out_dir=args.out_dir, dpi=args.dpi, grid_scale=args.grid_scale, box_width=args.box_width, font_scale=args.font_scale)
