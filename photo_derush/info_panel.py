import logging

from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget
from .utils import extract_exif, format_gps_info

class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("color: #aaa; background: #222; font-size: 13pt; border: none; padding: 8px;")
        self.text_edit.setTextInteractionFlags(
            self.text_edit.textInteractionFlags() | self.text_edit.textInteractionFlags().TextSelectableByMouse | self.text_edit.textInteractionFlags().TextSelectableByKeyboard
        )
        layout = QVBoxLayout(self)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)

    def update_info(self, img_name, img_path, group_idx, group_hash, image_hash, metrics=None, keep_prob=None, explanations=None, **kwargs):
        """Update info panel with EXIF + metrics + probability.
        Added optional 'explanations' list for model rationale without breaking existing callers.
        Extra **kwargs ignored for forward compatibility.
        """
        exif = extract_exif(img_path)
        logging.info(f"[InfoPanel] EXIF {img_name}: {list(exif.keys())}")
        # Ordered list of preferred primary fields (emojis optional)
        primary_fields = [
            ("DateTimeOriginal", "📅"),
            ("DateTime", "🕒"),
            ("Make", "🏭"),
            ("Model", "📷"),
            ("LensMake", "🔍"),
            ("LensModel", "🔭"),
            ("Software", "💾"),
            ("ExposureTime", "⏱️"),
            ("ShutterSpeedValue", "⚙️"),
            ("FNumber", "🔆"),
            ("ApertureValue", "🔆"),
            ("ISOSpeedRatings", "🌡️"),
            ("ExposureBiasValue", "➕➖"),
            ("BrightnessValue", "💡"),
            ("MeteringMode", "📐"),
            ("WhiteBalance", "⚪"),
            ("Flash", "⚡"),
            ("FocalLength", "🔎"),
            ("FocalLengthIn35mmFilm", "📏"),
            ("SubjectDistance", "📏"),
            ("DigitalZoomRatio", "🔍"),
            ("SceneCaptureType", "🎞️"),
            ("Contrast", "🌓"),
            ("Saturation", "🌈"),
            ("Sharpness", "🔪"),
            ("CompositeImage", "🧩"),
            ("GPSInfo", "📍"),
            ("ExifImageWidth", "↔️"),
            ("ExifImageHeight", "↕️"),
        ]
        # Decode GPS first (if dict)
        rows_primary = []
        shown = set()
        if 'GPSInfo' in exif and isinstance(exif['GPSInfo'], dict):
            try:
                gps_val = format_gps_info(exif['GPSInfo'])
            except Exception as e:  # noqa: PERF203
                gps_val = f"[Invalid GPSInfo: {e}]"
            rows_primary.append(("📍 GPSInfo", gps_val))
            shown.add('GPSInfo')
        def _fmt(v):
            try:
                from fractions import Fraction
                if isinstance(v, bytes):
                    try: return v.decode('utf-8','ignore')
                    except Exception: return repr(v[:16])
                if isinstance(v, Fraction):
                    return f"{float(v):.4g}"
                if isinstance(v, (list, tuple)):
                    if len(v) <= 6:
                        return '[' + ', '.join(_fmt(x) for x in v) + ']'
                    return '[' + ', '.join(_fmt(x) for x in v[:6]) + ', …]'
                if isinstance(v, dict):
                    # Compact dict preview
                    items = []
                    for k2, v2 in list(v.items())[:8]:
                        items.append(f"{k2}:{_fmt(v2)}")
                    more = ' …' if len(v) > 8 else ''
                    return '{' + ', '.join(items) + more + '}'
                return v
            except Exception:
                return v
        for key, emoji in primary_fields:
            if key == 'GPSInfo':
                continue
            if key in exif and key not in shown:
                rows_primary.append((f"{emoji} {key}", _fmt(exif[key])))
                shown.add(key)
        # Derived image size
        if 'ExifImageWidth' in exif and 'ExifImageHeight' in exif:
            rows_primary.append(("🖼️ Image Size", f"{exif['ExifImageWidth']} x {exif['ExifImageHeight']}"))
        # Remaining EXIF
        other_rows = []
        for k in sorted(exif.keys()):
            if k in shown:
                continue
            val = exif[k]
            if k == 'GPSInfo' and isinstance(val, dict):
                continue
            other_rows.append((k, _fmt(val)))
        def _table(rows, small=False):
            if not rows:
                return "<i style='color:#666;'>None</i>"
            fs = '10pt' if small else '11pt'
            html = f"<table style='font-size:{fs}; color:#bbb; background:#232629; border-collapse:collapse;'>"
            for k,v in rows:
                html += ("<tr><td style='font-weight:bold; padding:2px 12px 2px 0; vertical-align:top;'>" +
                         f"{k}</td><td style='padding:2px 0; word-break:break-word;'>" +
                         f"{v}</td></tr>")
            html += '</table>'
            return html
        primary_html = _table(rows_primary)
        other_html = _table(other_rows, small=True)
        # Metrics
        metrics_html = "<div style='margin-bottom:10px;'><b>Metrics</b><br>"
        if metrics:
            blur_score, sharpness_metrics, aesthetic_score = metrics
            lines = [
                f"Blur: <b>{blur_score:.1f}</b>" if blur_score is not None else "Blur: N/A",
                f"Laplacian: <b>{sharpness_metrics['variance_laplacian']:.1f}</b>" if sharpness_metrics else "Laplacian: N/A",
                f"Tenengrad: <b>{sharpness_metrics['tenengrad']:.1f}</b>" if sharpness_metrics else "Tenengrad: N/A",
                f"Brenner: <b>{sharpness_metrics['brenner']:.1f}</b>" if sharpness_metrics else "Brenner: N/A",
                f"Wavelet: <b>{sharpness_metrics['wavelet_energy']:.1f}</b>" if sharpness_metrics else "Wavelet: N/A",
                f"Aesthetic: <b>{aesthetic_score:.2f}</b>" if aesthetic_score is not None else "Aesthetic: N/A",
            ]
            metrics_html += '<br>'.join(lines)
        else:
            metrics_html += 'No metrics available.'
        metrics_html += '</div>'
        # Explanations (model rationale quick view)
        expl_html = ''
        if explanations:
            chips = []
            for txt, good in explanations:
                color = '#2e7d32' if good else '#b71c1c'
                chips.append(f"<span style='background:{color}; color:#fff; padding:2px 6px; border-radius:4px; margin:2px; font-size:11pt;'>{txt}</span>")
            expl_html = "<div style='margin:6px 0 12px 0;'><b>Explain:</b><br>" + ' '.join(chips) + '</div>'
        # File info & probability
        file_info = (f"<div style='margin-bottom:10px;'><b>File:</b> {img_name}<br>"
                     f"<b>Path:</b> {img_path}<br>"
                     f"<b>Group ID:</b> {group_idx}<br>"
                     f"<b>Group Hash:</b> {group_hash}<br>"
                     f"<b>Image Hash:</b> {image_hash}</div>")
        prob_html = (f"<div style='margin-bottom:10px; font-size:16pt; color:#4caf50;'><b>Keep Probability:</b> {keep_prob:.2%}</div>" if keep_prob is not None else '')
        exif_section = ("<div style='margin-top:10px;'><b>EXIF (Primary)</b><br>" + primary_html +
                        "<div style='margin-top:12px;'><b>All Other EXIF</b><br>" + other_html + '</div></div>')
        html = prob_html + file_info + expl_html + metrics_html + exif_section
        self.text_edit.setHtml(html)
