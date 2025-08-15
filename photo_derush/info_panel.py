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

    def update_info(self, img_name, img_path, group_idx, group_hash, image_hash, metrics=None):
        exif = extract_exif(img_path)
        # Define important EXIF fields in order
        important_fields = [
            "DateTimeOriginal", "DateTime", "Make", "Model", "LensMake", "LensModel", "Software",
            "ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength", "FocalLengthIn35mmFilm",
            "ExposureBiasValue", "MeteringMode", "WhiteBalance", "Flash", "GPSInfo",
            "ExifImageWidth", "ExifImageHeight"
        ]
        # Prepare important and other EXIF fields
        exif_display = []
        other_exif = []
        exif_keys = set(exif.keys())
        # GPSInfo first if present
        if "GPSInfo" in exif and isinstance(exif["GPSInfo"], dict):
            try:
                gps_str = format_gps_info(exif["GPSInfo"])
            except Exception as e:
                gps_str = f"[Invalid GPSInfo: {e}]"
            exif_display.append(("GPSInfo", gps_str))
        # Add other important fields in order
        for key in important_fields:
            if key == "GPSInfo":
                continue  # already handled
            if key in exif:
                exif_display.append((key, exif[key]))
        # Add image size as WxH if both present
        width = exif.get("ExifImageWidth")
        height = exif.get("ExifImageHeight")
        if width and height:
            exif_display.append(("Image Size", f"{width} x {height}"))
        # Collect other EXIF fields
        shown_keys = {k for k, _ in exif_display}
        for k in sorted(exif_keys - shown_keys):
            v = exif[k]
            if k == "GPSInfo" and isinstance(v, dict):
                continue  # already shown
            other_exif.append((k, v))
        # Render important EXIF fields as a table
        exif_table = "<table style='font-size:11pt; color:#bbb;'>"
        for k, v in exif_display:
            exif_table += f"<tr><td style='font-weight:bold; padding-right:10px;'>{k}</td><td>{v}</td></tr>"
        exif_table += "</table>"
        # Render other EXIF fields in a <details> block
        if other_exif:
            more_table = "<table style='font-size:10pt; color:#aaa;'>"
            for k, v in other_exif:
                more_table += f"<tr><td style='font-weight:bold; padding-right:10px;'>{k}</td><td>{v}</td></tr>"
            more_table += "</table>"
            exif_more = f"<details style='margin-top:8px;'><summary style='cursor:pointer; color:#88f;'>See more</summary>{more_table}</details>"
        else:
            exif_more = ""
        # Metrics section
        metrics_str = "<div style='margin-bottom:10px;'><b>Metrics</b><br>"
        if metrics:
            blur_score, sharpness_metrics, aesthetic_score = metrics
            lines = [
                f"Blur: <b>{blur_score:.1f}</b>" if blur_score is not None else "Blur: N/A",
                f"Laplacian: <b>{sharpness_metrics['variance_laplacian']:.1f}</b>" if sharpness_metrics else "Laplacian: N/A",
                f"Tenengrad: <b>{sharpness_metrics['tenengrad']:.1f}</b>" if sharpness_metrics else "Tenengrad: N/A",
                f"Brenner: <b>{sharpness_metrics['brenner']:.1f}</b>" if sharpness_metrics else "Brenner: N/A",
                f"Wavelet: <b>{sharpness_metrics['wavelet_energy']:.1f}</b>" if sharpness_metrics else "Wavelet: N/A",
                f"Aesthetic: <b>{aesthetic_score:.2f}</b>" if aesthetic_score is not None else "Aesthetic: N/A"
            ]
            metrics_str += "<br>".join(lines)
        else:
            metrics_str += "No metrics available."
        metrics_str += "</div>"
        # File info section
        file_info = f"<div style='margin-bottom:10px;'><b>File:</b> {img_name}<br>"
        file_info += f"<b>Path:</b> {img_path}<br>"
        file_info += f"<b>Group ID:</b> {group_idx}<br>"
        file_info += f"<b>Group Hash:</b> {group_hash}<br>"
        file_info += f"<b>Image Hash:</b> {image_hash}</div>"
        # EXIF section
        exif_section = f"<div style='margin-top:10px;'><b>EXIF</b><br>{exif_table}{exif_more}</div>"
        # Combine all sections
        html = file_info + metrics_str + exif_section
        self.text_edit.setHtml(html)
