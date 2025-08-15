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
        exif_lines = []
        for k, v in exif.items():
            if k == "GPSInfo" and isinstance(v, dict):
                try:
                    gps_str = format_gps_info(v)
                except Exception as e:
                    gps_str = f"[Invalid GPSInfo: {e}]"
                exif_lines.append(f"<b>GPSInfo:</b> {gps_str}")
            else:
                exif_lines.append(f"<b>{k}:</b> {v}")
        exif_str = "<br>".join(exif_lines) if exif_lines else "No EXIF data"
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
        exif_section = f"<div style='margin-top:10px;'><b>EXIF</b><br><div style='font-family:monospace; font-size:11pt; background:#222; color:#bbb; padding:6px 0;'>{exif_str}</div></div>"
        # Combine all sections
        html = file_info + metrics_str + exif_section
        self.text_edit.setHtml(html)
