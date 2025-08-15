from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget
from .utils import extract_exif, format_gps_info

class InfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("color: #aaa; background: #222; font-size: 14pt;")
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
                exif_lines.append(f"GPSInfo: {gps_str}")
            else:
                exif_lines.append(f"{k}: {v}")
        exif_str = "\n".join(exif_lines) if exif_lines else "No EXIF data"
        metrics_str = ""
        if metrics:
            blur_score, sharpness_metrics, aesthetic_score = metrics
            lines = [
                f"Blur: {blur_score:.1f}" if blur_score is not None else "Blur: N/A",
                f"Laplacian: {sharpness_metrics['variance_laplacian']:.1f}" if sharpness_metrics else "Laplacian: N/A",
                f"Tenengrad: {sharpness_metrics['tenengrad']:.1f}" if sharpness_metrics else "Tenengrad: N/A",
                f"Brenner: {sharpness_metrics['brenner']:.1f}" if sharpness_metrics else "Brenner: N/A",
                f"Wavelet: {sharpness_metrics['wavelet_energy']:.1f}" if sharpness_metrics else "Wavelet: N/A",
                f"Aesthetic: {aesthetic_score:.2f}" if aesthetic_score is not None else "Aesthetic: N/A"
            ]
            metrics_str = "<b>Metrics:</b><br>" + "<br>".join(lines) + "<br>"
        info = f"<b>File:</b> {img_name}<br>"
        info += f"<b>Path:</b> {img_path}<br>"
        info += f"<b>Group ID:</b> {group_idx}<br>"
        info += f"<b>Group Hash:</b> {group_hash}<br>"
        info += f"<b>Image Hash:</b> {image_hash}<br>"
        info += metrics_str
        info += f"<b>EXIF:</b><br><pre style='font-size:10pt'>{exif_str}</pre>"
        self.text_edit.setHtml(info)
