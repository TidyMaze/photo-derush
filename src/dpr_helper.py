import os
from PySide6.QtGui import QGuiApplication

def get_dpr(screen=None) -> float:
    """Get the device pixel ratio, supporting FORCE_DPR environment override."""
    env_dpr = os.environ.get("FORCE_DPR")
    if env_dpr:
        try:
            return float(env_dpr)
        except ValueError:
            pass
    if screen is None:
        try:
            screen = QGuiApplication.primaryScreen()
        except Exception:
            pass
    if screen:
        try:
            return float(screen.devicePixelRatio() or 1.0)
        except Exception:
            pass
    return 1.0
