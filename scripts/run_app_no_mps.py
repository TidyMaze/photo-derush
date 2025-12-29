import os
import runpy
import importlib

# Disable MPS via env vars before torch imports
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
os.environ.setdefault("TORCH_USE_MPS", "0")

# If torch is already importable, attempt to monkeypatch its MPS availability
try:
    if importlib.util.find_spec("torch"):
        import torch
        try:
            if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
                torch.backends.mps.is_available = lambda: False
                torch.backends.mps.is_built = lambda: False
        except Exception:
            pass
except Exception:
    pass

if __name__ == "__main__":
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    runpy.run_path(app_path, run_name="__main__")
