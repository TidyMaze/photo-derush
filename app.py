import argparse
import json
import logging
import os
import signal
import sys
import time

# Early mitigation: set conservative native-thread env vars and enable faulthandler
# Centralized in `src.init_mitigation` so other entry scripts can reuse it.
import src.init_mitigation  # noqa: F401 - module side-effects only

try:
    import qdarktheme
except ImportError:
    qdarktheme = None
from PySide6.QtCore import QTimer  # added for SIGINT heartbeat
from PySide6.QtWidgets import QApplication

# Configure logging BEFORE importing modules that emit logs at import time
_LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
# Only call basicConfig if no handlers are configured (prevents duplicate handlers)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=getattr(logging, _LOG_LEVEL, logging.INFO), format='%(levelname)s:%(name)s:%(message)s')

from src.features import safe_initialize_feature_cache
from src.view import PhotoView
from src.viewmodel import PhotoViewModel

# Prevent multiprocessing from using 'spawn' on macOS by forcing 'fork'
try:
    import multiprocessing as _mp
    _mp.set_start_method('fork', force=True)
except (RuntimeError, ValueError):
    # Already set or not available on this platform
    pass

# Reduce verbosity from PIL image plugins
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('PIL.TiffImagePlugin').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
logging.getLogger('PIL.JpegImagePlugin').setLevel(logging.WARNING)

CONFIG_PATH = os.path.expanduser('~/.photo_app_config.json')
MAX_IMAGES = None  # Unlimited images (was 20, removed limit)

def load_last_dir():
    try:
        with open(CONFIG_PATH) as f:
            data = json.load(f)
            last_dir = data.get('last_dir')
            if last_dir and os.path.isdir(last_dir):
                return last_dir
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return os.path.expanduser('~')

def save_last_dir(path):
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump({'last_dir': path}, f)
    except Exception as e:
        logging.warning("Could not save config: %s", e)

def main():
    # Enable profiling if PROFILING env var is set
    profiler = None
    if os.environ.get("PROFILING") == "1":
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        logging.info("[PROFILING] Enabled cProfile")
        
        # Profile dump will be set up after QApplication is created
    try:
        # Parse CLI args early so logging can be configured to file if requested
        parser = argparse.ArgumentParser(description='Photo Derush - Qt desktop application')
        parser.add_argument('--log-file', '-l', help='Path to write log output (appends)', default=None)
        args, _ = parser.parse_known_args()

        if args.log_file:
            try:
                fh = logging.FileHandler(args.log_file, mode='a')
                fh.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
                fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s'))
                logging.getLogger().addHandler(fh)
                logging.info(f"Logging also written to file: {args.log_file}")
            except OSError as e:
                logging.warning(f"Could not open log file {args.log_file}: {e}")

        logging.info("App main() starting...")
        # Set detection worker env early to prefer in-process detection during debugging
        os.environ.setdefault('DETECTION_WORKER', '0')

        # Create QApplication early so Qt event loop and signal handling are available
        app = QApplication(sys.argv)

        # Initialize feature cache after QApplication to avoid heavy work before GUI ready
        safe_initialize_feature_cache()

        # Install SIGINT and SIGTERM handlers for clean shutdown
        def _handle_signal(signum, frame):
            logging.info(f"Signal {signum} received, quitting application...")
            # Trigger cleanup before quitting
            try:
                if hasattr(app, 'viewmodel'):
                    app.viewmodel.cleanup()
            except Exception as e:
                logging.warning(f"Error during cleanup: {e}")
            app.quit()
            # Force exit after cleanup timeout
            QTimer.singleShot(2000, lambda: (logging.warning("Force exit after cleanup timeout"), os._exit(0)))
        
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # Heartbeat timer to allow Python signal handling while Qt blocks
        timer = QTimer()
        timer.setInterval(200)
        timer.timeout.connect(lambda: None)
        timer.start()
        
        # Setup profiling dump timer if profiling is enabled (after QApplication is created)
        if profiler:
            def dump_profile():
                if profiler:
                    try:
                        profiler.dump_stats("/tmp/app_profile.prof")
                        logging.info("[PROFILING] Dumped stats to /tmp/app_profile.prof")
                    except Exception as e:
                        logging.warning(f"[PROFILING] Failed to dump stats: {e}")
            
            profile_timer = QTimer()
            profile_timer.timeout.connect(dump_profile)
            profile_timer.start(30000)  # Every 30 seconds
            logging.info("[PROFILING] Profile dump timer started (30s interval)")

        if qdarktheme is not None:
            try:
                qdarktheme.setup_theme()
            except Exception:
                logging.exception('Failed to setup qdarktheme')
        else:
            logging.info('qdarktheme not available; skipping theme setup')
        last_dir = load_last_dir()
        logging.info(f"Loaded last_dir from config: {last_dir}")
        # PhotoViewModel expects an int; 0 means unlimited. If MAX_IMAGES is None, pass 0.
        max_images = 0 if MAX_IMAGES is None else MAX_IMAGES
        viewmodel = PhotoViewModel(last_dir, max_images=max_images)
        logging.info("PhotoViewModel created, about to create PhotoView and call load_images()")
        view = PhotoView(viewmodel, thumb_size=800)

        # Keep strong references to prevent premature garbage collection
        app.view = view  # type: ignore
        app.viewmodel = viewmodel  # type: ignore

        # Connect cleanup to app exit signal
        app.aboutToQuit.connect(viewmodel.cleanup)
        view.show()
        
        # macOS-specific: ensure app is active and window is on top
        try:
            from PySide6.QtGui import QGuiApplication
            from PySide6.QtCore import Qt
            QGuiApplication.setApplicationDisplayName("Photo Derush")
            # Force window to front on macOS
            current_state = view.windowState()
            view.setWindowState(current_state & ~Qt.WindowState.WindowMinimized | Qt.WindowState.WindowActive)
        except Exception as e:
            logging.debug(f"Window state setup failed: {e}")
        
        view.raise_()
        view.activateWindow()
        
        # Additional macOS activation: use QTimer to ensure window appears after event loop starts
        def ensure_window_visible():
            view.show()
            view.raise_()
            view.activateWindow()
        QTimer.singleShot(100, ensure_window_visible)
        QTimer.singleShot(0, viewmodel.load_images)
        
        # Optional: allow automatic quit after a short duration for profiling
        try:
            auto_ms = int(os.environ.get('PHOTO_DERUSH_AUTOQUIT_MS', '0'))
            if auto_ms > 0:
                QTimer.singleShot(auto_ms, lambda: (logging.info('Auto-quit timer fired'), app.quit()))
        except (ValueError, TypeError):
            pass
        exit_code = app.exec()
        
        # Save final profile on exit if profiling was enabled
        if profiler:
            try:
                profiler.disable()
                profiler.dump_stats("/tmp/app_profile.prof")
                logging.info(f"[PROFILING] Final profile saved to /tmp/app_profile.prof")
            except Exception as e:
                logging.warning(f"[PROFILING] Failed to save final profile: {e}")
        logging.info("Qt event loop exited. Performing final cleanup.")
        sys.exit(exit_code)
    except Exception as e:
        logging.exception("Fatal error in main()")
        raise

if __name__ == "__main__":
    main()
