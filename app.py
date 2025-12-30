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
from src.timing import log_stats, reset_stats
from src.view import PhotoView
from src.viewmodel import PhotoViewModel

# Set multiprocessing start method based on platform
# Windows requires 'spawn', Linux/Mac can use 'fork' (faster) or 'spawn' (safer)
try:
    import multiprocessing as _mp
    import platform
    if platform.system() == "Windows":
        # Windows only supports 'spawn'
        _mp.set_start_method('spawn', force=True)
    else:
        # Linux/Mac: prefer 'fork' for performance, but allow 'spawn' if fork fails
        try:
            _mp.set_start_method('fork', force=True)
        except (RuntimeError, ValueError):
            # Fallback to spawn if fork is not available
            _mp.set_start_method('spawn', force=True)
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
    # Non-intrusive profiling: Use external profilers (py-spy, scalene) that attach to process
    # No code instrumentation needed - just run the app and attach profiler externally
    # 
    # Usage examples:
    #   py-spy:   py-spy record -o /tmp/profile.json --pid $(pgrep -f "python.*app.py") --subprocesses
    #   scalene:  scalene --profile-all --outfile /tmp/profile.html --pid $(pgrep -f "python.*app.py")
    #
    # For internal profiling (if PROFILING=1), enable memory profiling (tracemalloc) and CPU profiling (cProfile)
    # Note: cProfile has higher overhead but works without root permissions
    memory_tracer = None
    cpu_profiler = None
    if os.environ.get("PROFILING") == "1":
        # Enable memory profiling (lightweight, non-intrusive)
        try:
            import tracemalloc
            tracemalloc.start()
            memory_tracer = tracemalloc
            logging.info("[PROFILING] Enabled tracemalloc for memory profiling (non-intrusive)")
        except Exception as e:
            logging.warning(f"[PROFILING] Failed to enable tracemalloc: {e}")
        
        # Enable CPU profiling (higher overhead, but works without root)
        try:
            import cProfile
            cpu_profiler = cProfile.Profile()
            cpu_profiler.enable()
            logging.info("[PROFILING] Enabled cProfile for CPU profiling (main thread only)")
        except Exception as e:
            logging.warning(f"[PROFILING] Failed to enable cProfile: {e}")
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

        # Create QApplication early so Qt event loop and signal handling are available
        app = QApplication(sys.argv)

        # Initialize feature cache after QApplication to avoid heavy work before GUI ready
        safe_initialize_feature_cache()
        
        # OPTIMIZATION: Pre-load YOLOv8 model in background thread to avoid 204s delay on first detection
        def preload_model():
            try:
                from src.object_detection import _load_model
                logging.info("[OPTIMIZATION] Pre-loading YOLOv8 model in background...")
                _load_model("auto")
                logging.info("[OPTIMIZATION] YOLOv8 model pre-loaded successfully")
            except Exception as e:
                logging.warning(f"[OPTIMIZATION] Failed to pre-load YOLOv8 model: {e}")
        
        # Pre-load model in background thread (non-blocking)
        import threading
        model_preload_thread = threading.Thread(target=preload_model, daemon=True)
        model_preload_thread.start()

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
        
        # Log PID for external profiler attachment (non-intrusive)
        if memory_tracer:
            pid = os.getpid()
            logging.info(f"[PROFILING] App PID: {pid} - Attach external profiler with:")
            logging.info(f"[PROFILING]   py-spy (CPU): py-spy record -o /tmp/app_profile_pyspy.json --pid {pid} --subprocesses --threads --rate 100")
            logging.info(f"[PROFILING]   Memory: tracemalloc enabled (snapshots in /tmp/app_memory_*.txt)")
            logging.info(f"[PROFILING]   Or use: ./scripts/profile_app.sh py-spy 60")
        
        # Setup memory profiling dump timer if enabled
        if memory_tracer:
            def dump_profile():
                try:
                    snapshot = memory_tracer.take_snapshot()
                    top_stats = snapshot.statistics('lineno')
                    
                    with open("/tmp/app_memory_snapshot.txt", "w") as f:
                        f.write("===== MEMORY SNAPSHOT =====\n")
                        f.write(f"Total allocated: {sum(stat.size for stat in top_stats) / 1024 / 1024:.2f} MB\n\n")
                        f.write("Top 50 memory allocations:\n")
                        for index, stat in enumerate(top_stats[:50], 1):
                            f.write(f"{index}. {stat}\n")
                    
                    snapshot.dump("/tmp/app_memory_snapshot.pkl")
                    logging.info("[PROFILING] Dumped memory snapshot to /tmp/app_memory_snapshot.txt")
                except Exception as e:
                    logging.warning(f"[PROFILING] Failed to dump memory stats: {e}")
            
            profile_timer = QTimer()
            profile_timer.timeout.connect(dump_profile)
            profile_timer.start(30000)  # Every 30 seconds
            logging.info("[PROFILING] Memory profile dump timer started (30s interval)")

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
        
        # Save final memory profile on exit if profiling was enabled
        # Note: External CPU profilers (py-spy, scalene) handle their own cleanup
        
        if memory_tracer:
            try:
                snapshot = memory_tracer.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                
                with open("/tmp/app_memory_final.txt", "w") as f:
                    f.write("===== FINAL MEMORY SNAPSHOT =====\n")
                    total_mb = sum(stat.size for stat in top_stats) / 1024 / 1024
                    f.write(f"Total allocated: {total_mb:.2f} MB\n\n")
                    f.write("Top 100 memory allocations:\n")
                    for index, stat in enumerate(top_stats[:100], 1):
                        f.write(f"{index}. {stat}\n")
                
                snapshot.dump("/tmp/app_memory_final.pkl")
                logging.info(f"[PROFILING] Final memory snapshot saved ({total_mb:.2f} MB total)")
            except Exception as e:
                logging.warning(f"[PROFILING] Failed to save final memory snapshot: {e}")
        
        # Log timing statistics
        log_stats()
        
        # Save CPU profile if enabled
        if cpu_profiler:
            try:
                profile_path = "/tmp/app_profile.prof"
                cpu_profiler.disable()
                cpu_profiler.dump_stats(profile_path)
                logging.info(f"[PROFILING] CPU profile saved to {profile_path}")
            except Exception as e:
                logging.warning(f"[PROFILING] Failed to save CPU profile: {e}")
        
        logging.info("Qt event loop exited. Performing final cleanup.")
        sys.exit(exit_code)
    except Exception as e:
        logging.exception("Fatal error in main()")
        raise

if __name__ == "__main__":
    main()
