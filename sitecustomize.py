import os
import signal
import functools

# Install a per-test timeout for unittest by monkeypatching TestCase.run.
# This file is imported automatically when running Python from the project root.

DEFAULT_TIMEOUT = int(os.environ.get('TEST_TIMEOUT', '15'))


def _raise_timeout(signum, frame):
    raise TimeoutError(f"Test exceeded timeout of {DEFAULT_TIMEOUT}s")


try:
    import unittest

    if hasattr(unittest, 'TestCase'):
        # Avoid double-patching
        if not getattr(unittest.TestCase, '_timeout_patched', False):
            original_run = unittest.TestCase.run

            def _run_with_timeout(self, result=None):
                # Only set alarm on POSIX-like systems
                if hasattr(signal, 'alarm'):
                    timeout = int(os.environ.get('TEST_TIMEOUT', str(DEFAULT_TIMEOUT)))
                    signal.signal(signal.SIGALRM, _raise_timeout)
                    signal.alarm(timeout)
                try:
                    return original_run(self, result)
                finally:
                    if hasattr(signal, 'alarm'):
                        signal.alarm(0)

            unittest.TestCase.run = _run_with_timeout
            unittest.TestCase._timeout_patched = True
except Exception:
    # Best effort: if unittest not available or patch fails, don't crash startup
    pass
