import os
import signal
import logging
import pytest

# Reduce noisy DEBUG logs emitted under the 'conftest' logger during pytest runs.
# Tests still benefit from INFO+ logs while DEBUG messages from the test harness
# (like QThreadPool wait tracing) are suppressed.
logging.getLogger('conftest').setLevel(logging.INFO)

# Reduce BLAS/OpenMP/MKL thread counts during tests to avoid parallel deadlocks
# Common libs that spawn threads: OpenMP (OMP_NUM_THREADS), OpenBLAS, MKL, NUMEXPR
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

# Default per-test timeout in seconds. Can be overridden with TEST_TIMEOUT env var.
DEFAULT_TIMEOUT = int(os.environ.get('TEST_TIMEOUT', '15'))


def _raise_timeout(signum, frame):
    raise TimeoutError(f"Test exceeded timeout of {DEFAULT_TIMEOUT}s")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # Only set alarm on POSIX-like systems where signal.alarm exists
    if hasattr(signal, 'alarm'):
        timeout = int(os.environ.get('TEST_TIMEOUT', str(DEFAULT_TIMEOUT)))
        # install handler
        signal.signal(signal.SIGALRM, _raise_timeout)
        signal.alarm(timeout)


def pytest_configure(config):
    """Apply env var thread limits early in the pytest life-cycle.

    This helps ensure libraries like XGBoost, OpenBLAS and torch don't spawn
    multiple worker threads which can deadlock the test runner on some
    platforms (macOS with MPS/OpenMP interactions).
    """
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
    os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    # Cancel alarm after test finishes
    if hasattr(signal, 'alarm'):
        signal.alarm(0)
