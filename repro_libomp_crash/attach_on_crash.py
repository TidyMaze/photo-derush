#!/usr/bin/env python3
"""Launch the harness and attach lldb to the process to capture native backtraces on crash.

This script:
- Starts `omp_pytorch_repro.py` as a child process with the same env vars recommended.
- Attaches lldb to the child PID and continues execution under lldb control.
- Writes lldb output to `repro_libomp_crash/logs/attach-lldb-<pid>.txt`.

Use this when the crash is intermittent — attaching lldb increases odds of capturing
EXC_BAD_ACCESS backtraces when they occur.
"""

import os
import subprocess
import time
import sys
from pathlib import Path

LOGDIR = Path(__file__).parent / 'logs'
LOGDIR.mkdir(exist_ok=True)
PYTHON = Path.cwd() / '.venv' / 'bin' / 'python'
HARNESS = Path(__file__).parent / 'omp_pytorch_repro.py'

ENV = os.environ.copy()
ENV.setdefault('OMP_NUM_THREADS', '1')
ENV.setdefault('MKL_NUM_THREADS', '1')
ENV.setdefault('OPENBLAS_NUM_THREADS', '1')
ENV.setdefault('VECLIB_MAXIMUM_THREADS', '1')
ENV.setdefault('KMP_BLOCKTIME', '0')
ENV.setdefault('KMP_AFFINITY', 'compact')

def main():
    args = [str(PYTHON), str(HARNESS), '--iterations', '100000', '--threads', '4', '--batch-size', '512']
    print('Launching harness:', ' '.join(args))
    child = subprocess.Popen(args, env=ENV)
    pid = child.pid
    print('Launched PID', pid)
    # allow process to initialize
    time.sleep(1.0)
    lldb_log = LOGDIR / f'attach-lldb-{pid}.txt'
    lldb_cmd = [
        'lldb', '-p', str(pid),
        '-o', 'process handle -p true -n true -s true SIGSEGV SIGABRT SIGBUS',
        '-o', 'process continue',
        '-o', 'thread backtrace all',
        '-o', 'quit'
    ]
    print('Attaching lldb, output ->', lldb_log)
    with open(lldb_log, 'wb') as out:
        try:
            proc = subprocess.Popen(lldb_cmd, stdout=out, stderr=subprocess.STDOUT)
            # Wait for child to exit (lldb will also block until exit/crash)
            child.wait()
            # Give lldb a moment to write backtrace
            proc.wait(timeout=5)
        except Exception as e:
            print('Error running lldb attach:', e)
            try:
                proc.kill()
            except Exception:
                pass
    print('Done; check', lldb_log)

if __name__ == '__main__':
    main()
