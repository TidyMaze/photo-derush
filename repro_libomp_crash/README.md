Minimal repro for libomp/libtorch EXC_BAD_ACCESS

This folder contains a small harness to reproduce and capture the native crash
that appears to be happening in `libomp.dylib` when PyTorch (libtorch) runs
CPU kernels under concurrent load on macOS.

Files:
- `omp_pytorch_repro.py`: Main Python harness. Runs light-weight tensor
  operations in multiple threads to stress the OpenMP-backed kernels.
- `run_yolov8_direct.py` (optional): If the pure-PyTorch harness doesn't
  reproduce the crash, copy your existing `run_yolov8_direct.py` (from the
  main repository) into this folder and run with `--try-yolo` to use it.
- `CONTEXT.md`: Short explanation of why this harness exists.

Recommended environment (macOS, arm64):
- Python 3.12 virtualenv (use the project's `.venv` if convenient)
- PyTorch matching the version you used in the app (document the version)
- Keep `yolov8` installed only if necessary for the YOLO fallback run

Example runs:

# Activate virtualenv (adjust path if needed)
source .venv/bin/activate

# Plain run
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export KMP_BLOCKTIME=0
export KMP_AFFINITY=compact
python omp_pytorch_repro.py --iterations 200 --threads 4 --batch-size 512

# Run under lldb with dynamic library printout and full thread backtraces
# (produces a large logs file `logs/lldb-omp-repro.txt` on crash)
mkdir -p logs
DYLD_PRINT_LIBRARIES=1 \
  lldb -o "run" -o "thread backtrace all" -o "quit" -- \
  python omp_pytorch_repro.py --iterations 10000 --threads 4 --batch-size 512 > logs/lldb-omp-repro.txt 2>&1

Collect `otool` outputs for the key dylibs (adjust paths to your venv):

otool -L .venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.dylib > logs/otool-libtorch_cpu.txt
otool -L .venv/lib/python3.12/site-packages/torch/lib/libtorch_python.dylib > logs/otool-libtorch_python.txt

# If you know which libomp is loaded (see DYLD_PRINT_LIBRARIES output),
# capture its otool output as well:
# otool -L /path/to/libomp.dylib > logs/otool-libomp.txt

What to include when filing an upstream bug or asking for help:
- `logs/lldb-omp-repro.txt` with the full `thread backtrace all` output
- `logs/otool-libtorch_cpu.txt` and `logs/otool-libtorch_python.txt`
- The exact env vars you used and the PyTorch/torch-nightly wheel version
- A short note stating whether the minimal PyTorch harness reproduces the
  crash or if you needed the YOLO fallback
