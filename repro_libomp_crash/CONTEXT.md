Context and purpose

This minimal repro folder exists to provide a focused, self-contained
reproducer for the native crash observed while running YOLOv8 in the
photo-derush app. The goal is to remove GUI, Qt, multiprocessing workers,
and other app-specific layers so an engineer or upstream PyTorch developer
can quickly reproduce and debug the EXC_BAD_ACCESS in `libomp.dylib`.

Summary of original problem:
- The app runs YOLOv8 detections via a spawn-based worker process.
- Under load the detection worker sometimes times out and the parent app
  later crashes with `EXC_BAD_ACCESS` in `libomp.dylib` while libtorch
  CPU kernels were executing (native backtraces captured in
  `logs/lldb-app-final.txt`).
- This repro aims to trigger the same native failure with only PyTorch
  CPU work to make the problem actionable for upstream.

If this simple PyTorch-based harness does not reproduce the crash,
copy `run_yolov8_direct.py` from the project into this folder and run
`omp_pytorch_repro.py --try-yolo` to exercise the YOLOv8 path.
