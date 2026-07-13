import time
import pytest
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication
from src.taskrunner import TaskRunner, ProgressReporter


def test_taskrunner_sequential_execution():
    # Ensure QApplication is initialized (needed for QObject signals)
    _ = QApplication.instance() or QApplication([])

    runner = TaskRunner()
    execution_order = []

    def task1(reporter: ProgressReporter):
        execution_order.append("task1_start")
        time.sleep(0.1)
        execution_order.append("task1_end")

    def task2(reporter: ProgressReporter):
        execution_order.append("task2_start")
        execution_order.append("task2_end")

    runner.run("task1", task1)
    runner.run("task2", task2)

    # Wait for worker thread to complete both tasks
    runner._queue.join()

    assert execution_order == ["task1_start", "task1_end", "task2_start", "task2_end"]
    runner.shutdown()


def test_taskrunner_signals():
    app = QApplication.instance() or QApplication([])

    runner = TaskRunner()
    started = []
    finished = []
    progress = []

    runner.task_started.connect(started.append)
    runner.task_finished.connect(lambda name, ok: finished.append((name, ok)))
    runner.task_progress.connect(lambda name, cur, tot, det: progress.append((name, cur, tot, det)))

    def sample_task(reporter: ProgressReporter):
        reporter.set_total(10)
        reporter.advance(2)
        reporter.detail("processing")

    runner.run("my_task", sample_task)
    runner._queue.join()

    # Process events to deliver cross-thread signals
    app.processEvents()

    assert started == ["my_task"]
    assert finished == [("my_task", True)]
    
    # Wait a brief moment to ensure debounced signals are processed
    time.sleep(0.1)
    app.processEvents()
    
    assert len(progress) > 0
    assert progress[-1][0] == "my_task"
    assert progress[-1][1] == 2
    assert progress[-1][2] == 10

    runner.shutdown()
