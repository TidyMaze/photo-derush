import os
from src.app import FileScanner
from PySide6.QtCore import QEventLoop, QCoreApplication

def test_file_scanner(tmp_path):
    # Create dummy image files
    (tmp_path / "a.jpg").write_bytes(b"1")
    (tmp_path / "b.png").write_bytes(b"2")
    (tmp_path / "c.txt").write_bytes(b"3")
    found = []
    app = QCoreApplication([])
    loop = QEventLoop()
    def on_found(files):
        found.extend(files)
        loop.quit()
    scanner = FileScanner(str(tmp_path))
    scanner.files_found.connect(on_found)
    scanner.start()
    loop.exec()
    assert set(found) == {"a.jpg", "b.png"}
# Simple Photo App

A minimal photo management app. Start simple, add one feature at a time.

## Features
- Select directory
- List image files
- All long tasks off GUI thread

## Setup
- Python 3.10+
- PySide6

## Run
```
python -m src.app
```

