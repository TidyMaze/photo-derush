#!/usr/bin/env python3
"""Small GUI to display cProfile `.prof` files using PySide6.

Usage: poetry run python tools/show_profile.py .cache/profile_60s_after_fix.prof
"""
import sys
from pathlib import Path
import pstats

from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QPushButton, QFileDialog, QHBoxLayout
from PySide6.QtGui import QFont


def format_stats(prof_path: Path, lines: int = None) -> str:
    p = pstats.Stats(str(prof_path))
    p.strip_dirs()
    p.sort_stats('cumulative')
    # capture print_stats output
    import io
    buf = io.StringIO()
    if lines is None:
        p.print_stats()
    else:
        p.print_stats(lines)
    # p.print_stats writes to stdout by default; use pstats.Stats to_string fallback
    # to ensure we get output in buf, re-run using run method that writes to stdout captured
    # simpler: call p.print_stats and capture sys.stdout temporarily
    import contextlib
    with contextlib.redirect_stdout(buf):
        if lines is None:
            p.print_stats()
        else:
            p.print_stats(lines)
    return buf.getvalue()


class ProfileWindow(QMainWindow):
    def __init__(self, prof_path: Path):
        super().__init__()
        self.setWindowTitle(f'Profile Viewer — {prof_path.name}')
        self.resize(1000, 800)
        self.prof_path = prof_path

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        btn_layout = QHBoxLayout()
        self.reload_btn = QPushButton('Reload')
        self.save_btn = QPushButton('Save As...')
        btn_layout.addWidget(self.reload_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        f = QFont('Courier New')
        f.setPointSize(10)
        self.text.setFont(f)
        layout.addWidget(self.text)

        self.reload_btn.clicked.connect(self.load)
        self.save_btn.clicked.connect(self.save_as)

        self.load()

    def load(self):
        try:
            txt = format_stats(self.prof_path, lines=200)
            header = f'Profile file: {self.prof_path}\n\n'
            self.text.setPlainText(header + txt)
        except Exception as e:
            self.text.setPlainText(f'Failed to load profile: {e}')

    def save_as(self):
        fn, _ = QFileDialog.getSaveFileName(self, 'Save profile text', str(self.prof_path.with_suffix('.txt')))
        if not fn:
            return
        try:
            with open(fn, 'w') as f:
                f.write(self.text.toPlainText())
        except Exception as e:
            # best-effort: show error in text box
            self.text.setPlainText(f'Failed to save: {e}')


def main(argv):
    if len(argv) < 2:
        print('Usage: show_profile.py <profile.prof>')
        return 2
    prof = Path(argv[1])
    if not prof.exists():
        print('Profile not found:', prof)
        return 2
    app = QApplication(argv)
    w = ProfileWindow(prof)
    w.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
