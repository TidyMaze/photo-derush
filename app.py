#!/usr/bin/env python3
"""
Top-level launcher to run the app without manipulating PYTHONPATH.
Usage: python app.py
This simply delegates to src.app.main().
"""
import sys
import logging

# Keep logging minimal here; the app configures logging itself.
from src import app as _src_app


def main():
    try:
        _src_app.main()
    except Exception:
        logging.exception("Failed to start src.app")
        sys.exit(1)


if __name__ == "__main__":
    main()

