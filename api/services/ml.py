"""ML service placeholder (module removed).

This module has been intentionally removed. Keeping a small placeholder
prevents accidental heavy dependency imports during tests and makes the
intent explicit.
"""

__all__ = []


def __getattr__(name: str):
    raise ImportError("api.services.ml has been removed from this codebase")
