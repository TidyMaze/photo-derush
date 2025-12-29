"""Filter controller separated from PhotoViewModel (SRP).
Handles rating/tag/date filter state and application.
"""

from __future__ import annotations

import logging


class FilterController:
    def __init__(self):
        self.rating: int = 0
        self.tag: str = ""
        self.date: str = ""
        self.hide_manual: bool = False

    def active(self) -> bool:
        return bool(self.rating or self.tag or self.date or self.hide_manual)

    def apply(self, images: list[str], model) -> list[str]:
        if not images or not self.active():
            return list(images) if images else []

        tag_lower = (self.tag or "").lower()
        out: list[str] = []
        for fname in images:
            path = model.get_image_path(fname)
            if not path:
                continue
            if self.rating:
                r = model.get_rating(path)
                if r < self.rating:
                    continue
            if tag_lower:
                tags = model.get_tags(path) or []
                tags_lower = [t.lower() for t in tags]
                if tag_lower not in tags_lower:
                    continue
            # date placeholder
            out.append(fname)
        logging.debug(
            "[filters] applied rating=%s tag=%s date=%s -> %d/%d",
            self.rating,
            self.tag,
            self.date,
            len(out),
            len(images),
        )
        return out

    def set_rating(self, rating: int) -> bool:
        rating = int(rating) if rating else 0
        if rating == self.rating:
            return False
        self.rating = rating
        return True

    def set_tag(self, tag: str) -> bool:
        tag = (tag or "").strip()
        if tag == self.tag:
            return False
        self.tag = tag
        return True

    def set_date(self, date: str) -> bool:
        date = (date or "").strip()
        if date == self.date:
            return False
        self.date = date
        return True

    def clear(self) -> bool:
        if not self.active():
            return False
        self.rating = 0
        self.tag = ""
        self.date = ""
        self.hide_manual = False
        return True

    def set_hide_manual(self, hide: bool) -> bool:
        hide = bool(hide)
        if hide == self.hide_manual:
            return False
        self.hide_manual = hide
        return True

    def set_batch(self, rating=None, tag=None, date=None, hide_manual=None) -> bool:
        changed = False
        if rating is not None:
            r = int(rating) if rating else 0
            if r != self.rating:
                self.rating = r
                changed = True
        if tag is not None:
            t = (tag or "").strip()
            if t != self.tag:
                self.tag = t
                changed = True
        if date is not None:
            d = (date or "").strip()
            if d != self.date:
                self.date = d
                changed = True
        if hide_manual is not None:
            h = bool(hide_manual)
            if h != self.hide_manual:
                self.hide_manual = h
                changed = True
        return changed


__all__ = ["FilterController"]
