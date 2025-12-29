"""Human-in-the-loop auto-label review system.

Tracks high-confidence auto-label predictions and flags them for user review.
Only manually-confirmed labels are added to training data to avoid feedback loops.
"""

from __future__ import annotations

import json
import logging
import os


class AutoLabelReviewManager:
    """Manages high-confidence auto-label candidates for human review."""

    def __init__(self, data_dir: str, confidence_threshold: float = 0.9):
        """Initialize review manager.

        Args:
            data_dir: Directory to store review metadata
            confidence_threshold: Min confidence (0-1) to flag for review (default 0.9)
        """
        self.data_dir = data_dir
        self.confidence_threshold = confidence_threshold
        self.review_file = os.path.join(data_dir, ".auto_label_review.json")
        self._review_data: dict[str, dict] = {}
        self._load_review_data()

    def _load_review_data(self) -> None:
        """Load review candidates from disk."""
        if os.path.isfile(self.review_file):
            try:
                with open(self.review_file) as f:
                    self._review_data = json.load(f)
                logging.info(f"[review] Loaded {len(self._review_data)} review candidates")
            except Exception as e:
                logging.warning(f"[review] Failed to load review data: {e}")
                self._review_data = {}
        else:
            self._review_data = {}

    def _save_review_data(self) -> None:
        """Save review candidates to disk."""
        try:
            with open(self.review_file, "w") as f:
                json.dump(self._review_data, f, indent=2)
        except Exception as e:
            logging.error(f"[review] Failed to save review data: {e}")

    def flag_for_review(self, filename: str, predicted_label: str, confidence: float) -> bool:
        """Flag a high-confidence prediction for user review.

        Returns: True if flagged, False if below threshold
        """
        if confidence < self.confidence_threshold:
            return False

        self._review_data[filename] = {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "reviewed": False,
            "user_label": None,
        }
        logging.debug(f"[review] Flagged {filename} ({predicted_label} @ {confidence:.2f})")
        return True

    def get_review_candidates(self) -> list[dict]:
        """Get list of flagged items pending review.

        Returns: [{filename, predicted_label, confidence}, ...]
        """
        pending = [
            {
                "filename": fname,
                "predicted_label": data["predicted_label"],
                "confidence": data["confidence"],
            }
            for fname, data in self._review_data.items()
            if not data.get("reviewed", False)
        ]
        return pending

    def confirm_label(self, filename: str, user_label: str) -> bool:
        """User confirms or corrects an auto-label.

        Args:
            filename: Image filename
            user_label: 'keep', 'trash', or '' to reject

        Returns: True if confirmed, False if not found
        """
        if filename not in self._review_data:
            return False

        self._review_data[filename]["reviewed"] = True
        self._review_data[filename]["user_label"] = user_label
        self._save_review_data()

        if user_label:
            logging.info(
                f"[review] Confirmed: {filename} → {user_label} "
                f"(was {self._review_data[filename]['predicted_label']} @ {self._review_data[filename]['confidence']:.2f})"
            )
        else:
            logging.info(f"[review] Rejected: {filename}")

        return True

    def get_confirmed_labels(self) -> dict[str, str]:
        """Get all user-confirmed labels from review candidates.

        Returns: {filename: label} for items user confirmed
        """
        confirmed = {
            fname: data["user_label"]
            for fname, data in self._review_data.items()
            if data.get("reviewed", False) and data.get("user_label")
        }
        return confirmed

    def get_review_stats(self) -> dict:
        """Get statistics on review progress."""
        total = len(self._review_data)
        reviewed = sum(1 for d in self._review_data.values() if d.get("reviewed", False))
        confirmed = sum(1 for d in self._review_data.values() if d.get("user_label"))
        rejected = reviewed - confirmed

        return {
            "total_flagged": total,
            "reviewed": reviewed,
            "pending": total - reviewed,
            "confirmed": confirmed,
            "rejected": rejected,
            "confidence_threshold": self.confidence_threshold,
        }

    def clear_reviewed(self) -> None:
        """Clear all reviewed items (archive old reviews)."""
        self._review_data = {
            fname: data for fname, data in self._review_data.items() if not data.get("reviewed", False)
        }
        self._save_review_data()
        logging.info("[review] Cleared reviewed items")

    def add_confirmed_to_repository(self, repo) -> int:
        """Apply confirmed labels to the repository.

        Args:
            repo: RatingsTagsRepository instance

        Returns: Number of labels added
        """
        confirmed = self.get_confirmed_labels()
        added = 0

        for filename, label in confirmed.items():
            try:
                if label == "keep":
                    repo.set_state(filename, "keep")
                    added += 1
                elif label == "trash":
                    repo.set_state(filename, "trash")
                    added += 1
            except Exception as e:
                logging.warning(f"[review] Failed to add label for {filename}: {e}")

        if added > 0:
            logging.info(f"[review] Added {added} confirmed labels to repository")

        return added


__all__ = ["AutoLabelReviewManager"]
