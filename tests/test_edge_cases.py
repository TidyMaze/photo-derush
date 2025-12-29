"""Tests for edge cases and error handling."""
from __future__ import annotations

import os

import pytest
from PIL import Image

from src.inference import load_model
from src.training import train_keep_trash_model


class TestCorruptImages:
    """Test handling of corrupt/invalid image files."""

    def test_corrupt_image_graceful_skip(self, tmp_path):
        """Corrupt image should be skipped, not crash extraction."""
        # Create valid image
        valid_img = tmp_path / "valid.jpg"
        img = Image.new('RGB', (100, 100), (255, 0, 0))
        img.save(valid_img)

        # Create corrupt image (truncated file)
        corrupt_img = tmp_path / "corrupt.jpg"
        with open(corrupt_img, 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0' + b'\x00' * 10)  # JPEG header but no data

        from src.features import extract_features

        # Valid image should extract fine
        feats_valid = extract_features(str(valid_img))
        assert feats_valid is not None
        assert len(feats_valid) > 0

        # Corrupt image should return None (handled gracefully)
        feats_corrupt = extract_features(str(corrupt_img))
        assert feats_corrupt is None

    def test_nonexistent_image_returns_none(self, tmp_path):
        """Non-existent image should not crash."""
        from src.features import extract_features

        result = extract_features(str(tmp_path / "nonexistent.jpg"))
        assert result is None

    def test_empty_file_returns_none(self, tmp_path):
        """Empty file should not crash."""
        empty_file = tmp_path / "empty.jpg"
        empty_file.write_text("")

        from src.features import extract_features
        result = extract_features(str(empty_file))
        assert result is None


class TestEmptyDirectories:
    """Test handling of empty or minimal directories."""

    def test_empty_directory_no_crash(self, tmp_path):
        """Empty directory should not crash training."""
        result = train_keep_trash_model(
            str(tmp_path),
            model_path=str(tmp_path / "model.joblib"),
            n_estimators=10,
            min_samples=2,
        )
        # Should return None gracefully (no data)
        assert result is None

    def test_directory_with_single_image_no_crash(self, tmp_path):
        """Single image without labels should not crash."""
        img_path = tmp_path / "single.jpg"
        img = Image.new('RGB', (100, 100), (0, 255, 0))
        img.save(img_path)

        result = train_keep_trash_model(
            str(tmp_path),
            model_path=str(tmp_path / "model.joblib"),
            n_estimators=10,
            min_samples=2,
        )
        # Should return None (insufficient labeled data)
        assert result is None

    def test_directory_with_only_labels_no_images(self, tmp_path):
        """Labels without images should not crash."""
        import json
        labels_file = tmp_path / ".ratings_tags.json"
        labels_file.write_text(json.dumps({"nonexistent.jpg": {"state": "keep"}}))

        result = train_keep_trash_model(
            str(tmp_path),
            model_path=str(tmp_path / "model.joblib"),
            n_estimators=10,
            min_samples=2,
        )
        # Should return None (no matching images)
        assert result is None


class TestMissingModels:
    """Test handling of missing/invalid model files."""

    def test_load_nonexistent_model_raises(self, tmp_path):
        """Loading non-existent model should raise FileNotFoundError."""
        model_path = str(tmp_path / "nonexistent_model.joblib")

        with pytest.raises(FileNotFoundError):
            load_model(model_path)

    def test_load_corrupted_model_graceful(self, tmp_path):
        """Corrupted model file should fail gracefully."""
        model_path = tmp_path / "corrupt_model.joblib"
        # Write corrupted joblib data
        model_path.write_bytes(b'\x80\x04\x95' + b'\x00' * 100)

        result = load_model(str(model_path))
        # Should return None gracefully instead of crashing
        assert result is None

    def test_load_empty_model_file(self, tmp_path):
        """Empty model file should fail gracefully."""
        model_path = tmp_path / "empty_model.joblib"
        model_path.write_bytes(b'')

        result = load_model(str(model_path))
        # Should return None gracefully instead of crashing
        assert result is None


class TestEdgeCaseDatasets:
    """Test handling of minimal/edge-case datasets."""

    def test_single_class_dataset_handled(self, tmp_path):
        """Dataset with only one class should be rejected."""
        # Create 5 keep images, 0 trash
        for i in range(5):
            img_path = tmp_path / f"keep_{i}.jpg"
            img = Image.new('RGB', (50, 50), (255, 0, 0))
            img.save(img_path)

        import json
        labels_file = tmp_path / ".ratings_tags.json"
        labels = {f"keep_{i}.jpg": {"state": "keep"} for i in range(5)}
        labels_file.write_text(json.dumps(labels))

        result = train_keep_trash_model(
            str(tmp_path),
            model_path=str(tmp_path / "model.joblib"),
            n_estimators=10,
            min_samples=2,
        )
        # Should return None (insufficient class balance)
        assert result is None

    def test_minimal_balanced_dataset(self, tmp_path):
        """Minimal balanced dataset (2 keep, 2 trash) should train."""
        # Create 2 keep, 2 trash
        for i in range(2):
            img_path = tmp_path / f"keep_{i}.jpg"
            img = Image.new('RGB', (50, 50), (255, 0, 0))
            img.save(img_path)

            img_path = tmp_path / f"trash_{i}.jpg"
            img = Image.new('RGB', (50, 50), (0, 255, 0))
            img.save(img_path)

        import json
        labels_file = tmp_path / ".ratings_tags.json"
        labels = {
            f"keep_{i}.jpg": {"state": "keep"} for i in range(2)
        }
        labels.update({
            f"trash_{i}.jpg": {"state": "trash"} for i in range(2)
        })
        labels_file.write_text(json.dumps(labels))

        result = train_keep_trash_model(
            str(tmp_path),
            model_path=str(tmp_path / "model.joblib"),
            n_estimators=10,
            min_samples=2,
        )
        # Should succeed with minimal dataset
        assert result is not None
        assert result.n_samples == 4
        assert result.n_keep == 2
        assert result.n_trash == 2


class TestFilePermissions:
    """Test handling of permission-denied scenarios."""

    def test_readonly_directory_no_crash(self, tmp_path):
        """Read-only directory should be handled gracefully."""
        # Create a small valid dataset first
        for i in range(2):
            img = Image.new('RGB', (50, 50), (255, 0, 0))
            img.save(tmp_path / f"img_{i}.jpg")

        # Try to train with read-only directory (may be OS-specific)
        original_mode = os.stat(tmp_path).st_mode
        try:
            os.chmod(tmp_path, 0o444)  # Read-only
            # Attempt should not crash (though may fail gracefully)
            result = train_keep_trash_model(
                str(tmp_path),
                model_path=str(tmp_path / "model.joblib"),
                n_estimators=10,
                min_samples=2,
            )
            # Result may be None or failed, but should not crash
            assert result is None or isinstance(result, object)
        finally:
            os.chmod(tmp_path, original_mode)  # Restore permissions


class TestLargeDatasetScaling:
    """Test handling of larger datasets."""

    def test_many_small_images(self, tmp_path):
        """Many small labeled images should scale OK."""
        # Create 50 images (25 keep, 25 trash)
        count = 0
        keep_count = 0
        trash_count = 0

        for i in range(50):
            img = Image.new('RGB', (32, 32), (255, 0, 0) if i % 2 == 0 else (0, 255, 0))
            img.save(tmp_path / f"img_{i:03d}.jpg")
            count += 1

        import json
        labels_file = tmp_path / ".ratings_tags.json"
        labels = {}
        for i in range(50):
            state = "keep" if i % 2 == 0 else "trash"
            if state == "keep":
                keep_count += 1
            else:
                trash_count += 1
            labels[f"img_{i:03d}.jpg"] = {"state": state}
        labels_file.write_text(json.dumps(labels))

        result = train_keep_trash_model(
            str(tmp_path),
            model_path=str(tmp_path / "model.joblib"),
            n_estimators=10,
            min_samples=2,
        )
        # Should handle 50 images fine
        assert result is not None
        assert result.n_samples == 50
        assert result.n_keep == keep_count
        assert result.n_trash == trash_count

