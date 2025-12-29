"""Tests for bounding box coordinate transformation from detection image to thumbnail."""

import unittest
from unittest.mock import Mock

import numpy as np


class TestBboxCoordinates(unittest.TestCase):
    """Test bounding box coordinate transformation logic."""

    def test_normalize_bbox_from_detection_image(self):
        """Test normalization of bbox coordinates from detection image."""
        # Detection image: 800x600 (after resize to max_size=800)
        det_w, det_h = 800, 600
        
        # Bbox in absolute pixels: [x1, y1, x2, y2]
        bbox_abs = [200, 150, 400, 300]  # Center of image
        
        # Normalize to [0,1]
        x1_norm = bbox_abs[0] / det_w
        y1_norm = bbox_abs[1] / det_h
        x2_norm = bbox_abs[2] / det_w
        y2_norm = bbox_abs[3] / det_h
        
        # Convert to [x, y, w, h] format
        x_norm = x1_norm
        y_norm = y1_norm
        w_norm = x2_norm - x1_norm
        h_norm = y2_norm - y1_norm
        
        # Expected normalized values
        self.assertAlmostEqual(x_norm, 0.25, places=3)
        self.assertAlmostEqual(y_norm, 0.25, places=3)
        self.assertAlmostEqual(w_norm, 0.25, places=3)
        self.assertAlmostEqual(h_norm, 0.25, places=3)

    def test_denormalize_bbox_to_thumbnail_same_aspect_ratio(self):
        """Test denormalization when detection and thumbnail have same aspect ratio."""
        # Detection image: 800x600 (4:3 aspect ratio)
        det_w, det_h = 800, 600
        
        # Normalized bbox (center of image)
        x_norm, y_norm, w_norm, h_norm = 0.25, 0.25, 0.25, 0.25
        
        # Thumbnail: 200x150 (same 4:3 aspect ratio, scaled by 0.25)
        # Canvas: 200x200 (square)
        thumb_w, thumb_h = 200, 150
        offset_x, offset_y = 0, 25  # Centered vertically
        
        # Denormalize to thumbnail dimensions
        x = int(x_norm * thumb_w) + offset_x
        y = int(y_norm * thumb_h) + offset_y
        w = int(w_norm * thumb_w)
        h = int(h_norm * thumb_h)
        
        # Expected: bbox should be at center of thumbnail image
        self.assertEqual(x, 50)  # 0.25 * 200 + 0
        self.assertEqual(y, 62)  # 0.25 * 150 + 25
        self.assertEqual(w, 50)  # 0.25 * 200
        self.assertEqual(h, 37)  # 0.25 * 150 (rounded down)

    def test_denormalize_bbox_to_thumbnail_different_aspect_ratio(self):
        """Test denormalization when detection and thumbnail have different aspect ratios.
        
        This tests the case where:
        - Original image: 3000x4000 (3:4 portrait)
        - Detection image: resized to max_size=800, so 600x800 (3:4 portrait)
        - Thumbnail: created from original, resized to fit 200x200, so 150x200 (3:4 portrait)
        
        Both preserve aspect ratio, so relative positions should be correct.
        """
        # Detection image: 600x800 (3:4 aspect ratio)
        det_w, det_h = 600, 800
        
        # Normalized bbox (center of detection image)
        x_norm, y_norm, w_norm, h_norm = 0.5, 0.5, 0.2, 0.2
        
        # Thumbnail: 150x200 (same 3:4 aspect ratio)
        # Canvas: 200x200 (square)
        thumb_w, thumb_h = 150, 200
        offset_x, offset_y = 25, 0  # Centered horizontally
        
        # Denormalize to thumbnail dimensions
        x = int(x_norm * thumb_w) + offset_x
        y = int(y_norm * thumb_h) + offset_y
        w = int(w_norm * thumb_w)
        h = int(h_norm * thumb_h)
        
        # Expected: bbox should be at center of thumbnail image
        self.assertEqual(x, 100)  # 0.5 * 150 + 25
        self.assertEqual(y, 100)  # 0.5 * 200 + 0
        self.assertEqual(w, 30)   # 0.2 * 150
        self.assertEqual(h, 40)  # 0.2 * 200

    def test_bbox_at_corners(self):
        """Test bbox coordinates at image corners."""
        # Detection image: 800x600
        det_w, det_h = 800, 600
        
        # Top-left corner bbox
        bbox_tl = [0, 0, 100, 100]
        x1_norm = bbox_tl[0] / det_w
        y1_norm = bbox_tl[1] / det_h
        x2_norm = bbox_tl[2] / det_w
        y2_norm = bbox_tl[3] / det_h
        x_norm = x1_norm
        y_norm = y1_norm
        w_norm = x2_norm - x1_norm
        h_norm = y2_norm - y1_norm
        
        # Thumbnail: 200x150, offset (0, 25)
        thumb_w, thumb_h = 200, 150
        offset_x, offset_y = 0, 25
        
        x = int(x_norm * thumb_w) + offset_x
        y = int(y_norm * thumb_h) + offset_y
        w = int(w_norm * thumb_w)
        h = int(h_norm * thumb_h)
        
        # Should be at top-left of thumbnail image (not canvas)
        self.assertEqual(x, 0)
        self.assertEqual(y, 25)
        
        # Bottom-right corner bbox
        bbox_br = [700, 500, 800, 600]
        x1_norm = bbox_br[0] / det_w
        y1_norm = bbox_br[1] / det_h
        x2_norm = bbox_br[2] / det_w
        y2_norm = bbox_br[3] / det_h
        x_norm = x1_norm
        y_norm = y1_norm
        w_norm = x2_norm - x1_norm
        h_norm = y2_norm - y1_norm
        
        x = int(x_norm * thumb_w) + offset_x
        y = int(y_norm * thumb_h) + offset_y
        w = int(w_norm * thumb_w)
        h = int(h_norm * thumb_h)
        
        # Should be at bottom-right of thumbnail image
        self.assertEqual(x, 175)  # 0.875 * 200 + 0
        self.assertEqual(y, 150)  # 0.833 * 150 + 25 = 124.95 + 25 = 149.95 -> 150 (rounded)

    def test_retina_scaling(self):
        """Test that retina scaling is applied correctly."""
        # Logical dimensions
        thumb_w_logical, thumb_h_logical = 150, 200
        offset_x_logical, offset_y_logical = 25, 0
        
        # Retina scaling (dpr=2)
        dpr = 2.0
        thumb_w_retina = int(thumb_w_logical * dpr)
        thumb_h_retina = int(thumb_h_logical * dpr)
        offset_x_retina = int(offset_x_logical * dpr)
        offset_y_retina = int(offset_y_logical * dpr)
        
        # Normalized bbox
        x_norm, y_norm, w_norm, h_norm = 0.5, 0.5, 0.2, 0.2
        
        # Denormalize using retina dimensions
        x = int(x_norm * thumb_w_retina) + offset_x_retina
        y = int(y_norm * thumb_h_retina) + offset_y_retina
        w = int(w_norm * thumb_w_retina)
        h = int(h_norm * thumb_h_retina)
        
        # Should be scaled by dpr
        self.assertEqual(x, 200)  # 0.5 * 300 + 50 = 150 + 50 = 200
        self.assertEqual(y, 200)  # 0.5 * 400 + 0 = 200
        self.assertEqual(w, 60)   # 0.2 * 300 = 60
        self.assertEqual(h, 80)   # 0.2 * 400 = 80


if __name__ == "__main__":
    unittest.main()

