import unittest
from src.detections import normalize_detections, Detection


class TestNormalizeDetections(unittest.TestCase):
    def test_tuples_and_strings(self):
        inp = [("cat", 0.9), ("dog", 0.1), 'apple']
        dets = normalize_detections(inp)
        self.assertEqual(len(dets), 3)
        self.assertIsInstance(dets[0], Detection)
        self.assertEqual(dets[0].name, 'cat')
        self.assertAlmostEqual(dets[0].confidence, 0.9)

    def test_bbox_dicts(self):
        inp = [{'bbox': [0, 0, 10, 10], 'confidence': 0.75, 'name': 'person'}]
        dets = normalize_detections(inp)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].name, 'person')
        self.assertEqual(dets[0].bbox, [0, 0, 10, 10])

    def test_worker_style(self):
        inp = {'detections': [{'label': 'bird', 'score': 0.6, 'bbox': [1,2,3,4]}]}
        dets = normalize_detections(inp)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].name, 'bird')
        self.assertAlmostEqual(dets[0].confidence, 0.6)

    def test_empty(self):
        self.assertEqual(normalize_detections([]), [])


if __name__ == '__main__':
    unittest.main()
