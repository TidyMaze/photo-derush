import torch
from PIL import Image


def test_torchvision_adapter_preprocess_and_predict():
    # Create a dummy weights object with a transforms() method that returns a callable
    class DummyWeights:
        def transforms(self):
            # Return a callable that converts PIL image -> torch tensor (C,H,W)
            def _t(img: Image.Image):
                # produce a deterministic tensor of shape (3, H, W)
                arr = torch.zeros((3, img.height, img.width), dtype=torch.float32)
                return arr
            return _t

    # Dummy model that returns a torchvision-like prediction dict when called
    class DummyModel:
        def __call__(self, batch):
            # batch is a tensor with shape (1, C, H, W)
            return [{
                'boxes': torch.zeros((0, 4)),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'scores': torch.zeros((0,))
            }]

    from src.object_detection import TorchvisionAdapter

    dummy_weights = DummyWeights()
    dummy_model = DummyModel()
    adapter = TorchvisionAdapter(dummy_model, weights=dummy_weights)

    img = Image.new('RGB', (32, 24), color='white')
    tensor = adapter.preprocess(img, device='cpu', max_size=32)
    # preprocess should return a batched tensor
    assert hasattr(tensor, 'shape') and tensor.shape[0] == 1

    preds = adapter.predict(tensor, conf=0.5)
    assert isinstance(preds, list)
    assert isinstance(preds[0], dict)
    assert 'boxes' in preds[0] and 'labels' in preds[0] and 'scores' in preds[0]

    # boxes/labels/scores should be torch tensors (or convertible)
    assert hasattr(preds[0]['boxes'], 'shape')
    assert hasattr(preds[0]['scores'], 'shape')


def test_yolov8_adapter_conversion():
    # Create a fake ultralytics-like result object with boxes attribute
    class FakeBoxes:
        def __init__(self, xyxy, conf, cls):
            import numpy as _np
            self.xyxy = _np.asarray(xyxy)
            self.conf = _np.asarray(conf)
            self.cls = _np.asarray(cls)

    class FakeResult:
        def __init__(self, boxes):
            self.boxes = boxes

    class FakeYOLO:
        def __init__(self):
            pass
        def predict(self, pil_img, imgsz=None, conf=0.001, verbose=False):
            # return a list with one FakeResult
            boxes = FakeBoxes([[10, 10, 20, 20]], [0.99], [0])
            return [FakeResult(boxes)]

    # Patch the YOLOv8Adapter to use the fake YOLO
    from src.object_detection import YOLOv8Adapter
    adapter = YOLOv8Adapter.__new__(YOLOv8Adapter)
    adapter._yolo = FakeYOLO()

    img = Image.new('RGB', (32, 32), color='white')
    results = adapter.predict([img], conf=0.5)
    assert isinstance(results, list)
    assert 'boxes' in results[0] and 'labels' in results[0] and 'scores' in results[0]
    # Ensure outputs are torch tensors
    import torch as _torch
    assert isinstance(results[0]['boxes'], _torch.Tensor)
    assert isinstance(results[0]['labels'], _torch.Tensor)
    assert isinstance(results[0]['scores'], _torch.Tensor)
