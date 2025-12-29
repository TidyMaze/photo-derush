from PIL import Image
import time
import io
import random
import sys

# Create a random RGBA image at 1024x1024
W = 1024
H = 1024
arr = bytearray(random.getrandbits(8) for _ in range(W * H * 3))
img = Image.frombytes('RGB', (W, H), bytes(arr))

# Resize target
size = 128

def bench_pil_resize_tobytes(iterations=20):
    t0 = time.perf_counter()
    for _ in range(iterations):
        r = img.resize((size, size), resample=Image.LANCZOS).convert('RGBA')
        b = r.tobytes('raw', 'RGBA')
        # touch
        _ = b[0]
    return time.perf_counter() - t0

def bench_pil_resize_png(iterations=20):
    t0 = time.perf_counter()
    for _ in range(iterations):
        r = img.resize((size, size), resample=Image.LANCZOS)
        buf = io.BytesIO()
        r.save(buf, format='PNG')
        data = buf.getvalue()
        _ = data[0]
    return time.perf_counter() - t0


def bench_imageqt_to_qpixmap(iterations=20):
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QPixmap
        from PIL.ImageQt import ImageQt
    except Exception as e:
        print('Qt/ImageQt import failed:', e)
        return None
    app = QApplication.instance() or QApplication([])
    t0 = time.perf_counter()
    for _ in range(iterations):
        r = img.resize((size, size), resample=Image.LANCZOS).convert('RGBA')
        qimg = ImageQt(r)
        pm = QPixmap.fromImage(qimg)
        _ = pm.size()
    return time.perf_counter() - t0


def bench_qpixmap_loadfromdata(iterations=20):
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QPixmap
    except Exception as e:
        print('Qt import failed:', e)
        return None
    app = QApplication.instance() or QApplication([])
    t0 = time.perf_counter()
    for _ in range(iterations):
        r = img.resize((size, size), resample=Image.LANCZOS)
        buf = io.BytesIO()
        r.save(buf, format='PNG')
        data = buf.getvalue()
        pm = QPixmap()
        ok = pm.loadFromData(data)
        _ = ok
    return time.perf_counter() - t0

if __name__ == '__main__':
    iters = 30
    print('Iterations:', iters)
    t = bench_pil_resize_tobytes(iters)
    print('PIL resize+tobytes: %.3fs total, %.3f ms/op' % (t, (t/iters)*1000))
    t = bench_pil_resize_png(iters)
    print('PIL resize+PNG save: %.3fs total, %.3f ms/op' % (t, (t/iters)*1000))
    t = bench_imageqt_to_qpixmap(iters)
    print('ImageQt->QPixmap: %s' % (('%.3fs total, %.3f ms/op' % (t, (t/iters)*1000)) if t is not None else 'skipped'))
    t = bench_qpixmap_loadfromdata(iters)
    print('QPixmap.loadFromData(PNG): %s' % (('%.3fs total, %.3f ms/op' % (t, (t/iters)*1000)) if t is not None else 'skipped'))
