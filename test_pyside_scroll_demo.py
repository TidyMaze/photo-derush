def test_pyside_scroll_demo():
    # Import inside test to ensure environment variables can be set first if needed
    from photo_derush.pyside_scroll_demo import create_scroll_window
    items = [f"Row {i}" for i in range(25)]
    app, window = create_scroll_window(items=items, auto_close_ms=50)
    window.show()
    # Manually process a few event cycles instead of app.exec() to stay fast
    for _ in range(20):
        app.processEvents()
    # Find all label widgets showing our items
    from PySide6.QtWidgets import QLabel
    labels = [w for w in window.findChildren(QLabel) if w.text().startswith("Row ")]
    assert len(labels) == len(items), f"Expected {len(items)} labels, got {len(labels)}"
    # Close window cleanly
    window.close()
    for _ in range(5):
        app.processEvents()

