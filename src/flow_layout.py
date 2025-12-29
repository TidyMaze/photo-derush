from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout


class FlowLayout(QLayout):
    """A simple flow layout that places child widgets horizontally and wraps them.

    Source: lightweight adaptation for PySide6 inspired by Qt examples.
    """

    def __init__(self, parent=None, margin=0, spacing=6):
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        # Do NOT force a wide minimum width (summing child widths can produce
        # a very large minimum and push parent widgets to expand). Instead,
        # return a flexible minimum width (0) and a sensible minimum height
        # based on the tallest child plus margins. This allows the parent to
        # constrain width and lets the FlowLayout reflow children using
        # height-for-width logic in `_do_layout`.
        height = 0
        for item in self._items:
            sz = item.minimumSize()
            height = max(height, sz.height())

        m = self.contentsMargins()
        height += m.top() + m.bottom()

        return QSize(0, height)

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        x = rect.x()
        y = rect.y()
        line_height = 0

        effective_rect = rect.adjusted(
            +self.contentsMargins().left(),
            +self.contentsMargins().top(),
            -self.contentsMargins().right(),
            -self.contentsMargins().bottom(),
        )
        x = effective_rect.x()
        y = effective_rect.y()
        max_width = effective_rect.width()

        for item in self._items:
            w = item.sizeHint().width()
            h = item.sizeHint().height()
            if x + w > effective_rect.x() + max_width and x != effective_rect.x():
                x = effective_rect.x()
                y = y + line_height + self._spacing
                line_height = 0

            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = x + w + self._spacing
            line_height = max(line_height, h)

        return y + line_height + self.contentsMargins().bottom()
