from PySide6.QtCore import Signal, Qt, QRectF
from PySide6.QtGui import QPen, QColor, QBrush
import pyqtgraph as pg

class ScopeViewBox(pg.ViewBox):
    """Custom ViewBox with oscilloscope-style mouse controls.

    Controls:
        Left-drag        → Pan (X and Y)
        Scroll wheel     → Zoom X (time axis)
        Ctrl + scroll    → Zoom Y (value axis)
        Right-drag       → Rubber-band zoom to region
        Double-click     → Reset view / re-enable auto-scroll
        Middle-click     → Context menu
    """

    # Signal emitted on double-click so the main app can re-enable auto-scroll
    doubleClicked = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Left-drag = pan (not rubber band)
        self.setMouseMode(pg.ViewBox.PanMode)
        # Disable default right-click context menu (we use right-drag for zoom)
        self.menu = None
        # Rubber band rectangle overlay
        self._rb_rect = None
        # When True, wheel zooms both axes uniformly (for XY mode)
        self.uniform_zoom = False

    def wheelEvent(self, ev, axis=None):
        """Scroll wheel: zoom Y by default, zoom X with Ctrl held.
        In uniform_zoom mode (XY plots), scroll zooms both axes together."""
        if self.uniform_zoom:
            # Zoom both axes together — no axis restriction
            super().wheelEvent(ev, axis=None)
            return
        modifiers = ev.modifiers() if hasattr(ev, 'modifiers') else Qt.NoModifier
        if modifiers == Qt.ControlModifier:
            # Ctrl + scroll → zoom X only (time axis)
            super().wheelEvent(ev, axis=0)
        else:
            # Plain scroll → zoom Y only
            super().wheelEvent(ev, axis=1)

    def mouseDragEvent(self, ev, axis=None):
        """Left-drag = pan, Right-drag = rubber-band zoom."""
        if ev.button() == Qt.RightButton:
            ev.accept()
            if ev.isStart():
                # Create rubber band rectangle
                self._rb_rect = pg.QtWidgets.QGraphicsRectItem(self.childGroup)
                pen = QPen(QColor('#FFA500'), 1, Qt.DashLine)
                pen.setCosmetic(True)
                self._rb_rect.setPen(pen)
                self._rb_rect.setBrush(QBrush(QColor(255, 165, 0, 40)))

            if ev.isFinish():
                # Remove rectangle
                if self._rb_rect is not None:
                    self._rb_rect.setParentItem(None)
                    self._rb_rect = None

                # Zoom to the dragged region
                r = pg.Point(ev.buttonDownPos()) - pg.Point(ev.pos())
                start = self.mapToView(ev.buttonDownPos())
                end = self.mapToView(ev.pos())
                x0, x1 = sorted([start.x(), end.x()])
                y0, y1 = sorted([start.y(), end.y()])
                if abs(r.x()) > 5 or abs(r.y()) > 5:
                    self.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0)
            else:
                # Update rubber band rectangle during drag
                if self._rb_rect is not None:
                    start = self.mapToView(ev.buttonDownPos())
                    end = self.mapToView(ev.pos())
                    r = QRectF(start, end).normalized()
                    self._rb_rect.setRect(r)
        else:
            # Left-drag: default pan behavior
            super().mouseDragEvent(ev, axis)

    def mouseDoubleClickEvent(self, ev):
        """Double-click: reset view to auto-range and re-enable auto-scroll."""
        if ev.button() == Qt.LeftButton:
            self.enableAutoRange()
            self.doubleClicked.emit()
            ev.accept()
        else:
            super().mouseDoubleClickEvent(ev)
