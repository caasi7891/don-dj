"""Painted jog wheel, knobs, faders, and waveforms."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QConicalGradient,
    QFont,
    QImage,
    QPainter,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

HOT_COLORS = [
    "#e11d48",
    "#f97316",
    "#eab308",
    "#22c55e",
    "#14b8a6",
    "#3b82f6",
    "#8b5cf6",
    "#ec4899",
]


def _rgb_image(arr: np.ndarray) -> QImage:
    data = np.ascontiguousarray(arr, dtype=np.uint8)
    height, width, _channels = data.shape
    image = QImage(data.data, width, height, 3 * width, QImage.Format_RGB888)
    return image.copy()


def waveform_image(peaks: np.ndarray, height: int = 96) -> QImage | None:
    if peaks is None or len(peaks) == 0:
        return None
    width = len(peaks)
    img = np.zeros((height, width, 3), np.uint8)
    img[:] = (12, 14, 20)
    amp = np.clip(peaks[:, 0], 0, 1)
    color = np.zeros((width, 3), np.uint8)
    color[:, 0] = np.clip(40 + np.clip(peaks[:, 1], 0, 1) * 215, 0, 255)
    color[:, 1] = np.clip(28 + np.clip(peaks[:, 2], 0, 1) * 200, 0, 255)
    color[:, 2] = np.clip(55 + np.clip(peaks[:, 3], 0, 1) * 200, 0, 255)
    mid_y = height // 2
    reach = np.maximum(1, (amp * (height * 0.46)).astype(np.int32))
    rows = np.arange(height)[:, None]
    mask = (rows >= (mid_y - reach)[None, :]) & (rows < (mid_y + reach)[None, :])
    painted = np.broadcast_to(color, (height, width, 3))
    img[mask] = painted[mask]
    return _rgb_image(img)


class WaveformView(QWidget):
    seekRequested = Signal(float)

    def __init__(self, overview: bool = False, parent=None):
        super().__init__(parent)
        self.overview = overview
        self.image = None
        self.duration = 0.0
        self.detail_hz = 40.0
        self.pos = 0.0
        self.bpm = 0.0
        self.beat_offset = 0.0
        self.cue = 0.0
        self.hot = [None] * 8
        self.loop = None
        self.window_sec = 6.0
        self.empty = "곡을 올려 주세요"
        self.setMinimumHeight(36 if overview else 64)
        self.setMouseTracking(True)

    def set_track(self, peaks, duration: float, detail_hz: float | None) -> None:
        self.duration = float(duration or 0)
        if detail_hz:
            self.detail_hz = float(detail_hz)
        self.image = waveform_image(peaks, 72 if self.overview else 110) if peaks is not None else None
        self.update()

    def set_playhead(self, pos, bpm, beat_offset, cue, hot, loop) -> None:
        self.pos = float(pos)
        self.bpm = float(bpm or 0)
        self.beat_offset = float(beat_offset or 0)
        self.cue = float(cue or 0)
        self.hot = list(hot or [None] * 8)
        self.loop = loop
        self.update()

    def _time_at(self, x: float) -> float:
        width = max(1, self.width())
        if self.overview:
            if self.duration <= 0:
                return 0.0
            return self.duration * float(np.clip(x / width, 0, 1))
        left = self.pos - self.window_sec * 0.28
        return left + self.window_sec * (x / width)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.duration > 0:
            self.seekRequested.emit(max(0.0, self._time_at(event.position().x())))

    def mouseMoveEvent(self, event) -> None:
        if event.buttons() & Qt.LeftButton and self.duration > 0:
            self.seekRequested.emit(max(0.0, self._time_at(event.position().x())))

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect()
        painter.fillRect(rect, QColor("#0c0e14"))
        if self.image is None or self.duration <= 0:
            painter.setPen(QColor("#8b93a7"))
            painter.drawText(rect, Qt.AlignCenter, self.empty)
            painter.setPen(QPen(QColor("#2c3344"), 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
            return
        if self.overview:
            painter.drawImage(QRectF(rect), self.image)
            self._draw_overlays(painter, rect, 0.0, self.duration)
        else:
            left = self.pos - self.window_sec * 0.28
            self._blit_detail(painter, rect, left)
            self._draw_overlays(painter, rect, left, left + self.window_sec)
        painter.setPen(QPen(QColor("#2c3344"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

    def _blit_detail(self, painter: QPainter, rect, left: float) -> None:
        if self.detail_hz <= 0:
            return
        col_left = left * self.detail_hz
        col_w = max(1.0, self.window_sec * self.detail_hz)
        width = self.image.width()

        def x_of(col: float) -> float:
            return rect.left() + (col - col_left) / col_w * rect.width()

        vis_l = max(0.0, col_left)
        vis_r = min(float(width), col_left + col_w)
        if vis_r <= vis_l:
            return
        src = QRectF(vis_l, 0, vis_r - vis_l, self.image.height())
        dst = QRectF(x_of(vis_l), rect.top(), max(1.0, x_of(vis_r) - x_of(vis_l)), rect.height())
        painter.drawImage(dst, self.image, src)

    def _x_for_time(self, rect, start: float, end: float, seconds: float) -> float:
        span = max(1e-6, end - start)
        return rect.left() + (seconds - start) / span * rect.width()

    def _draw_overlays(self, painter: QPainter, rect, start: float, end: float) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        if self.loop and self.loop[1] > self.loop[0]:
            x1 = self._x_for_time(rect, start, end, self.loop[0])
            x2 = self._x_for_time(rect, start, end, self.loop[1])
            painter.fillRect(QRectF(x1, rect.top(), x2 - x1, rect.height()), QColor(34, 197, 94, 55))
        if self.bpm > 1 and not self.overview:
            beat = 60.0 / self.bpm
            first = math.floor((start - self.beat_offset) / beat) - 1
            last = math.ceil((end - self.beat_offset) / beat) + 1
            for step in range(int(first), int(last) + 1):
                t = self.beat_offset + step * beat
                if t < start or t > end:
                    continue
                x = self._x_for_time(rect, start, end, t)
                strong = step % 4 == 0
                painter.setPen(QPen(QColor(255, 255, 255, 90 if strong else 35), 1))
                painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
        for index, cue in enumerate(self.hot):
            if cue is None:
                continue
            x = self._x_for_time(rect, start, end, float(cue))
            if rect.left() - 2 <= x <= rect.right() + 2:
                painter.setPen(QPen(QColor(HOT_COLORS[index]), 2))
                painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
        if self.cue > 0.01:
            x = self._x_for_time(rect, start, end, self.cue)
            painter.setPen(QPen(QColor("#f5b942"), 2))
            painter.drawLine(int(x), rect.top(), int(x), rect.bottom())
        x = self._x_for_time(rect, start, end, self.pos)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(int(x), rect.top(), int(x), rect.bottom())


class Knob(QWidget):
    valueChanged = Signal(float)

    def __init__(self, label: str, default: float = 0.5, parent=None):
        super().__init__(parent)
        self.label = label
        self.value = float(default)
        self.default = float(default)
        self.accent = QColor("#d5dbe8")
        self.dragging = False
        self._origin_y = 0.0
        self._origin_value = self.value
        self.setMinimumSize(52, 62)
        self.setCursor(Qt.PointingHandCursor)

    def set_value(self, value: float, emit: bool = False) -> None:
        self.value = float(np.clip(value, 0.0, 1.0))
        self.update()
        if emit:
            self.valueChanged.emit(self.value)

    def mousePressEvent(self, event) -> None:
        self.dragging = True
        self._origin_y = event.position().y()
        self._origin_value = self.value

    def mouseMoveEvent(self, event) -> None:
        if not self.dragging:
            return
        delta = (self._origin_y - event.position().y()) / 140.0
        self.set_value(self._origin_value + delta, emit=True)

    def mouseReleaseEvent(self, _event) -> None:
        self.dragging = False

    def mouseDoubleClickEvent(self, _event) -> None:
        self.set_value(self.default, emit=True)

    def wheelEvent(self, event) -> None:
        step = 0.03 if event.angleDelta().y() > 0 else -0.03
        self.set_value(self.value + step, emit=True)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        side = min(self.width() - 8, self.height() - 18)
        rect = QRectF((self.width() - side) / 2, 2, side, side)
        pen = QPen(QColor("#2a3142"), 3)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, 225 * 16, -270 * 16)
        painter.setPen(QPen(self.accent, 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawArc(rect, 225 * 16, int(-270 * self.value * 16))
        angle = math.radians(225 - 270 * self.value)
        radius = side / 2
        center = rect.center()
        end = QPointF(center.x() + math.cos(angle) * (radius - 3), center.y() - math.sin(angle) * (radius - 3))
        inner = QPointF(center.x() + math.cos(angle) * (radius * 0.45), center.y() - math.sin(angle) * (radius * 0.45))
        painter.setPen(QPen(QColor("#f4f6fb"), 2))
        painter.drawLine(inner, end)
        painter.setPen(QColor("#c5ccda"))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(QRectF(0, self.height() - 16, self.width(), 16), Qt.AlignCenter, self.label)


class VFader(QWidget):
    """Vertical fader. top_value is the value at the top of the travel."""

    valueChanged = Signal(float)

    def __init__(self, top_value: float, bottom_value: float, default: float, bipolar: bool = False, parent=None):
        super().__init__(parent)
        self.top_value = float(top_value)
        self.bottom_value = float(bottom_value)
        self.default = float(default)
        self.value = float(default)
        self.bipolar = bipolar
        self.accent = QColor("#e8edf7")
        self.dragging = False
        self.setMinimumWidth(36)
        self.setMinimumHeight(96)
        self.setCursor(Qt.PointingHandCursor)

    def set_value(self, value: float, emit: bool = False) -> None:
        low = min(self.top_value, self.bottom_value)
        high = max(self.top_value, self.bottom_value)
        self.value = float(np.clip(value, low, high))
        self.update()
        if emit:
            self.valueChanged.emit(self.value)

    def _value_from_y(self, y: float) -> float:
        span = max(1.0, self.height() - 20)
        t = float(np.clip((y - 8) / span, 0.0, 1.0))
        value = self.top_value + (self.bottom_value - self.top_value) * t
        if self.bipolar and abs(value) < 0.035:
            value = 0.0
        return value

    def mousePressEvent(self, event) -> None:
        self.dragging = True
        self.set_value(self._value_from_y(event.position().y()), emit=True)

    def mouseMoveEvent(self, event) -> None:
        if self.dragging:
            self.set_value(self._value_from_y(event.position().y()), emit=True)

    def mouseReleaseEvent(self, _event) -> None:
        self.dragging = False

    def mouseDoubleClickEvent(self, _event) -> None:
        self.set_value(self.default, emit=True)

    def wheelEvent(self, event) -> None:
        step = (self.top_value - self.bottom_value) * 0.04
        direction = 1 if event.angleDelta().y() > 0 else -1
        self.set_value(self.value + step * direction, emit=True)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        span = max(1.0, self.height() - 20)
        low = min(self.top_value, self.bottom_value)
        high = max(self.top_value, self.bottom_value)
        t = 0.0 if high == low else (self.value - self.top_value) / (self.bottom_value - self.top_value)
        y = 8 + float(np.clip(t, 0, 1)) * span
        groove = QRectF(self.width() / 2 - 3, 8, 6, span)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#12151c"))
        painter.drawRoundedRect(groove, 3, 3)
        if self.bipolar:
            center_t = (0 - self.top_value) / (self.bottom_value - self.top_value)
            cy = 8 + center_t * span
            painter.setPen(QPen(QColor("#667088"), 1))
            painter.drawLine(int(groove.left() - 4), int(cy), int(groove.right() + 4), int(cy))
        cap = QRectF(self.width() / 2 - 11, y - 9, 22, 18)
        painter.setBrush(self.accent)
        painter.drawRoundedRect(cap, 4, 4)
        painter.setPen(QPen(QColor("#0c0e13"), 1))
        painter.drawLine(int(cap.left() + 4), int(cap.center().y()), int(cap.right() - 4), int(cap.center().y()))


class HFader(QWidget):
    valueChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.5
        self.dragging = False
        self.setMinimumHeight(36)
        self.setCursor(Qt.PointingHandCursor)

    def set_value(self, value: float, emit: bool = False) -> None:
        value = float(np.clip(value, 0.0, 1.0))
        if abs(value - 0.5) < 0.02:
            value = 0.5
        self.value = value
        self.update()
        if emit:
            self.valueChanged.emit(self.value)

    def _from_x(self, x: float) -> float:
        pad = 12
        return float(np.clip((x - pad) / max(1.0, self.width() - pad * 2), 0.0, 1.0))

    def mousePressEvent(self, event) -> None:
        self.dragging = True
        self.set_value(self._from_x(event.position().x()), emit=True)

    def mouseMoveEvent(self, event) -> None:
        if self.dragging:
            self.set_value(self._from_x(event.position().x()), emit=True)

    def mouseReleaseEvent(self, _event) -> None:
        self.dragging = False

    def mouseDoubleClickEvent(self, _event) -> None:
        self.set_value(0.5, emit=True)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pad = 12
        y = self.height() / 2
        painter.setPen(QPen(QColor("#2a3142"), 6, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(pad, int(y), self.width() - pad, int(y))
        painter.setPen(QPen(QColor("#3b82f6"), 6, Qt.SolidLine, Qt.RoundCap))
        mid = pad + self.value * (self.width() - pad * 2)
        painter.drawLine(pad, int(y), int(mid), int(y))
        painter.setPen(QPen(QColor("#ff7a18"), 6, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(int(mid), int(y), self.width() - pad, int(y))
        center = pad + 0.5 * (self.width() - pad * 2)
        painter.setPen(QPen(QColor("#667088"), 1))
        painter.drawLine(int(center), int(y - 10), int(center), int(y + 10))
        painter.setBrush(QColor("#f4f6fb"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(QRectF(mid - 7, y - 12, 14, 24), 3, 3)


class Meter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0.0
        self.peak = 0.0
        self.setMinimumWidth(10)
        self.setMinimumHeight(40)

    def set_level(self, level: float) -> None:
        level = float(np.clip(level, 0.0, 1.0))
        self.level = self.level * 0.55 + level * 0.45
        self.peak = max(level, self.peak * 0.92)
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#12151c"))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 2, 2)
        height = self.height() - 4
        filled = int(height * min(1.0, self.level * 1.3))
        color = QColor("#22c55e") if self.level < 0.8 else QColor("#f5b942") if self.level < 0.95 else QColor("#ef4444")
        painter.setBrush(color)
        painter.drawRect(3, self.height() - 2 - filled, self.width() - 6, filled)


class JogWheel(QWidget):
    pressedAt = Signal(float)
    moved = Signal(float, float, float)
    released = Signal()

    def __init__(self, letter: str, accent: str, parent=None):
        super().__init__(parent)
        self.letter = letter
        self.accent = QColor(accent)
        self.angle = 0.0
        self._last = None
        self._total = 0.0
        self.setMinimumSize(118, 118)
        self.setCursor(Qt.OpenHandCursor)

    def set_angle_from_seconds(self, seconds: float) -> None:
        # 33 1/3 revolutions per minute, matching a turntable.
        self.angle = (seconds * (33.333 / 60.0)) * 2 * math.pi
        self.update()

    def _point(self, event):
        center = QPointF(self.width() / 2, self.height() / 2)
        pos = event.position()
        dx = pos.x() - center.x()
        dy = pos.y() - center.y()
        radius = math.hypot(dx, dy)
        limit = min(center.x(), center.y())
        ratio = 0.0 if limit <= 0 else radius / limit
        angle = math.atan2(dy, dx)
        return angle, ratio

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        angle, ratio = self._point(event)
        self._last = angle
        self._total = 0.0
        self.setCursor(Qt.ClosedHandCursor)
        self.pressedAt.emit(ratio)

    def mouseMoveEvent(self, event) -> None:
        if self._last is None or not (event.buttons() & Qt.LeftButton):
            return
        angle, ratio = self._point(event)
        delta = angle - self._last
        if delta > math.pi:
            delta -= 2 * math.pi
        elif delta < -math.pi:
            delta += 2 * math.pi
        self._last = angle
        self._total += delta
        self.moved.emit(self._total / (2 * math.pi), delta / (2 * math.pi), ratio)

    def mouseReleaseEvent(self, _event) -> None:
        self._last = None
        self.setCursor(Qt.OpenHandCursor)
        self.released.emit()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height()) - 4
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        center = rect.center()
        gradient = QRadialGradient(center, side / 2)
        gradient.setColorAt(0.0, QColor("#3a4154"))
        gradient.setColorAt(0.72, QColor("#1a1e28"))
        gradient.setColorAt(0.84, QColor("#0c0e13"))
        gradient.setColorAt(1.0, self.accent)
        painter.setPen(QPen(QColor("#0a0c10"), 2))
        painter.setBrush(gradient)
        painter.drawEllipse(rect)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor("#6d768c"), 1))
        dots = rect.adjusted(8, 8, -8, -8)
        for mark in range(24):
            painter.drawArc(dots, int(mark * 15 * 16), 3 * 16)
        mark_angle = -self.angle
        radius = side * 0.36
        tip = QPointF(center.x() + math.cos(mark_angle) * radius, center.y() + math.sin(mark_angle) * radius)
        painter.setPen(QPen(QColor("#f8fafc"), 3, Qt.SolidLine, Qt.RoundCap))
        painter.drawLine(center, tip)
        inner = side * 0.34
        inner_rect = QRectF(center.x() - inner / 2, center.y() - inner / 2, inner, inner)
        cone = QConicalGradient(center, 0)
        cone.setColorAt(0.0, QColor("#262b38"))
        cone.setColorAt(1.0, QColor("#171b24"))
        painter.setPen(QPen(self.accent, 2))
        painter.setBrush(cone)
        painter.drawEllipse(inner_rect)
        painter.setPen(QColor("#f4f6fb"))
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(inner_rect, Qt.AlignCenter, self.letter)


class PhaseMeter(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase_a = None
        self.phase_b = None
        self.setMinimumSize(72, 72)

    def set_phases(self, phase_a, phase_b) -> None:
        self.phase_a = phase_a
        self.phase_b = phase_b
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        side = min(self.width(), self.height()) - 8
        rect = QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        center = rect.center()
        painter.setPen(QPen(QColor("#2a3142"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(rect)
        painter.setPen(QPen(QColor("#667088"), 1))
        painter.drawLine(int(center.x()), int(rect.top()), int(center.x()), int(rect.bottom()))
        radius = side / 2 - 6

        def dot(phase, color):
            if phase is None:
                return
            angle = -math.pi / 2 + float(phase) * 2 * math.pi
            point = QPointF(center.x() + math.cos(angle) * radius, center.y() + math.sin(angle) * radius)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(color))
            painter.drawEllipse(point, 5, 5)

        dot(self.phase_a, "#3b82f6")
        dot(self.phase_b, "#ff7a18")
        if self.phase_a is not None and self.phase_b is not None:
            err = abs((self.phase_a - self.phase_b + 0.5) % 1.0 - 0.5)
            painter.setPen(QColor("#22c55e") if err < 0.06 else QColor("#8b93a7"))
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            label = "맞춤" if err < 0.06 else "박자"
            painter.drawText(rect, Qt.AlignCenter, label)
