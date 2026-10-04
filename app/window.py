"""The don-dj window: two decks, a mixer, and a music library."""
from __future__ import annotations

import os
import threading

import queue

from PySide6.QtCore import QEvent, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.audioio import is_audio, prepare_file, read_tags
from app.dsp import SR
from app.engine import Engine
from app.store import Store
from app.widgets import HOT_COLORS, HFader, JogWheel, Knob, Meter, PhaseMeter, VFader, WaveformView

STYLE = """
QWidget { color: #e7e9f0; font-family: "Malgun Gothic", "Segoe UI"; font-size: 12px; }
QMainWindow, QWidget#root, QScrollArea, QScrollArea > QWidget > QWidget { background: #101218; }
QLineEdit, QComboBox {
  background: #12151c; border: 1px solid #2a3142; border-radius: 6px; padding: 4px 8px;
}
QTableWidget {
  background: #12151c; alternate-background-color: #171c27; gridline-color: #232938;
  border: 1px solid #2a3142; border-radius: 8px;
  selection-background-color: #1e3a5f; selection-color: white;
}
QHeaderView::section { background: #1b2130; color: #aeb6c8; border: none; padding: 4px; }
QPushButton {
  background: #232833; border: 1px solid #343b4d; border-radius: 6px; padding: 4px 8px;
}
QPushButton:hover { background: #2c3344; }
QPushButton:checked { background: #1d4ed8; border-color: #60a5fa; }
QPushButton#play { background: #166534; color: white; font-weight: 700; min-height: 32px; }
QPushButton#play:checked { background: #22c55e; color: #052e16; }
QPushButton#cue { background: #9a3412; color: white; font-weight: 700; min-height: 32px; }
QPushButton#accent { background: #1d4ed8; color: white; font-weight: 700; }
QPushButton#record:checked { background: #be123c; color: white; }
QFrame#panel, QFrame#deck, QFrame#deckB {
  background: #171b24; border: 1px solid #2a3142; border-radius: 10px;
}
QFrame#deck[focused="true"] { border: 2px solid #3b82f6; }
QFrame#deckB[focused="true"] { border: 2px solid #ff7a18; }
QLabel#dim { color: #9aa3b5; }
QLabel#bpm { font-family: Consolas, "Cascadia Mono"; font-size: 22px; font-weight: 700; }
QLabel#tip {
  color: #dbe4ff; background: #151a28; border: 1px solid #243049;
  border-radius: 8px; padding: 6px 10px;
}
QScrollArea { border: none; }
"""

LOOP_CHOICES = [
    ("1/8", 0.125),
    ("1/4", 0.25),
    ("1/2", 0.5),
    ("1", 1),
    ("2", 2),
    ("4", 4),
    ("8", 8),
    ("16", 16),
]


def _button(text: str, tip: str, checkable: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setToolTip(tip)
    button.setCursor(Qt.PointingHandCursor)
    button.setCheckable(checkable)
    return button


def _fmt_clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes}:{rest:04.1f}"


def _fmt_short(seconds: float) -> str:
    seconds = max(0, int(seconds or 0))
    return f"{seconds // 60}:{seconds % 60:02d}"


class ScanWorker(QThread):
    found = Signal(object)
    analyzed = Signal(object)
    status = Signal(str)

    def __init__(self, folders: list, store: Store):
        super().__init__()
        self.folders = folders
        self.store = store
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        paths = []
        for folder in self.folders:
            if self._stop or not os.path.isdir(folder):
                continue
            for current, _dirs, files in os.walk(folder):
                for name in files:
                    path = os.path.join(current, name)
                    if is_audio(path):
                        paths.append(path)
        total = len(paths)
        if total == 0:
            self.status.emit("폴더에서 재생할 곡을 찾지 못했습니다.")
            return
        for index, path in enumerate(paths):
            if self._stop:
                return
            self.status.emit(f"분석 중 {index + 1}/{total}")
            try:
                tags = read_tags(path)
                self.found.emit(tags)
                self.store.upsert_tags(tags)
                if self.store.has_analysis(path, tags["mtime"]):
                    continue
                from app.analyze import analyze_samples
                from app.audioio import decode

                samples = decode(path)
                meta = analyze_samples(samples, SR)
                meta.update(tags)
                meta["duration"] = len(samples) / SR
                self.store.upsert_analysis(meta)
                self.analyzed.emit(_brief(meta))
                del samples
            except Exception:
                self.status.emit("건너뜀: " + os.path.basename(path))
        self.status.emit(f"라이브러리 {total}곡")


def _brief(meta: dict) -> dict:
    return {
        "path": meta.get("path") or "",
        "title": meta.get("title") or "",
        "artist": meta.get("artist") or "",
        "duration": meta.get("duration") or 0,
        "bpm": meta.get("bpm") or 0,
        "musical_key": meta.get("musical_key") or "",
        "camelot": meta.get("camelot") or "",
    }


class LibraryPanel(QFrame):
    loadRequested = Signal(int, str)
    folderRequested = Signal()
    practiceRequested = Signal()
    clearRequested = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        tools = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("제목, 아티스트 검색")
        self.search.textChanged.connect(self._filter)
        folder = _button("음악 폴더 추가", "mp3, wav, flac, ogg 파일이 있는 폴더를 고릅니다.")
        folder.clicked.connect(self.folderRequested.emit)
        practice = _button("연습 비트 넣기", "곡이 없어도 덱 두 개에 연습용 비트를 올립니다.")
        practice.setObjectName("accent")
        practice.clicked.connect(self.practiceRequested.emit)
        clear = _button("목록 지우기", "목록만 지웁니다. 음악 파일은 삭제하지 않습니다.")
        clear.clicked.connect(self.clearRequested.emit)
        self.load_a = _button("덱 A에 넣기", "고른 곡을 왼쪽 덱에 올립니다.")
        self.load_b = _button("덱 B에 넣기", "고른 곡을 오른쪽 덱에 올립니다.")
        self.load_a.clicked.connect(lambda: self._emit_load(0))
        self.load_b.clicked.connect(lambda: self._emit_load(1))
        tools.addWidget(self.search, 1)
        tools.addWidget(folder)
        tools.addWidget(practice)
        tools.addWidget(self.load_a)
        tools.addWidget(self.load_b)
        tools.addWidget(clear)
        layout.addLayout(tools)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["제목", "아티스트", "BPM", "키", "길이"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.cellDoubleClicked.connect(lambda _r, _c: self._emit_load(-1))
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(2, 70)
        self.table.setColumnWidth(3, 90)
        self.table.setColumnWidth(4, 60)
        layout.addWidget(self.table)
        self.status = QLabel("음악 폴더를 추가하거나 연습 비트를 넣으세요.")
        self.status.setObjectName("dim")
        layout.addWidget(self.status)

    def selected_path(self) -> str:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        return item.data(Qt.UserRole) if item else ""

    def _emit_load(self, deck: int) -> None:
        path = self.selected_path()
        if path:
            self.loadRequested.emit(deck, path)

    def upsert(self, meta: dict) -> None:
        path = meta.get("path") or ""
        if not path:
            return
        row = self._row(path)
        self.table.setSortingEnabled(False)
        if row < 0:
            row = self.table.rowCount()
            self.table.insertRow(row)
        values = [
            meta.get("title") or os.path.basename(path),
            meta.get("artist") or "",
            f"{meta['bpm']:.1f}" if meta.get("bpm") else "…",
            _key_text(meta),
            _fmt_short(meta.get("duration") or 0),
        ]
        for col, text in enumerate(values):
            item = QTableWidgetItem(text)
            if col == 0:
                item.setData(Qt.UserRole, path)
            if col == 2:
                item.setData(Qt.UserRole, float(meta.get("bpm") or 0))
            self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)
        self._filter(self.search.text())

    def _row(self, path: str) -> int:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.UserRole) == path:
                return row
        return -1

    def clear_rows(self) -> None:
        self.table.setRowCount(0)

    def _filter(self, text: str) -> None:
        query = (text or "").strip().lower()
        for row in range(self.table.rowCount()):
            blob = " ".join(
                (self.table.item(row, col).text() if self.table.item(row, col) else "")
                for col in (0, 1)
            ).lower()
            self.table.setRowHidden(row, bool(query) and query not in blob)


def _key_text(meta: dict) -> str:
    camelot = meta.get("camelot") or ""
    musical = meta.get("musical_key") or ""
    if camelot and musical:
        return f"{camelot}  {musical}"
    return camelot or musical or "—"


class DeckPanel(QFrame):
    activated = Signal(int)

    def __init__(self, index: int, engine: Engine, letter: str, accent: str):
        super().__init__()
        self.index = index
        self.engine = engine
        self.accent = accent
        self.setObjectName("deck" if index == 0 else "deckB")
        self.scratch_origin = None
        self.jog_mode = "bend"
        self._hot_state = None
        self._loop_state = None
        self._focused = False
        self.setAcceptDrops(True)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        header = QHBoxLayout()
        badge = QLabel(letter)
        badge.setFixedSize(28, 28)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f"background:{accent}; color:#0b0d12; border-radius:14px; font-weight:700;"
        )
        self.title_label = QLabel("곡 없음")
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.key_label = QLabel("—")
        self.key_label.setStyleSheet(
            "background:#111827; border:1px solid #374151; border-radius:8px; padding:2px 8px; font-weight:700;"
        )
        self.time_label = QLabel("0:00.0")
        self.remain_label = QLabel("-0:00.0")
        self.remain_label.setObjectName("dim")
        header.addWidget(badge)
        header.addWidget(self.title_label, 1)
        header.addWidget(self.key_label)
        header.addWidget(self.time_label)
        header.addWidget(self.remain_label)
        root.addLayout(header)

        info = QHBoxLayout()
        self.bpm_label = QLabel("—")
        self.bpm_label.setObjectName("bpm")
        self.percent_label = QLabel("0.0%")
        self.bar_label = QLabel("—.—")
        self.bar_label.setToolTip("마디.박. 1.1 은 첫 마디의 첫 박입니다.")
        half = _button("÷2", "BPM이 두 배로 잡혔으면 절반으로 나눕니다.")
        twice = _button("×2", "BPM이 절반으로 잡혔으면 두 배로 올립니다.")
        grid_l = _button("그리드 ←", "박자 그리드를 살짝 앞으로 옮깁니다.")
        grid_r = _button("그리드 →", "박자 그리드를 살짝 뒤로 옮깁니다.")
        zero = _button("템포 0", "템포를 원래 속도로 되돌리고 싱크를 끕니다.")
        half.clicked.connect(lambda: self._grid(("bpm_scale", 0.5)))
        twice.clicked.connect(lambda: self._grid(("bpm_scale", 2.0)))
        grid_l.clicked.connect(lambda: self._grid(("grid_nudge", -0.01)))
        grid_r.clicked.connect(lambda: self._grid(("grid_nudge", 0.01)))
        zero.clicked.connect(self._reset_tempo)
        self.range_box = QComboBox()
        for label, value in (("±6%", 6), ("±8%", 8), ("±10%", 10), ("±16%", 16), ("WIDE", 50)):
            self.range_box.addItem(label, value)
        self.range_box.setCurrentIndex(1)
        self.range_box.setToolTip("템포 슬라이더가 움직이는 범위입니다. CDJ의 템포 레인저와 같습니다.")
        self.range_box.currentIndexChanged.connect(self._range_changed)
        info.addWidget(self.bpm_label)
        info.addWidget(self.percent_label)
        info.addWidget(self.bar_label)
        info.addWidget(half)
        info.addWidget(twice)
        info.addWidget(grid_l)
        info.addWidget(grid_r)
        info.addWidget(zero)
        info.addWidget(self.range_box)
        info.addStretch(1)
        root.addLayout(info)

        self.overview = WaveformView(overview=True)
        self.detail = WaveformView(overview=False)
        self.overview.seekRequested.connect(self._seek)
        self.detail.seekRequested.connect(self._seek)
        self.overview.empty = "곡을 올려 주세요. 파형을 클릭하면 그 위치로 이동합니다."
        root.addWidget(self.overview)
        root.addWidget(self.detail, 1)

        body = QHBoxLayout()
        self.jog = JogWheel(letter, accent)
        self.jog.pressedAt.connect(self._jog_pressed)
        self.jog.moved.connect(self._jog_moved)
        self.jog.released.connect(self._jog_released)
        self.jog.setToolTip("가운데를 긁으면 스크래치, 가장자리를 밀면 박자를 살짝 밀거나 당깁니다. 멈춘 상태에서는 위치 이동입니다.")
        body.addWidget(self.jog)

        controls = QVBoxLayout()
        controls.setSpacing(4)
        transport = QHBoxLayout()
        self.cue_btn = _button("CUE", "재생 중에는 시작점으로 돌아가 멈춥니다. 멈춘 상태에서 누르면 지금 위치가 시작점이 되고, 누르고 있는 동안만 들립니다.")
        self.cue_btn.setObjectName("cue")
        self.play_btn = _button("PLAY", "재생과 일시정지. 큐를 누르고 있는 동안 PLAY를 누르면 그대로 재생이 이어집니다.")
        self.play_btn.setObjectName("play")
        self.play_btn.setCheckable(True)
        self.sync_btn = _button("SYNC", "MASTER 덱의 BPM과 박자에 맞춥니다. 한 번 더 누르면 풀립니다.", True)
        self.master_btn = _button("MASTER", "이 덱을 기준 박자로 정합니다. 다른 덱의 SYNC가 여기를 따라갑니다.", True)
        self.cue_btn.pressed.connect(lambda: self._cmd(("cue_down",)))
        self.cue_btn.released.connect(lambda: self._cmd(("cue_up",)))
        self.play_btn.clicked.connect(lambda: self._cmd(("toggle",)))
        self.sync_btn.clicked.connect(self._toggle_sync)
        self.master_btn.clicked.connect(self._set_master)
        for button in (self.cue_btn, self.play_btn, self.sync_btn, self.master_btn):
            transport.addWidget(button)
        controls.addLayout(transport)

        modes = QHBoxLayout()
        self.key_btn = _button("키락", "템포를 바꿔도 음정을 유지합니다. 켜 두면 목소리가 얇아지지 않습니다.", True)
        self.vinyl_btn = _button("바이닐", "켜면 조그 가운데를 긁어 스크래치할 수 있습니다.", True)
        self.quant_btn = _button("퀀타이즈", "큐, 루프, 핫큐를 가장 가까운 박자에 붙입니다.", True)
        self.slip_btn = _button("슬립", "루프나 스크래치가 끝나도 원래 흘러가던 위치로 돌아옵니다.", True)
        self.key_btn.clicked.connect(lambda: self._cmd(("keylock", self.key_btn.isChecked())))
        self.vinyl_btn.clicked.connect(lambda: self._cmd(("vinyl", self.vinyl_btn.isChecked())))
        self.quant_btn.clicked.connect(lambda: self._cmd(("quantize", self.quant_btn.isChecked())))
        self.slip_btn.clicked.connect(lambda: self._cmd(("slip", self.slip_btn.isChecked())))
        for button in (self.key_btn, self.vinyl_btn, self.quant_btn, self.slip_btn):
            modes.addWidget(button)
        controls.addLayout(modes)

        cues = QHBoxLayout()
        self.hot_btns = []
        for index in range(8):
            button = _button(str(index + 1), "빈 칸을 누르면 지금 위치를 저장합니다. 다시 누르면 그 위치로 갑니다. 우클릭하면 지웁니다.")
            button.setStyleSheet("min-width: 28px; font-weight: 700;")
            button.clicked.connect(lambda _c=False, i=index: self._hot(i, False))
            button.setContextMenuPolicy(Qt.CustomContextMenu)
            button.customContextMenuRequested.connect(lambda _p, i=index: self._hot(i, True))
            self.hot_btns.append(button)
            cues.addWidget(button)
        controls.addLayout(cues)

        loops = QHBoxLayout()
        self.loop_btns = []
        for label, beats in LOOP_CHOICES:
            button = _button(label, f"지금 위치부터 {label}박을 반복합니다. 같은 버튼을 다시 누르면 루프가 풀립니다.")
            button.clicked.connect(lambda _c=False, value=beats: self._cmd(("loop", value)))
            self.loop_btns.append((beats, button))
            loops.addWidget(button)
        controls.addLayout(loops)

        edits = QHBoxLayout()
        for label, cmd, tip in (
            ("IN", ("loop_in",), "루프 시작점을 지금 위치에 찍습니다."),
            ("OUT", ("loop_out",), "루프 끝점을 찍고 반복을 시작합니다."),
            ("RELOOP", ("reloop",), "저장해 둔 루프의 처음으로 돌아가 다시 반복합니다."),
            ("EXIT", ("loop_exit",), "루프를 풀고 앞으로 진행합니다."),
            ("-4", ("jump", -4), "4박 뒤로 이동합니다."),
            ("-1", ("jump", -1), "1박 뒤로 이동합니다."),
            ("+1", ("jump", 1), "1박 앞으로 이동합니다."),
            ("+4", ("jump", 4), "4박 앞으로 이동합니다."),
        ):
            button = _button(label, tip)
            button.clicked.connect(lambda _c=False, item=cmd: self._cmd(item))
            edits.addWidget(button)
        controls.addLayout(edits)
        body.addLayout(controls, 1)

        tempo_box = QVBoxLayout()
        minus = QLabel("− 느림")
        minus.setAlignment(Qt.AlignCenter)
        minus.setObjectName("dim")
        plus = QLabel("+ 빠름")
        plus.setAlignment(Qt.AlignCenter)
        plus.setObjectName("dim")
        self.tempo = VFader(-1.0, 1.0, 0.0, bipolar=True)
        self.tempo.accent = __import__("PySide6.QtGui", fromlist=["QColor"]).QColor(accent)
        self.tempo.setToolTip("위로 올리면 느려지고 아래로 내리면 빨라집니다. CDJ와 같습니다. 두 번 클릭하면 가운데로 돌아옵니다.")
        self.tempo.valueChanged.connect(self._tempo_changed)
        tempo_box.addWidget(minus)
        tempo_box.addWidget(self.tempo, 1)
        tempo_box.addWidget(plus)
        body.addLayout(tempo_box)
        root.addLayout(body)

    def set_focused(self, focused: bool) -> None:
        if self._focused == focused:
            return
        self._focused = focused
        self.setProperty("focused", "true" if focused else "false")
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:
        self.activated.emit(self.index)
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if is_audio(path):
            self.activated.emit(self.index)
            window = self.window()
            if hasattr(window, "load_path"):
                window.load_path(self.index, path)

    def _touch(self) -> None:
        self.activated.emit(self.index)

    def _cmd(self, cmd) -> None:
        self._touch()
        self.engine.decks[self.index].commands.put(cmd)

    def _grid(self, cmd) -> None:
        self._cmd(cmd)

    def _toggle_sync(self) -> None:
        self._touch()
        deck = self.engine.decks[self.index]
        turning_on = not deck.sync_on
        if turning_on and self.engine.master_index == self.index:
            self.engine.master_index = 1 - self.index
        deck.commands.put(("sync", turning_on))

    def _set_master(self) -> None:
        self._touch()
        self.engine.master_index = self.index
        self.engine.decks[self.index].commands.put(("sync", False))

    def _reset_tempo(self) -> None:
        self._touch()
        deck = self.engine.decks[self.index]
        deck.sync_on = False
        deck.rate = 1.0

    def _range_changed(self) -> None:
        value = float(self.range_box.currentData())
        self.engine.decks[self.index].tempo_range = value

    def _tempo_changed(self, value: float) -> None:
        self._touch()
        deck = self.engine.decks[self.index]
        deck.sync_on = False
        span = max(0.01, deck.tempo_range / 100.0)
        deck.rate = float(max(0.5, min(1.5, 1.0 + value * span)))

    def _seek(self, seconds: float) -> None:
        self._cmd(("seek", max(0.0, seconds)))

    def _hot(self, index: int, delete: bool) -> None:
        self._cmd(("hot", index, delete))

    def _jog_pressed(self, radius: float) -> None:
        self._touch()
        deck = self.engine.decks[self.index]
        self.scratch_origin = deck.pos
        self.jog_mode = "scratch" if (not deck.playing or (deck.vinyl and radius < 0.82)) else "bend"

    def _jog_moved(self, total: float, delta: float, _radius: float) -> None:
        deck = self.engine.decks[self.index]
        if self.jog_mode == "scratch":
            origin = deck.pos if self.scratch_origin is None else self.scratch_origin
            deck.scratch_target = origin + total * 1.8 * SR
            deck.scratching = True
        else:
            bend = max(-0.08, min(0.08, delta * 3.0))
            deck.bend = bend
            deck.bend_hold = True

    def _jog_released(self) -> None:
        deck = self.engine.decks[self.index]
        deck.scratching = False
        deck.bend_hold = False
        self.scratch_origin = None

    def refresh(self, master_index: int, busy: bool) -> None:
        deck = self.engine.decks[self.index]
        if busy:
            self.title_label.setText("불러오는 중…")
        elif deck.error and not deck.loaded:
            self.title_label.setText(deck.error)
        else:
            title = deck.title if deck.loaded else "곡 없음"
            if deck.artist:
                title = f"{deck.artist}  —  {title}"
            width = max(80, self.title_label.width() - 8)
            self.title_label.setText(QFontMetrics(self.title_label.font()).elidedText(title, Qt.ElideRight, width))
        if deck.camelot or deck.musical_key:
            self.key_label.setText(f"{deck.camelot}  {deck.musical_key}".strip())
        else:
            self.key_label.setText("키 —")
        effective = deck.bpm * deck.rate if deck.bpm > 1 else 0
        self.bpm_label.setText(f"{effective:.1f}" if effective else "—")
        self.percent_label.setText(f"{(deck.rate - 1) * 100:+.1f}%")
        self.bar_label.setText(_bar_text(deck))
        pos_sec = deck.pos / SR if deck.loaded else 0
        remain = max(0.0, (deck.duration - pos_sec) / max(0.25, deck.rate)) if deck.loaded else 0
        self.time_label.setText(_fmt_clock(pos_sec))
        self.remain_label.setText("-" + _fmt_clock(remain))
        loop = (deck.loop_start / SR, deck.loop_end / SR) if deck.loop_active else None
        hot = list(deck.hot_cues)
        if deck.revision != getattr(self, "_revision", -1):
            self._revision = deck.revision
            self.overview.set_track(deck.overview, deck.duration, None)
            self.detail.set_track(deck.detail, deck.duration, deck.detail_hz)
        self.overview.set_playhead(pos_sec, deck.bpm, deck.beat_offset, deck.cue, hot, loop)
        self.detail.set_playhead(pos_sec, deck.bpm, deck.beat_offset, deck.cue, hot, loop)
        self.jog.set_angle_from_seconds(pos_sec)
        self._set_checked(self.play_btn, deck.playing)
        self._set_checked(self.sync_btn, deck.sync_on)
        self._set_checked(self.master_btn, master_index == self.index)
        self._set_checked(self.key_btn, deck.keylock)
        self._set_checked(self.vinyl_btn, deck.vinyl)
        self._set_checked(self.quant_btn, deck.quantize)
        self._set_checked(self.slip_btn, deck.slip)
        if not self.tempo.dragging:
            span = max(0.01, deck.tempo_range / 100.0)
            shown = max(-1.0, min(1.0, (deck.rate - 1.0) / span))
            if abs(shown - self.tempo.value) > 0.01:
                self.tempo.set_value(shown, emit=False)
        self._paint_hot(hot)
        self._paint_loop(deck)

    def _set_checked(self, button: QPushButton, checked: bool) -> None:
        if button.isChecked() != checked:
            button.setChecked(checked)

    def _paint_hot(self, hot: list) -> None:
        signature = tuple(hot)
        if signature == self._hot_state:
            return
        self._hot_state = signature
        for index, button in enumerate(self.hot_btns):
            if index < len(hot) and hot[index] is not None:
                button.setStyleSheet(
                    f"min-width:28px; font-weight:700; background:{HOT_COLORS[index]}; color:white;"
                )
            else:
                button.setStyleSheet("min-width:28px; font-weight:700;")

    def _paint_loop(self, deck) -> None:
        active = None
        if deck.loop_active and deck.bpm > 1:
            beats = (deck.loop_end - deck.loop_start) / (SR * 60.0 / deck.bpm)
            for choice, _button in self.loop_btns:
                if abs(choice - beats) < 0.2:
                    active = choice
                    break
        if active == self._loop_state:
            return
        self._loop_state = active
        for choice, button in self.loop_btns:
            button.setChecked(False)
            if choice == active:
                button.setStyleSheet("background:#166534; color:white; font-weight:700;")
            else:
                button.setStyleSheet("")


def _bar_text(deck) -> str:
    if deck.bpm <= 1:
        return "—.—"
    beat = SR * 60.0 / deck.bpm
    index = int((deck.pos - deck.beat_offset * SR) // beat)
    if index < 0:
        return "—.—"
    return f"{index // 4 + 1}.{index % 4 + 1}"


class MixerPanel(QFrame):
    def __init__(self, engine: Engine):
        super().__init__()
        self.engine = engine
        self.setObjectName("panel")
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.addWidget(self._strip(0, "덱 A", "#3b82f6"))
        root.addWidget(self._center(), 1)
        root.addWidget(self._strip(1, "덱 B", "#ff7a18"))

    def _strip(self, index: int, name: str, accent: str) -> QWidget:
        deck = self.engine.decks[index]
        box = QFrame()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(4, 0, 4, 0)
        top = QHBoxLayout()
        label = QLabel(name)
        label.setStyleSheet(f"color:{accent}; font-weight:700;")
        cue = _button("CUE", "이 덱을 큐로 고릅니다. 큐 믹스 노브를 오른쪽으로 돌리면 이 덱만 들립니다.", True)
        cue.clicked.connect(lambda: setattr(deck, "pfl", cue.isChecked()))
        meter = Meter()
        top.addWidget(label)
        top.addWidget(cue)
        top.addStretch(1)
        layout.addLayout(top)
        row = QHBoxLayout()
        specs = (
            ("게인", "trim", 0.5, "입력 크기입니다. 가운데가 0dB, 왼쪽 끝은 거의 무음입니다."),
            ("하이", "eq_high", 0.5, "고음입니다. 왼쪽 끝으로 내리면 하이햇이 사라집니다."),
            ("미드", "eq_mid", 0.5, "중음입니다. 보컬이 주로 여기에 있습니다."),
            ("로우", "eq_low", 0.5, "저음입니다. 킥이 두 곡에서 겹치면 한쪽을 끝까지 내리세요."),
            ("필터", "filter_knob", 0.5, "가운데는 그대로, 왼쪽은 먹먹한 필터, 오른쪽은 얇은 필터입니다."),
        )
        for text, attr, default, tip in specs:
            knob = Knob(text, default)
            knob.accent = __import__("PySide6.QtGui", fromlist=["QColor"]).QColor(accent)
            knob.setToolTip(tip + " 두 번 클릭하면 가운데로 돌아옵니다.")
            knob.valueChanged.connect(lambda value, d=deck, field=attr: setattr(d, field, value))
            row.addWidget(knob)
        fader = VFader(1.0, 0.0, deck.fader)
        fader.accent = __import__("PySide6.QtGui", fromlist=["QColor"]).QColor(accent)
        fader.setToolTip("채널 볼륨입니다. 위가 크고 아래가 작습니다.")
        fader.valueChanged.connect(lambda value, d=deck: setattr(d, "fader", value))
        meter.setToolTip("이 채널의 크기입니다.")
        row.addWidget(fader)
        row.addWidget(meter)
        layout.addLayout(row)
        box.meter = meter
        box.cue = cue
        box.deck_index = index
        self.__dict__[f"strip_{index}"] = box
        return box

    def _center(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(8, 0, 8, 0)
        title = QLabel("크로스페이더")
        title.setAlignment(Qt.AlignCenter)
        labels = QHBoxLayout()
        left = QLabel("A")
        left.setStyleSheet("color:#3b82f6; font-weight:700;")
        right = QLabel("B")
        right.setStyleSheet("color:#ff7a18; font-weight:700;")
        labels.addWidget(left)
        labels.addStretch(1)
        labels.addWidget(right)
        self.cross = HFader()
        self.cross.setToolTip("왼쪽은 덱 A, 오른쪽은 덱 B, 가운데는 둘 다입니다. 두 번 클릭하면 가운데입니다.")
        self.cross.valueChanged.connect(lambda value: setattr(self.engine, "crossfader", value))
        knobs = QHBoxLayout()
        curve = Knob("커브", 0.25)
        curve.setToolTip("크로스페이더가 바뀌는 모양입니다. 오른쪽으로 갈수록 가운데가 짧아집니다.")
        curve.valueChanged.connect(lambda value: setattr(self.engine, "cross_curve", value))
        cue_mix = Knob("큐믹스", 0.0)
        cue_mix.setToolTip("헤드폰 출력이 따로 없을 때 씁니다. 왼쪽은 전체 믹스, 오른쪽은 CUE를 켠 덱만 들립니다.")
        cue_mix.valueChanged.connect(lambda value: setattr(self.engine, "cue_mix", value))
        depth = Knob("FX깊이", 0.0)
        depth.setToolTip("비트 FX의 양입니다. 왼쪽은 효과 없음, 오른쪽으로 갈수록 효과가 커집니다.")
        depth.valueChanged.connect(lambda value: setattr(self.engine, "fx_depth", value))
        master = Knob("마스터", 0.75)
        master.setToolTip("전체 출력 크기입니다. 처음에는 너무 크지 않게 두세요.")
        master.valueChanged.connect(lambda value: setattr(self.engine, "master", value))
        for knob in (curve, cue_mix, depth, master):
            knobs.addWidget(knob)
        fx_row = QHBoxLayout()
        self.fx_kind = QComboBox()
        for label, kind in (
            ("에코", "echo"),
            ("리버브", "reverb"),
            ("플랜저", "flanger"),
            ("페이저", "phaser"),
            ("오토필터", "filter"),
        ):
            self.fx_kind.addItem(label, kind)
        self.fx_kind.setToolTip("마스터에 걸리는 비트 FX입니다. 에코는 박자에 맞춰 반복됩니다.")
        self.fx_kind.currentIndexChanged.connect(
            lambda: setattr(self.engine, "fx_kind", self.fx_kind.currentData())
        )
        self.fx_beats = QComboBox()
        for label, beats in (("1/4", 0.25), ("1/2", 0.5), ("1", 1.0), ("2", 2.0), ("4", 4.0)):
            self.fx_beats.addItem(label + "박", beats)
        self.fx_beats.setCurrentIndex(2)
        self.fx_beats.setToolTip("효과의 길이입니다. 1박은 킥 한 칸입니다.")
        self.fx_beats.currentIndexChanged.connect(
            lambda: setattr(self.engine, "fx_beats", float(self.fx_beats.currentData()))
        )
        self.phase = PhaseMeter()
        self.phase.setToolTip("두 점이 겹치면 박자가 맞은 것입니다. 파란 점이 덱 A, 주황 점이 덱 B입니다.")
        self.master_meter = Meter()
        fx_row.addWidget(QLabel("비트 FX"))
        fx_row.addWidget(self.fx_kind)
        fx_row.addWidget(self.fx_beats)
        fx_row.addWidget(self.phase)
        fx_row.addWidget(self.master_meter)
        layout.addWidget(title)
        layout.addLayout(labels)
        layout.addWidget(self.cross)
        layout.addLayout(knobs)
        layout.addLayout(fx_row)
        return box

    def refresh(self) -> None:
        for index in (0, 1):
            strip = getattr(self, f"strip_{index}")
            deck = self.engine.decks[index]
            strip.meter.set_level(deck.level)
            if strip.cue.isChecked() != deck.pfl:
                strip.cue.setChecked(deck.pfl)
        self.master_meter.set_level(self.engine.master_level)
        phases = []
        for deck in self.engine.decks:
            if deck.bpm > 1 and deck.loaded:
                beat = SR * 60.0 / deck.bpm
                phases.append(((deck.pos - deck.beat_offset * SR) / beat) % 1.0)
            else:
                phases.append(None)
        self.phase.set_phases(phases[0], phases[1])


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("don-dj 사용법")
        self.resize(560, 520)
        layout = QVBoxLayout(self)
        label = QLabel(
            "don-dj는 내 컴퓨터에서만 도는 DJ 연습 프로그램입니다.\n\n"
            "1. 곡이 없으면 위의 연습 비트 넣기를 누릅니다. 왼쪽이 먼저 재생됩니다.\n"
            "2. 오른쪽 PLAY를 누르고 SYNC를 누르면 박자가 맞습니다.\n"
            "3. 크로스페이더를 천천히 오른쪽으로 옮겨 곡을 넘깁니다.\n"
            "4. 킥이 두 개 겹쳐 웅웅거리면 한쪽 로우를 끝까지 내립니다.\n"
            "5. 박자를 직접 맞추려면 SYNC를 끄고, 템포 슬라이더와 조그 가장자리를 씁니다.\n\n"
            "CUE: 재생 중에는 시작점으로 돌아가 멈춥니다. 멈춘 뒤 누르면 그 자리가 새 시작점이고, "
            "누르고 있는 동안만 소리가 납니다.\n"
            "핫 큐 1–8: 빈 칸은 저장, 색이 칠해진 칸은 이동, 우클릭은 삭제.\n"
            "루프 숫자: 그 박자만큼 반복. IN/OUT으로 직접 구간을 찍을 수도 있습니다.\n"
            "키락: 템포를 바꿔도 음정이 유지됩니다.\n"
            "바이닐: 조그 가운데를 긁으면 스크래치가 됩니다.\n"
            "퀀타이즈: 버튼을 박자에 맞춰 줍니다.\n"
            "슬립: 루프나 스크래치가 끝나면 원래 흐르던 위치로 돌아옵니다.\n"
            "큐 믹스: 헤드폰 단자가 따로 없을 때, 오른쪽으로 돌리면 CUE를 켠 덱만 들립니다.\n"
            "템포 슬라이더: 위는 느리게, 아래는 빠르게. 클럽용 CDJ와 같은 방향입니다.\n\n"
            "키보드: A 재생/S 큐는 덱 A, K 재생/L 큐는 덱 B.\n"
            "포커스된 덱은 Space 재생, C 큐, Y 싱크, 숫자 1–8 핫 큐, 좌우 화살표 1박 이동.\n"
            "[ 와 ] 는 크로스페이더를 조금씩 옮깁니다.\n\n"
            "녹음은 마스터 출력을 문서 폴더의 don-dj 안에 wav로 저장합니다."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        close = QPushButton("닫기")
        close.clicked.connect(self.accept)
        layout.addWidget(close)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("don-dj")
        self.engine = Engine()
        self.store = Store()
        self.settings = QSettings("don-dj", "don-dj")
        self.focus_deck = 0
        self.busy = [False, False]
        self.worker = None
        self.notes = queue.SimpleQueue()
        self._build()
        self._restore_library()
        self.engine.start()
        self._fill_devices()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
        self._tick()

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addLayout(self._header())
        self.library = LibraryPanel()
        self.library.loadRequested.connect(self._load_request)
        self.library.folderRequested.connect(self._pick_folder)
        self.library.practiceRequested.connect(self._practice)
        self.library.clearRequested.connect(self._clear_library)
        self.library.setMinimumHeight(132)
        self.library.setMaximumHeight(210)
        layout.addWidget(self.library)
        decks = QHBoxLayout()
        self.deck_a = DeckPanel(0, self.engine, "A", "#3b82f6")
        self.deck_b = DeckPanel(1, self.engine, "B", "#ff7a18")
        self.deck_a.activated.connect(self._focus)
        self.deck_b.activated.connect(self._focus)
        decks.addWidget(self.deck_a, 1)
        decks.addWidget(self.deck_b, 1)
        layout.addLayout(decks, 1)
        self.mixer = MixerPanel(self.engine)
        layout.addWidget(self.mixer)
        self.tip = QLabel("")
        self.tip.setObjectName("tip")
        self.tip.setWordWrap(True)
        layout.addWidget(self.tip)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        self.setAcceptDrops(True)
        self._focus(0)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def _header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        title = QLabel("don-dj")
        title.setStyleSheet("font-size:20px; font-weight:800;")
        sub = QLabel("초보를 위한 로컬 DJ")
        sub.setObjectName("dim")
        self.device_box = QComboBox()
        self.device_box.setMinimumWidth(220)
        self.device_box.currentIndexChanged.connect(self._device_changed)
        self.record_btn = _button("녹음", "마스터 출력을 wav 파일로 저장합니다.")
        self.record_btn.setObjectName("record")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._toggle_record)
        help_btn = _button("사용법", "버튼이 무슨 뜻인지 짧게 설명합니다.")
        help_btn.clicked.connect(lambda: HelpDialog(self).exec())
        open_btn = _button("파일 열기", "곡 파일을 하나 골라 포커스된 덱에 올립니다.")
        open_btn.clicked.connect(self._open_file)
        row.addWidget(title)
        row.addWidget(sub)
        row.addStretch(1)
        row.addWidget(QLabel("출력"))
        row.addWidget(self.device_box)
        row.addWidget(open_btn)
        row.addWidget(self.record_btn)
        row.addWidget(help_btn)
        return row

    def _fill_devices(self) -> None:
        self.device_box.blockSignals(True)
        self.device_box.clear()
        self.device_box.addItem("기본 출력", None)
        for index, name in self.engine.list_devices():
            self.device_box.addItem(name, index)
        self.device_box.blockSignals(False)

    def _device_changed(self) -> None:
        self.engine.open_device(self.device_box.currentData())

    def _restore_library(self) -> None:
        for row in self.store.list_tracks():
            self.library.upsert(row)
        folders = self.settings.value("folders", []) or []
        if isinstance(folders, str):
            folders = [folders]
        if folders:
            self._scan(list(folders))
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def _pick_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "음악 폴더")
        if folder:
            self._remember_folder(folder)
            self._scan([folder])

    def _remember_folder(self, folder: str) -> None:
        folders = self.settings.value("folders", []) or []
        if isinstance(folders, str):
            folders = [folders]
        if folder not in folders:
            folders.append(folder)
        self.settings.setValue("folders", folders)

    def _scan(self, folders: list) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(200)
        self.worker = ScanWorker(folders, self.store)
        self.worker.found.connect(self.library.upsert)
        self.worker.analyzed.connect(self.library.upsert)
        self.worker.status.connect(self.library.status.setText)
        self.worker.start()

    def _load_request(self, deck: int, path: str) -> None:
        target = self.focus_deck if deck < 0 else deck
        self.load_path(target, path)

    def load_path(self, deck: int, path: str) -> None:
        if not path or not os.path.isfile(path):
            return
        self._focus(deck)
        self.busy[deck] = True
        threading.Thread(target=self._load_worker, args=(deck, path), daemon=True).start()

    def _load_worker(self, deck: int, path: str) -> None:
        try:
            track = prepare_file(path, self.store)
            self.engine.enqueue_load(deck, track)
            self.notes.put(("row", _brief(track)))
        except Exception as exc:
            self.engine.decks[deck].error = "이 곡은 열 수 없습니다. " + str(exc)
            self.notes.put(("status", self.engine.decks[deck].error))
        finally:
            self.busy[deck] = False

    def _open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "곡 열기",
            "",
            "오디오 (*.mp3 *.wav *.flac *.ogg *.aiff *.aif *.m4a)",
        )
        if path:
            self.load_path(self.focus_deck, path)

    def _practice(self) -> None:
        self.engine.load_practice()
        self._focus(0)

    def _clear_library(self) -> None:
        answer = QMessageBox.question(
            self,
            "목록 지우기",
            "라이브러리 목록을 지울까요? 음악 파일은 삭제되지 않습니다.",
        )
        if answer != QMessageBox.Yes:
            return
        self.store.clear()
        self.library.clear_rows()
        self.settings.setValue("folders", [])

    def _focus(self, index: int) -> None:
        self.focus_deck = index
        self.deck_a.set_focused(index == 0)
        self.deck_b.set_focused(index == 1)

    def _toggle_record(self) -> None:
        if self.record_btn.isChecked():
            self.engine.start_recording()
            return
        path = self.engine.stop_recording()
        note = self.engine.record_note
        if path:
            QMessageBox.information(self, "녹음", (note + "\n" if note else "") + path)
        else:
            QMessageBox.information(self, "녹음", "저장할 소리가 없었습니다.")

    def _nudge_cross(self, delta: float) -> None:
        self.mixer.cross.set_value(self.engine.crossfader + delta, emit=True)

    def _tick(self) -> None:
        self._drain_notes()
        self.deck_a.refresh(self.engine.master_index, self.busy[0])
        self.deck_b.refresh(self.engine.master_index, self.busy[1])
        self.mixer.refresh()
        self._persist_edits()
        self.tip.setText(self._tip_text())
        recording = self.engine.recording
        if self.record_btn.isChecked() != recording:
            self.record_btn.setChecked(recording)
        self.record_btn.setText("녹음 중" if recording else "녹음")

    def _drain_notes(self) -> None:
        while True:
            try:
                kind, payload = self.notes.get_nowait()
            except queue.Empty:
                return
            if kind == "row":
                self.library.upsert(payload)
            elif kind == "status":
                self.library.status.setText(str(payload))

    def _persist_edits(self) -> None:
        for deck in self.engine.decks:
            if not deck.path or deck.path.startswith("practice:"):
                deck.cues_dirty = False
                deck.grid_dirty = False
                continue
            if deck.cues_dirty:
                hot = list(deck.hot_cues)
                cue = deck.cue
                deck.cues_dirty = False
                self.store.update_cues(deck.path, cue, hot)
            if deck.grid_dirty:
                bpm = deck.bpm
                offset = deck.beat_offset
                deck.grid_dirty = False
                self.store.update_grid(deck.path, bpm, offset)
                self.library.upsert(
                    {
                        "path": deck.path,
                        "title": deck.title,
                        "artist": deck.artist,
                        "duration": deck.duration,
                        "bpm": bpm,
                        "musical_key": deck.musical_key,
                        "camelot": deck.camelot,
                    }
                )

    def _tip_text(self) -> str:
        if self.engine.device_error:
            return self.engine.device_error
        if self.engine.record_note and not self.engine.recording:
            return self.engine.record_note
        left, right = self.engine.decks
        if not left.loaded and not right.loaded:
            return "처음이라면 위의 연습 비트 넣기를 누르세요. 가진 곡이 있으면 음악 폴더 추가를 누르세요."
        if left.loaded and not right.loaded:
            return "덱 A에 곡이 있습니다. 다른 곡을 골라 덱 B에 넣기를 누르세요."
        if right.loaded and not left.loaded:
            return "덱 B에 곡이 있습니다. 덱 A에도 한 곡 올려 보세요."
        if left.loaded and right.loaded and not left.playing and not right.playing:
            return "두 덱이 준비됐습니다. 왼쪽 PLAY로 먼저 들어 보세요. CUE는 시작점입니다."
        if left.playing and not right.playing:
            return "오른쪽 PLAY를 누른 다음 SYNC를 누르면 덱 B의 박자가 덱 A에 맞습니다."
        if right.playing and not left.playing:
            return "왼쪽도 재생해 보세요. 기준 덱의 MASTER가 켜져 있어야 SYNC가 그 박자를 따라갑니다."
        if left.playing and right.playing and not right.sync_on and not left.sync_on:
            if left.bpm > 1 and right.bpm > 1 and abs(left.bpm * left.rate - right.bpm * right.rate) > 0.6:
                return "BPM 숫자가 다릅니다. 맞추고 싶은 덱에서 SYNC를 누르거나, 템포 슬라이더로 숫자를 맞추세요."
        if left.playing and right.playing and self._aligned():
            return "박자가 맞습니다. 크로스페이더를 천천히 옮기세요. 킥이 겹치면 한쪽 로우를 끝까지 내리세요."
        if left.playing and right.playing:
            return "조그 가장자리를 살짝 밀면 박자가 앞뒤로 움직입니다. 페이즈의 두 점이 겹치면 맞춘 것입니다."
        return "파형을 클릭하면 그 위치로 이동합니다. 핫 큐는 빈 칸을 눌러 저장하고, 다시 눌러 돌아갑니다."

    def _aligned(self) -> bool:
        phases = []
        bpms = []
        for deck in self.engine.decks:
            if deck.bpm <= 1 or not deck.playing:
                return False
            beat = SR * 60.0 / deck.bpm
            phases.append(((deck.pos - deck.beat_offset * SR) / beat) % 1.0)
            bpms.append(deck.bpm * deck.rate)
        if abs(bpms[0] - bpms[1]) > 0.4:
            return False
        err = abs((phases[0] - phases[1] + 0.5) % 1.0 - 0.5)
        return err < 0.08

    def eventFilter(self, obj, event):
        if QApplication.activeWindow() is not self or self._typing():
            return super().eventFilter(obj, event)
        if event.type() == QEvent.KeyPress and not event.isAutoRepeat() and self._handle_key(event, True):
            return True
        if event.type() == QEvent.KeyRelease and not event.isAutoRepeat() and self._handle_key(event, False):
            return True
        return super().eventFilter(obj, event)

    def _handle_key(self, event, pressed: bool) -> bool:
        key = event.key()
        focused = self.focus_deck
        shift = bool(event.modifiers() & Qt.ShiftModifier)
        if pressed and key == Qt.Key_A:
            self.engine.decks[0].commands.put(("toggle",))
        elif key == Qt.Key_S:
            self.engine.decks[0].commands.put(("cue_down",) if pressed else ("cue_up",))
        elif pressed and key == Qt.Key_K:
            self.engine.decks[1].commands.put(("toggle",))
        elif key == Qt.Key_L:
            self.engine.decks[1].commands.put(("cue_down",) if pressed else ("cue_up",))
        elif pressed and key == Qt.Key_Space:
            self.engine.decks[focused].commands.put(("toggle",))
        elif key == Qt.Key_C:
            self.engine.decks[focused].commands.put(("cue_down",) if pressed else ("cue_up",))
        elif pressed and key == Qt.Key_Y:
            panel = self.deck_a if focused == 0 else self.deck_b
            panel._toggle_sync()
        elif pressed and key == Qt.Key_Left:
            self.engine.decks[focused].commands.put(("jump", -4 if shift else -1))
        elif pressed and key == Qt.Key_Right:
            self.engine.decks[focused].commands.put(("jump", 4 if shift else 1))
        elif pressed and key == Qt.Key_BracketLeft:
            self._nudge_cross(-0.05)
        elif pressed and key == Qt.Key_BracketRight:
            self._nudge_cross(0.05)
        elif pressed and Qt.Key_1 <= key <= Qt.Key_8:
            self.engine.decks[focused].commands.put(("hot", key - Qt.Key_1, bool(event.modifiers() & Qt.ControlModifier)))
        else:
            return False
        return True

    def _typing(self) -> bool:
        return self.library.search.hasFocus()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isdir(path):
                self._remember_folder(path)
                self._scan([path])
            elif is_audio(path):
                self.load_path(self.focus_deck, path)

    def closeEvent(self, event) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        if self.engine.recording:
            self.engine.stop_recording()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(300)
        self.engine.stop()
        super().closeEvent(event)


def run() -> None:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    app.setFont(QFont("Malgun Gothic", 10))
    window = MainWindow()
    screen = app.primaryScreen()
    if screen is not None:
        area = screen.availableGeometry()
        window.resize(min(area.width() - 40, 1440), min(area.height() - 40, 900))
    window.show()
    app.exec()
