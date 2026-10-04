"""Local library cache. Music files stay where the user put them."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path

import numpy as np


def app_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    path = Path(base) / "don-dj"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pack(arr: np.ndarray | None) -> bytes | None:
    if arr is None:
        return None
    return np.ascontiguousarray(arr, dtype=np.float32).tobytes()


def _unpack(blob, cols: int) -> np.ndarray | None:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size % 4 != 0:
        return None
    view = arr.reshape(-1, 4).copy()
    if cols and len(view) != cols:
        return view
    return view


class Store:
    def __init__(self):
        self.path = app_dir() / "library.db"
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    path TEXT PRIMARY KEY,
                    mtime REAL,
                    title TEXT,
                    artist TEXT,
                    duration REAL,
                    bpm REAL,
                    beat_offset REAL,
                    musical_key TEXT,
                    camelot TEXT,
                    overview BLOB,
                    detail BLOB,
                    detail_hz REAL,
                    cues TEXT,
                    fail TEXT
                )
                """
            )
            self.conn.commit()

    def has_analysis(self, path: str, mtime: float) -> bool:
        with self.lock:
            row = self.conn.execute(
                "SELECT mtime, bpm, overview IS NOT NULL AS ready FROM tracks WHERE path = ?",
                (path,),
            ).fetchone()
        if row is None or not row["ready"] or not row["bpm"]:
            return False
        return abs((row["mtime"] or 0) - mtime) < 0.01

    def list_tracks(self) -> list:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT path, mtime, title, artist, duration, bpm, beat_offset,
                       musical_key, camelot, cues, fail
                FROM tracks
                ORDER BY artist, title
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, path: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM tracks WHERE path = ?", (path,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["overview"] = _unpack(data.get("overview"), 0)
        data["detail"] = _unpack(data.get("detail"), 0)
        data["cues"] = _cues(data.get("cues"))
        return data

    def upsert_tags(self, tags: dict) -> None:
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO tracks (path, mtime, title, artist, duration)
                VALUES (:path, :mtime, :title, :artist, :duration)
                ON CONFLICT(path) DO UPDATE SET
                    mtime = excluded.mtime,
                    title = excluded.title,
                    artist = excluded.artist,
                    duration = excluded.duration
                """,
                tags,
            )
            self.conn.commit()

    def upsert_analysis(self, meta: dict) -> None:
        payload = {
            "path": meta["path"],
            "mtime": meta.get("mtime") or 0,
            "title": meta.get("title") or "",
            "artist": meta.get("artist") or "",
            "duration": meta.get("duration") or 0,
            "bpm": meta.get("bpm") or 0,
            "beat_offset": meta.get("beat_offset") or 0,
            "musical_key": meta.get("musical_key") or "",
            "camelot": meta.get("camelot") or "",
            "overview": _pack(meta.get("overview")),
            "detail": _pack(meta.get("detail")),
            "detail_hz": meta.get("detail_hz") or 40,
            "fail": meta.get("fail") or "",
        }
        with self.lock:
            self.conn.execute(
                """
                INSERT INTO tracks (
                    path, mtime, title, artist, duration, bpm, beat_offset,
                    musical_key, camelot, overview, detail, detail_hz, fail
                ) VALUES (
                    :path, :mtime, :title, :artist, :duration, :bpm, :beat_offset,
                    :musical_key, :camelot, :overview, :detail, :detail_hz, :fail
                )
                ON CONFLICT(path) DO UPDATE SET
                    mtime = excluded.mtime,
                    title = excluded.title,
                    artist = excluded.artist,
                    duration = excluded.duration,
                    bpm = excluded.bpm,
                    beat_offset = excluded.beat_offset,
                    musical_key = excluded.musical_key,
                    camelot = excluded.camelot,
                    overview = excluded.overview,
                    detail = excluded.detail,
                    detail_hz = excluded.detail_hz,
                    fail = excluded.fail
                """,
                payload,
            )
            self.conn.commit()

    def update_cues(self, path: str, cue: float, hot: list) -> None:
        blob = json.dumps({"cue": cue, "hot": hot})
        with self.lock:
            self.conn.execute("UPDATE tracks SET cues = ? WHERE path = ?", (blob, path))
            self.conn.commit()

    def update_grid(self, path: str, bpm: float, beat_offset: float) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE tracks SET bpm = ?, beat_offset = ? WHERE path = ?",
                (bpm, beat_offset, path),
            )
            self.conn.commit()

    def clear(self) -> None:
        with self.lock:
            self.conn.execute("DELETE FROM tracks")
            self.conn.commit()


def _cues(raw) -> dict:
    empty = {"cue": 0.0, "hot": [None] * 8}
    if not raw:
        return empty
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return empty
    hot = list(data.get("hot") or [])
    while len(hot) < 8:
        hot.append(None)
    return {"cue": float(data.get("cue") or 0.0), "hot": hot[:8]}
