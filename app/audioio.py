"""Decode local audio files and read their tags."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from app.analyze import analyze_samples
from app.dsp import SR

AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".aiff", ".aif", ".aac", ".m4a", ".wma"}


def is_audio(path: str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXT


def read_tags(path: str) -> dict:
    stem = Path(path).stem
    artist, title = "", stem
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
    duration = 0.0
    try:
        from mutagen import File as MutagenFile

        info = MutagenFile(path, easy=True)
        if info is not None:
            if info.get("title"):
                title = str(info.get("title")[0])
            if info.get("artist"):
                artist = str(info.get("artist")[0])
            length = getattr(info.info, "length", None)
            if length:
                duration = float(length)
    except Exception:
        pass
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return {
        "path": path,
        "mtime": mtime,
        "title": title or stem,
        "artist": artist,
        "duration": duration,
    }


def decode(path: str) -> np.ndarray:
    import miniaudio

    decoded = miniaudio.decode_file(
        path,
        nchannels=2,
        sample_rate=SR,
        output_format=miniaudio.SampleFormat.FLOAT32,
    )
    arr = np.array(decoded.samples, dtype=np.float32, copy=True)
    channels = int(getattr(decoded, "nchannels", 2) or 2)
    if arr.ndim == 1:
        if channels <= 1:
            arr = np.column_stack((arr, arr))
        else:
            frames = arr.size // channels
            arr = arr[: frames * channels].reshape(frames, channels)
    if arr.ndim != 2:
        raise RuntimeError("이 파일의 소리를 읽지 못했습니다.")
    if arr.shape[1] == 1:
        arr = np.column_stack((arr[:, 0], arr[:, 0]))
    elif arr.shape[1] > 2:
        arr = arr[:, :2]
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 2.0:
        arr = arr / 32768.0
    if len(arr) < SR // 4:
        raise RuntimeError("파일이 너무 짧습니다.")
    return np.ascontiguousarray(arr, dtype=np.float32)


def prepare_file(path: str, store) -> dict:
    tags = read_tags(path)
    samples = decode(path)
    cached = store.get(path)
    fresh = (
        cached
        and abs((cached.get("mtime") or 0) - tags["mtime"]) < 0.01
        and cached.get("overview") is not None
        and (cached.get("bpm") or 0) > 0
    )
    if fresh:
        meta = cached
        meta["title"] = tags["title"]
        meta["artist"] = tags["artist"]
    else:
        meta = analyze_samples(samples, SR)
        meta.update(tags)
        meta["duration"] = len(samples) / SR
        if cached and cached.get("cues"):
            meta_cues = cached["cues"]
        else:
            meta_cues = {"cue": 0.0, "hot": [None] * 8}
        store.upsert_analysis(meta)
        meta = store.get(path) or meta
        meta["cues"] = meta_cues if not meta.get("cues") else meta["cues"]
    cues = (cached or meta).get("cues") or {"cue": 0.0, "hot": [None] * 8}
    if isinstance(cues, str):
        cues = {"cue": 0.0, "hot": [None] * 8}
    return {
        "path": path,
        "title": meta.get("title") or tags["title"],
        "artist": meta.get("artist") or tags["artist"],
        "samples": samples,
        "bpm": float(meta.get("bpm") or 0),
        "beat_offset": float(meta.get("beat_offset") or 0),
        "musical_key": meta.get("musical_key") or "",
        "camelot": meta.get("camelot") or "",
        "overview": meta.get("overview"),
        "detail": meta.get("detail"),
        "detail_hz": float(meta.get("detail_hz") or 40),
        "duration": float(meta.get("duration") or len(samples) / SR),
        "hot_cues": list(cues.get("hot") or [None] * 8),
        "cue": float(cues.get("cue") or 0.0),
        "auto_loop": False,
        "auto_play": False,
    }
