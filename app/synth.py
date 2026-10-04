"""Short house loops so a beginner can practice with no music files."""
from __future__ import annotations

import numpy as np

from app.analyze import build_waveform
from app.dsp import SR


def _kick(sr: int) -> np.ndarray:
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    freq = 150.0 * np.exp(-t * 30.0) + 48.0
    phase = 2 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * np.exp(-t * 11.0)
    click_n = int(0.004 * sr)
    click = np.random.uniform(-1, 1, click_n).astype(np.float32)
    click *= np.linspace(1.0, 0.0, click_n).astype(np.float32)
    body[:click_n] += click * 0.35
    return body.astype(np.float32)


def _hat(sr: int, open_hat: bool) -> np.ndarray:
    n = int((0.11 if open_hat else 0.035) * sr)
    noise = np.random.randn(n).astype(np.float32)
    noise = np.diff(noise, prepend=noise[:1])
    t = np.arange(n) / sr
    amp = np.exp(-t * (16.0 if open_hat else 60.0))
    return (noise * amp * 0.28).astype(np.float32)


def _clap(sr: int) -> np.ndarray:
    n = int(0.16 * sr)
    y = np.zeros(n, np.float32)
    for delay in (0.0, 0.008, 0.015):
        burst_n = int(0.07 * sr)
        burst = np.random.randn(burst_n).astype(np.float32)
        burst *= np.exp(-np.arange(burst_n) / (sr * 0.028)).astype(np.float32)
        i = int(delay * sr)
        y[i:i + burst_n] += burst * 0.45
    return y


def _bass(sr: int, freq: float, dur: float) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = np.sin(2 * np.pi * freq * t) + 0.28 * np.sin(2 * np.pi * freq * 2 * t)
    amp = np.exp(-t * 4.2) * (1.0 - np.exp(-t * 250.0))
    return (y * amp * 0.42).astype(np.float32)


def _stab(sr: int, freqs, dur: float = 0.18) -> np.ndarray:
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs) / float(len(freqs))
    amp = np.exp(-t * 9.0) * (1.0 - np.exp(-t * 400.0))
    return (y * amp * 0.2).astype(np.float32)


def _place(dest: np.ndarray, sound: np.ndarray, at: int, gain: float) -> None:
    if at >= len(dest) or at < 0:
        return
    take = min(len(sound), len(dest) - at)
    dest[at:at + take] += sound[:take] * gain


def make_loop(bpm: float, root: float, stab_freqs, hat_sixteenths: bool) -> np.ndarray:
    sr = SR
    bars = 8
    beat = 60.0 / bpm
    beats = bars * 4
    n = int(round(beats * beat * sr))
    mix = np.zeros(n, np.float32)
    hats = np.zeros(n, np.float32)
    kick = _kick(sr)
    hat = _hat(sr, False)
    open_hat = _hat(sr, True)
    clap = _clap(sr)
    for i in range(beats):
        t = i * beat
        _place(mix, kick, int(round(t * sr)), 0.95)
        if i % 4 in (1, 3):
            _place(mix, clap, int(round(t * sr)), 0.42)
        if i % 2 == 1:
            _place(mix, _bass(sr, root, beat * 0.85), int(round(t * sr)), 1.0)
        if i % 8 == 0:
            _place(mix, _stab(sr, stab_freqs), int(round(t * sr)), 1.0)
        if hat_sixteenths:
            for sub in range(4):
                ht = t + sub * beat / 4.0
                _place(hats, hat, int(round(ht * sr)), 0.45 if sub % 2 == 0 else 0.28)
        else:
            _place(hats, hat, int(round(t * sr)), 0.55)
            _place(hats, hat, int(round((t + beat * 0.5) * sr)), 0.4)
            if i % 2 == 1:
                _place(hats, open_hat, int(round((t + beat * 0.5) * sr)), 0.55)
    # Keep the downbeat kick, and leave the seam quiet aside from that.
    stereo_l = mix + hats
    delay = int(0.004 * sr)
    stereo_r = mix + np.concatenate((np.zeros(delay, np.float32), hats[:-delay] if delay else hats))
    audio = np.column_stack((stereo_l, stereo_r)).astype(np.float32)
    peak = float(np.max(np.abs(audio))) or 1.0
    return np.ascontiguousarray(audio / peak * 0.9, dtype=np.float32)


def _track(samples, path, title, bpm, musical, camelot):
    overview, detail, detail_hz = build_waveform(samples, SR)
    return {
        "path": path,
        "title": title,
        "artist": "연습",
        "samples": samples,
        "bpm": float(bpm),
        "beat_offset": 0.0,
        "musical_key": musical,
        "camelot": camelot,
        "overview": overview,
        "detail": detail,
        "detail_hz": detail_hz,
        "duration": len(samples) / SR,
        "hot_cues": [None] * 8,
        "cue": 0.0,
        "auto_loop": True,
        "auto_play": False,
    }


def make_practice_tracks():
    a = make_loop(120.0, 110.0, (220.0, 261.63, 329.63), hat_sixteenths=False)
    b = make_loop(124.0, 82.41, (329.63, 392.0, 493.88), hat_sixteenths=True)
    left = _track(a, "practice://warm-floor", "Warm Floor", 120.0, "Am", "8A")
    right = _track(b, "practice://soft-room", "Soft Room", 124.0, "Em", "9A")
    left["auto_play"] = True
    return left, right
