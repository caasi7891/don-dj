"""BPM, key, and waveform measurements for a decoded track."""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter, stft

from app.dsp import SR

MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Camelot number+letter for pitch class 0=C.
CAMELOT_MAJOR = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]
CAMELOT_MINOR = ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]


def camelot_distance(a: str, b: str) -> int | None:
    if not a or not b or len(a) < 2 or len(b) < 2:
        return None
    try:
        na, la = int(a[:-1]), a[-1]
        nb, lb = int(b[:-1]), b[-1]
    except ValueError:
        return None
    if la == lb:
        diff = abs(na - nb) % 12
        return min(diff, 12 - diff)
    if na == nb:
        return 1
    return 3


def _one_pole(x: np.ndarray, cutoff: float, sr: float) -> np.ndarray:
    cutoff = float(np.clip(cutoff, 10.0, sr * 0.45))
    coeff = math.exp(-2 * math.pi * cutoff / sr)
    return lfilter([1.0 - coeff], [1.0, -coeff], x).astype(np.float32)


def _reduce(sig: np.ndarray, cols: int) -> np.ndarray:
    cols = max(1, int(cols))
    n = len(sig)
    if n == 0:
        return np.zeros(cols, np.float32)
    chunk = int(math.ceil(n / cols))
    pad = chunk * cols - n
    frames = np.pad(sig, (0, pad)).reshape(cols, chunk)
    return np.max(np.abs(frames), axis=1).astype(np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    peak = float(np.percentile(v, 98))
    if peak < 1e-6:
        peak = float(np.max(v))
    if peak < 1e-8:
        return np.zeros_like(v)
    return np.clip(v / peak, 0.0, 1.0).astype(np.float32)


def build_waveform(samples: np.ndarray, sr: int = SR) -> tuple:
    mono = samples.mean(axis=1).astype(np.float32) if samples.ndim == 2 else samples.astype(np.float32)
    duration = len(mono) / float(sr)
    step = 4 if sr >= 40000 else 1
    mono_ds = mono[::step]
    sr_ds = sr / step
    low = _one_pole(mono_ds, 180.0, sr_ds)
    high_rest = mono_ds - _one_pole(mono_ds, 2800.0, sr_ds)
    mid = mono_ds - low - high_rest

    overview_cols = int(np.clip(len(mono_ds) / 400, 240, 1400))
    detail_hz = 40.0
    detail_cols = int(duration * detail_hz)
    if detail_cols > 24000 and duration > 0:
        detail_hz = 24000.0 / duration
        detail_cols = 24000
    detail_cols = max(8, detail_cols)

    def pack(cols: int) -> np.ndarray:
        amp = _normalize(_reduce(mono_ds, cols))
        lo = _normalize(_reduce(low, cols))
        md = _normalize(_reduce(mid, cols))
        hi = _normalize(_reduce(high_rest, cols))
        return np.column_stack((amp, lo, md, hi)).astype(np.float32)

    return pack(overview_cols), pack(detail_cols), float(detail_hz)


def _autocorr(onset: np.ndarray) -> np.ndarray:
    nfft = 1 << int(math.ceil(math.log2(len(onset) * 2)))
    spec = np.fft.rfft(onset, nfft)
    ac = np.fft.irfft(spec * np.conj(spec), nfft)
    ac = ac[: len(onset)].astype(np.float64)
    if ac[0] > 0:
        ac /= ac[0]
    ac[0] = 0.0
    return ac


def detect_bpm(samples: np.ndarray, sr: int = SR) -> tuple:
    mono = samples.mean(axis=1).astype(np.float32) if samples.ndim == 2 else samples.astype(np.float32)
    if len(mono) < sr:
        return 0.0, 0.0
    # Long tracks often open with a quiet intro. Short loops must be used whole.
    if len(mono) > sr * 90:
        mono = mono[int(sr * 20): int(sr * 120)]
    if np.max(np.abs(mono)) < 1e-5:
        return 0.0, 0.0
    hop = 512
    n = len(mono) // hop * hop
    if n < hop * 8:
        return 0.0, 0.0
    frames = mono[:n].reshape(-1, hop)
    env = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    diff = np.diff(env, prepend=env[:1])
    onset = np.maximum(diff, 0.0)
    peak = float(onset.max())
    if peak < 1e-8:
        return 0.0, 0.0
    onset = onset / peak
    ac = _autocorr(onset)
    env_sr = sr / hop
    min_bpm, max_bpm = 72.0, 180.0
    min_lag = max(1, int(env_sr * 60.0 / max_bpm))
    max_lag = min(len(ac) - 2, int(env_sr * 60.0 / min_bpm))
    if max_lag <= min_lag + 2:
        return 0.0, 0.0
    k = int(np.argmax(ac[min_lag:max_lag]) + min_lag)
    if 1 <= k < len(ac) - 1:
        a, b, c = ac[k - 1], ac[k], ac[k + 1]
        denom = a - 2 * b + c
        if abs(denom) > 1e-9:
            k = k + 0.5 * (a - c) / denom
    bpm = 60.0 * env_sr / k

    def score(candidate: float) -> float:
        lag = 60.0 * env_sr / candidate
        i = int(round(lag))
        if i < 1 or i >= len(ac):
            return -1.0
        value = ac[i]
        if i * 2 < len(ac):
            value += 0.55 * ac[i * 2]
        if 96 <= candidate <= 140:
            value *= 1.12
        return float(value)

    candidates = [bpm]
    if 72 <= bpm / 2 <= 180:
        candidates.append(bpm / 2)
    if 72 <= bpm * 2 <= 180:
        candidates.append(bpm * 2)
    bpm = max(candidates, key=score)

    beat_hops = env_sr * 60.0 / bpm
    best_phase, best_s = 0.0, -1.0
    grid = np.arange(0.0, len(onset) - 1, beat_hops)
    for phase in np.linspace(0.0, beat_hops, 48, endpoint=False):
        pos = (grid + phase).astype(int)
        pos = pos[(pos >= 0) & (pos < len(onset))]
        if len(pos) == 0:
            continue
        value = float(onset[pos].sum())
        if value > best_s:
            best_s = value
            best_phase = float(phase)
    return float(bpm), float(best_phase / env_sr)


def detect_key(samples: np.ndarray, sr: int = SR) -> tuple:
    mono = samples.mean(axis=1).astype(np.float32) if samples.ndim == 2 else samples.astype(np.float32)
    if len(mono) > sr * 45:
        mono = mono[: sr * 45]
    if len(mono) < sr or np.max(np.abs(mono)) < 1e-5:
        return "", "", 0.0
    _f, _t, zxx = stft(mono, sr, nperseg=4096, noverlap=2048, boundary=None)
    mag = np.abs(zxx)
    freqs = np.fft.rfftfreq(4096, 1.0 / sr)
    chroma = np.zeros(12, np.float64)
    for index in range(1, min(len(freqs), mag.shape[0])):
        freq = freqs[index]
        if freq < 50 or freq > 5000:
            continue
        midi = 69 + 12 * math.log2(freq / 440.0)
        pc = int(round(midi)) % 12
        chroma[pc] += float(mag[index].sum())
    total = float(chroma.sum())
    if total < 1e-6:
        return "", "", 0.0
    chroma /= total
    best_name, best_mode, best_score = "", "major", -2.0
    best_pc = 0
    for pc in range(12):
        major_score = _corr(chroma, np.roll(MAJOR, pc))
        minor_score = _corr(chroma, np.roll(MINOR, pc))
        if major_score > best_score:
            best_score = major_score
            best_name = NAMES[pc]
            best_mode = "major"
            best_pc = pc
        if minor_score > best_score:
            best_score = minor_score
            best_name = NAMES[pc] + "m"
            best_mode = "minor"
            best_pc = pc
    if best_score < 0.25:
        return "", "", float(best_score)
    camelot = CAMELOT_MAJOR[best_pc] if best_mode == "major" else CAMELOT_MINOR[best_pc]
    return best_name, camelot, float(best_score)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - a.mean()
    bb = b - b.mean()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(aa, bb) / denom)


def analyze_samples(samples: np.ndarray, sr: int = SR) -> dict:
    bpm, offset = detect_bpm(samples, sr)
    musical, camelot, score = detect_key(samples, sr)
    overview, detail, detail_hz = build_waveform(samples, sr)
    return {
        "bpm": bpm,
        "beat_offset": offset,
        "musical_key": musical,
        "camelot": camelot,
        "key_score": score,
        "overview": overview,
        "detail": detail,
        "detail_hz": detail_hz,
        "duration": len(samples) / float(sr),
    }
