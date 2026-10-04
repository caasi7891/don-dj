"""Resampling, time-stretch, channel EQ, and beat effects."""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

SR = 44100
BLOCK = 256


def eq_gain(value: float) -> float:
    """Map a 0..1 knob to linear gain. Center is unity, bottom is a kill."""
    v = float(np.clip(value, 0.0, 1.0))
    if v < 0.02:
        return 0.0
    if v >= 0.5:
        db = (v - 0.5) / 0.5 * 6.0
    else:
        db = (v - 0.5) / 0.5 * 36.0
    return float(10 ** (db / 20.0))


def trim_gain(value: float) -> float:
    v = float(np.clip(value, 0.0, 1.0))
    if v < 0.01:
        return 0.0
    if v >= 0.5:
        db = (v - 0.5) / 0.5 * 6.0
    else:
        db = (v - 0.5) / 0.5 * 24.0
    return float(10 ** (db / 20.0))


def fader_gain(value: float) -> float:
    v = float(np.clip(value, 0.0, 1.0))
    return float(v ** 1.55)


def cross_gains(position: float, sharpness: float) -> tuple:
    """Constant-power crossfader. 0 is full A, 1 is full B."""
    x = float(np.clip(position, 0.0, 1.0))
    sharp = float(np.clip(sharpness, 0.0, 1.0))
    if sharp >= 0.98:
        if x <= 0.06:
            return 1.0, 0.0
        if x >= 0.94:
            return 0.0, 1.0
    left = math.cos(x * math.pi * 0.5)
    right = math.sin(x * math.pi * 0.5)
    power = 1.0 + sharp * 2.4
    return left ** power, right ** power


def _interp(samples: np.ndarray, idx: np.ndarray) -> np.ndarray:
    n = len(samples)
    out_n = len(idx)
    if n < 2 or out_n == 0:
        return np.zeros((out_n, 2), np.float32)
    i0 = np.floor(idx).astype(np.int64)
    frac = (idx - i0).astype(np.float32)[:, None]
    i0 = np.clip(i0, 0, n - 2)
    i1 = i0 + 1
    return (samples[i0] * (1.0 - frac) + samples[i1] * frac).astype(np.float32)


def _interp_loop(samples: np.ndarray, idx: np.ndarray, start: int, end: int) -> np.ndarray:
    n = len(samples)
    out_n = len(idx)
    if n < 2 or out_n == 0:
        return np.zeros((out_n, 2), np.float32)
    start = int(np.clip(start, 0, n - 2))
    end = int(np.clip(end, start + 2, n))
    length = end - start
    rel = np.mod(idx - start, length)
    base = start + rel
    i0f = np.floor(base)
    frac = (base - i0f).astype(np.float32)[:, None]
    i0 = start + np.mod(i0f.astype(np.int64) - start, length)
    i1 = start + np.mod(i0 - start + 1, length)
    return (samples[i0] * (1.0 - frac) + samples[i1] * frac).astype(np.float32)


def render_resample(samples, pos: float, n: int, rate: float, loop):
    idx = pos + np.arange(n, dtype=np.float64) * rate
    if loop is None:
        audio = _interp(samples, idx)
        return audio, pos + n * rate
    start, end = loop
    audio = _interp_loop(samples, idx, int(start), int(end))
    length = float(end - start)
    if length <= 1:
        return audio, pos
    new_pos = start + (pos + n * rate - start) % length
    return audio, float(new_pos)


def render_scratch(samples, pos: float, target: float, n: int):
    delta = float(target - pos)
    if n <= 0 or abs(delta) < 0.75:
        return np.zeros((max(n, 0), 2), np.float32), float(target)
    rate = delta / n
    idx = pos + np.arange(n, dtype=np.float64) * rate
    return _interp(samples, idx), float(target)


def read_span(samples: np.ndarray, start: int, count: int, loop) -> np.ndarray:
    out = np.zeros((count, 2), np.float32)
    n = len(samples)
    if n <= 0 or count <= 0:
        return out
    if loop is None:
        if start < 0:
            pad = min(count, -int(start))
            take = min(count - pad, n)
            if take > 0:
                out[pad:pad + take] = samples[:take]
            return out
        if start >= n:
            return out
        end = min(n, start + count)
        got = end - start
        out[:got] = samples[start:end]
        return out
    s = int(np.clip(loop[0], 0, n - 1))
    e = int(np.clip(loop[1], s + 1, n))
    length = e - s
    rel = int(start - s) % length
    done = 0
    while done < count:
        pos = s + rel
        take = min(count - done, length - rel)
        out[done:done + take] = samples[pos:pos + take]
        done += take
        rel = 0
    return out


class TimeStretcher:
    """Overlap-add stretch. Rate 1 reconstructs the input after a short fade-in.

    Rate is how fast the playhead moves through the file. Pitch stays put,
    so a faster rate raises BPM without chipmunking the vocal.
    """

    def __init__(self, win: int = 1024, hop: int = 256):
        self.win = win
        self.hop = hop
        phase = np.arange(win, dtype=np.float64)
        window = (0.5 - 0.5 * np.cos(2 * math.pi * phase / win)).astype(np.float32)
        self.window = window[:, None]
        self.cola = self._cola(window, hop)
        self.acc = np.zeros((win, 2), np.float32)
        self.pending = np.zeros((0, 2), np.float32)
        self.pos = 0.0

    @staticmethod
    def _cola(window: np.ndarray, hop: int) -> float:
        buf = np.zeros(len(window) + hop * 8, np.float32)
        for i in range(8):
            a = i * hop
            buf[a:a + len(window)] += window
        steady = buf[len(window): len(window) + hop]
        gain = float(np.mean(steady))
        return gain if gain > 1e-6 else 1.0

    def reset(self, pos: float) -> None:
        self.acc[:] = 0
        self.pending = np.zeros((0, 2), np.float32)
        self.pos = float(pos)

    def process(self, samples, n_out: int, rate: float, loop) -> np.ndarray:
        if n_out <= 0:
            return np.zeros((0, 2), np.float32)
        parts = []
        have = 0
        if len(self.pending):
            parts.append(self.pending)
            have = len(self.pending)
            self.pending = np.zeros((0, 2), np.float32)
        rate = float(rate) if rate else 1.0
        while have < n_out:
            hop_audio = self._hop(samples, rate, loop)
            parts.append(hop_audio)
            have += len(hop_audio)
        audio = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        if len(audio) > n_out:
            self.pending = np.ascontiguousarray(audio[n_out:])
            audio = audio[:n_out]
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _hop(self, samples, rate: float, loop) -> np.ndarray:
        # Nominal advance keeps the tempo. The small search keeps the grain
        # in phase with what is already overlapping, so pitch stays put.
        start = self._align(samples, loop, rate)
        grain = read_span(samples, start, self.win, loop)
        self.acc += grain * self.window
        out = (self.acc[: self.hop] / self.cola).astype(np.float32)
        self.acc[:-self.hop] = self.acc[self.hop:]
        self.acc[-self.hop:] = 0
        self.pos += self.hop * rate
        if loop is not None:
            start_loop, end_loop = float(loop[0]), float(loop[1])
            length = end_loop - start_loop
            if length > 1.0:
                self.pos = start_loop + (self.pos - start_loop) % length
        return out

    def _align(self, samples, loop, rate: float) -> int:
        natural = int(self.pos)
        if abs(rate - 1.0) < 0.003 or loop is not None:
            return natural
        overlap = self.win - self.hop
        ref = self.acc[:overlap, 0]
        if float(np.max(np.abs(ref))) < 1e-4:
            return natural
        deltas = np.arange(-64, 65, 4)
        starts = natural + deltas
        if starts[0] < 0 or starts[-1] + overlap >= len(samples):
            return natural
        index = starts[:, None] + np.arange(overlap)[None, :]
        grains = samples[index, 0]
        scores = grains @ ref
        return int(natural + deltas[int(np.argmax(scores))])


class Biquad:
    def __init__(self):
        self.b = np.array([1.0, 0.0, 0.0])
        self.a = np.array([1.0, 0.0, 0.0])
        self.zi = [np.zeros(2), np.zeros(2)]

    def reset(self) -> None:
        self.zi = [np.zeros(2), np.zeros(2)]

    def _set(self, b0, b1, b2, a0, a1, a2) -> None:
        self.b = np.array([b0 / a0, b1 / a0, b2 / a0], np.float64)
        self.a = np.array([1.0, a1 / a0, a2 / a0], np.float64)

    def _omega(self, sr: float, freq: float, q: float):
        freq = float(np.clip(freq, 20.0, sr * 0.49))
        w0 = 2 * math.pi * freq / sr
        cw = math.cos(w0)
        sw = math.sin(w0)
        alpha = sw / (2 * max(0.05, q))
        return cw, alpha

    def set_lowpass(self, sr: float, freq: float, q: float = 0.707) -> None:
        cw, alpha = self._omega(sr, freq, q)
        self._set((1 - cw) / 2, 1 - cw, (1 - cw) / 2, 1 + alpha, -2 * cw, 1 - alpha)

    def set_highpass(self, sr: float, freq: float, q: float = 0.707) -> None:
        cw, alpha = self._omega(sr, freq, q)
        self._set((1 + cw) / 2, -(1 + cw), (1 + cw) / 2, 1 + alpha, -2 * cw, 1 - alpha)

    def set_allpass(self, sr: float, freq: float, q: float = 0.7) -> None:
        cw, alpha = self._omega(sr, freq, q)
        self._set(1 - alpha, -2 * cw, 1 + alpha, 1 + alpha, -2 * cw, 1 - alpha)

    def set_lowshelf(self, sr: float, freq: float, db: float, slope: float = 1.0) -> None:
        self._set(*_shelf(sr, freq, db, slope, high=False))

    def set_highshelf(self, sr: float, freq: float, db: float, slope: float = 1.0) -> None:
        self._set(*_shelf(sr, freq, db, slope, high=True))

    def set_peak(self, sr: float, freq: float, db: float, q: float = 0.7) -> None:
        amp = 10 ** (db / 40.0)
        freq = float(np.clip(freq, 20.0, sr * 0.49))
        w0 = 2 * math.pi * freq / sr
        cw = math.cos(w0)
        alpha = math.sin(w0) / (2 * max(0.05, q))
        self._set(
            1 + alpha * amp,
            -2 * cw,
            1 - alpha * amp,
            1 + alpha / amp,
            -2 * cw,
            1 - alpha / amp,
        )

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return x
        y0, self.zi[0] = lfilter(self.b, self.a, x[:, 0], zi=self.zi[0])
        y1, self.zi[1] = lfilter(self.b, self.a, x[:, 1], zi=self.zi[1])
        return np.column_stack((y0, y1)).astype(np.float32)


def _shelf(sr: float, freq: float, db: float, slope: float, high: bool):
    amp = 10 ** (db / 40.0)
    freq = float(np.clip(freq, 20.0, sr * 0.49))
    w0 = 2 * math.pi * freq / sr
    cw = math.cos(w0)
    alpha = math.sin(w0) / 2 * math.sqrt((amp + 1 / amp) * (1 / max(0.1, slope) - 1) + 2)
    two = 2 * math.sqrt(amp) * alpha
    if high:
        b0 = amp * ((amp + 1) + (amp - 1) * cw + two)
        b1 = -2 * amp * ((amp - 1) + (amp + 1) * cw)
        b2 = amp * ((amp + 1) + (amp - 1) * cw - two)
        a0 = (amp + 1) - (amp - 1) * cw + two
        a1 = 2 * ((amp - 1) - (amp + 1) * cw)
        a2 = (amp + 1) - (amp - 1) * cw - two
    else:
        b0 = amp * ((amp + 1) - (amp - 1) * cw + two)
        b1 = 2 * amp * ((amp - 1) - (amp + 1) * cw)
        b2 = amp * ((amp + 1) - (amp - 1) * cw - two)
        a0 = (amp + 1) + (amp - 1) * cw + two
        a1 = -2 * ((amp - 1) + (amp + 1) * cw)
        a2 = (amp + 1) + (amp - 1) * cw - two
    return b0, b1, b2, a0, a1, a2


class Isolator:
    """DJ-style shelves. Center is a true bypass. The bottom of a knob is a deep cut."""

    def __init__(self, sr: int = SR):
        self.sr = sr
        self.low = Biquad()
        self.mid = Biquad()
        self.high = Biquad()
        self._db = None
        self.reset()

    def reset(self) -> None:
        self.low.reset()
        self.mid.reset()
        self.high.reset()
        self._db = None

    def _apply_db(self, low_db: float, mid_db: float, high_db: float) -> None:
        self.low.set_lowshelf(self.sr, 180.0, low_db)
        self.mid.set_peak(self.sr, 1000.0, mid_db, q=0.8)
        self.high.set_highshelf(self.sr, 3200.0, high_db)
        self._db = (low_db, mid_db, high_db)

    def process(self, x, g_low: float, g_mid: float, g_high: float):
        levels = tuple(self._to_db(gain) for gain in (g_low, g_mid, g_high))
        if max(abs(level) for level in levels) < 0.08:
            return x
        if levels != self._db:
            self._apply_db(*levels)
        y = self.low.process(x)
        y = self.mid.process(y)
        return self.high.process(y)

    @staticmethod
    def _to_db(gain: float) -> float:
        if gain <= 0.001:
            return -48.0
        return float(np.clip(20.0 * math.log10(max(gain, 1e-4)), -48.0, 6.0))


class ColorFilter:
    """Channel filter. Center bypass, left low-pass, right high-pass."""

    def __init__(self, sr: int = SR):
        self.sr = sr
        self.f = Biquad()
        self.f.set_lowpass(sr, 18000.0)

    def reset(self) -> None:
        self.f.reset()
        self.f.set_lowpass(self.sr, 18000.0)

    def process(self, x, knob: float):
        knob = float(np.clip(knob, 0.0, 1.0))
        if knob < 0.48:
            t = knob / 0.5
            freq = 220.0 * ((17000.0 / 220.0) ** t)
            self.f.set_lowpass(self.sr, freq, q=0.78)
            return self.f.process(x)
        if knob > 0.52:
            t = (knob - 0.5) / 0.5
            freq = 50.0 * ((6000.0 / 50.0) ** t)
            self.f.set_highpass(self.sr, freq, q=0.78)
            return self.f.process(x)
        self.f.set_lowpass(self.sr, 18000.0)
        self.f.process(x)
        return x


class _Comb:
    def __init__(self, length: int):
        self.buf = np.zeros((length, 2), np.float32)
        self.w = 0
        self.zi = [np.zeros(1), np.zeros(1)]

    def process(self, x, feedback: float, damp: float):
        n = len(x)
        length = len(self.buf)
        if self.w + n <= length:
            delayed = self.buf[self.w:self.w + n].copy()
        else:
            first = length - self.w
            delayed = np.concatenate((self.buf[self.w:], self.buf[: n - first]), axis=0)
        fb = delayed * feedback
        coeff = float(np.clip(damp, 0.0, 0.95))
        y0, self.zi[0] = lfilter([1.0 - coeff], [1.0, -coeff], fb[:, 0], zi=self.zi[0])
        y1, self.zi[1] = lfilter([1.0 - coeff], [1.0, -coeff], fb[:, 1], zi=self.zi[1])
        write = (x + np.column_stack((y0, y1))).astype(np.float32)
        if self.w + n <= length:
            self.buf[self.w:self.w + n] = write
        else:
            first = length - self.w
            self.buf[self.w:] = write[:first]
            self.buf[: n - first] = write[first:]
        self.w = (self.w + n) % length
        return delayed


class BeatFX:
    def __init__(self, sr: int = SR):
        self.sr = sr
        self.delay = np.zeros((sr * 2, 2), np.float32)
        self.dw = 0
        self.combs = [_Comb(n) for n in (1557, 1617, 1491, 1422)]
        self.allpass = [Biquad(), Biquad()]
        self.verb_hp = Biquad()
        self.verb_hp.set_highpass(sr, 180.0)
        self.flanger = np.zeros((sr, 2), np.float32)
        self.fw = 0
        self.phase = 0.0
        self.phasers = [Biquad() for _ in range(4)]
        self.auto = Biquad()

    def _echo(self, x, depth: float, beats: float, bpm: float):
        bpm = bpm if bpm and bpm > 40 else 120.0
        delay = int(self.sr * (60.0 / bpm) * max(0.05, beats))
        delay = int(np.clip(delay, BLOCK + 8, len(self.delay) - 1))
        n = len(x)
        length = len(self.delay)
        read_at = (self.dw - delay + np.arange(n)) % length
        delayed = self.delay[read_at]
        feedback = 0.22 + 0.5 * depth
        write = np.tanh(x + delayed * feedback).astype(np.float32)
        write_at = (self.dw + np.arange(n)) % length
        self.delay[write_at] = write
        self.dw = int((self.dw + n) % length)
        if depth < 0.01:
            self.delay *= 0.92
            return x
        return (x * (1.0 - depth) + delayed * depth).astype(np.float32)

    def _reverb(self, x, depth: float):
        if depth < 0.01:
            return x
        src = self.verb_hp.process(x)
        wet = np.zeros_like(src)
        feedback = 0.55 + 0.3 * depth
        for comb in self.combs:
            wet += comb.process(src, feedback, 0.35)
        wet *= 0.25
        freqs = (900.0, 1400.0)
        for filt, freq in zip(self.allpass, freqs):
            filt.set_allpass(self.sr, freq, q=0.65)
            wet = filt.process(wet)
        return (x * (1.0 - depth) + wet * depth * 0.9).astype(np.float32)

    def _flanger(self, x, depth: float, beats: float, bpm: float):
        if depth < 0.01:
            return x
        bpm = bpm if bpm and bpm > 40 else 120.0
        period = (60.0 / bpm) * max(beats, 0.25)
        inc = 2 * math.pi / max(1.0, period * self.sr)
        n = len(x)
        t = np.arange(n, dtype=np.float64)
        lfo = np.sin(self.phase + t * inc)
        self.phase = float((self.phase + n * inc) % (2 * math.pi))
        delay = (0.0022 * self.sr) + (0.0016 * self.sr * depth) * (0.5 + 0.5 * lfo)
        out = np.empty_like(x)
        buf = self.flanger
        length = len(buf)
        w = self.fw
        pos = 0
        while pos < n:
            m = min(32, n - pos)
            sl = slice(pos, pos + m)
            wi = w + np.arange(m)
            ri = wi - delay[sl]
            i0 = np.floor(ri).astype(np.int64) % length
            frac = (ri - np.floor(ri)).astype(np.float32)[:, None]
            i1 = (i0 + 1) % length
            delayed = buf[i0] * (1.0 - frac) + buf[i1] * frac
            mixed = x[sl] + delayed * (0.45 * depth)
            buf[wi % length] = mixed.astype(np.float32)
            wet = 0.65 * depth
            out[sl] = x[sl] * (1.0 - wet) + delayed * wet
            w = (w + m) % length
            pos += m
        self.fw = int(w)
        return out.astype(np.float32)

    def _phaser(self, x, depth: float, beats: float, bpm: float):
        if depth < 0.01:
            return x
        bpm = bpm if bpm and bpm > 40 else 120.0
        period = (60.0 / bpm) * max(beats, 0.5)
        self.phase = float((self.phase + 2 * math.pi * len(x) / (period * self.sr)) % (2 * math.pi))
        base = 280.0 + 900.0 * (0.5 + 0.5 * math.sin(self.phase))
        y = x
        for index, filt in enumerate(self.phasers):
            filt.set_allpass(self.sr, base * (1.55 ** index), q=0.85)
            y = filt.process(y)
        mix = depth * 0.65
        return (x * (1.0 - mix) + y * mix).astype(np.float32)

    def _auto_filter(self, x, depth: float, beats: float, bpm: float):
        if depth < 0.01:
            return x
        bpm = bpm if bpm and bpm > 40 else 120.0
        period = (60.0 / bpm) * max(beats, 0.5)
        self.phase = float((self.phase + 2 * math.pi * len(x) / (period * self.sr)) % (2 * math.pi))
        lfo = 0.5 + 0.5 * math.sin(self.phase)
        closed = 280.0
        opened = 280.0 * (40 ** max(0.15, depth))
        freq = closed + (opened - closed) * lfo
        self.auto.set_lowpass(self.sr, freq, q=0.9)
        wet = self.auto.process(x)
        mix = min(1.0, depth * 1.2)
        return (x * (1.0 - mix) + wet * mix).astype(np.float32)

    def process(self, x, kind: str, depth: float, beats: float, bpm: float):
        depth = float(np.clip(depth, 0.0, 1.0))
        if kind != getattr(self, "_kind", None):
            self.delay.fill(0)
            self._kind = kind
            self.phase = 0.0
        if kind == "echo":
            return self._echo(x, depth, beats, bpm)
        if kind == "reverb":
            return self._reverb(x, depth)
        if kind == "flanger":
            return self._flanger(x, depth, beats, bpm)
        if kind == "phaser":
            return self._phaser(x, depth, beats, bpm)
        if kind == "filter":
            return self._auto_filter(x, depth, beats, bpm)
        return x


def soft_limit(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 0.98:
        x = x * (0.98 / peak)
    return x
