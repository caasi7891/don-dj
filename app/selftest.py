"""Check the sound engine, then open the window long enough to grab a picture."""
from __future__ import annotations

import os
import tempfile
import time

import numpy as np

from app.analyze import detect_bpm, detect_key
from app.dsp import SR, Isolator, TimeStretcher, eq_gain, render_resample
from app.engine import Engine
from app.synth import make_loop


def _peak_freq(audio: np.ndarray, sr: int) -> float:
    window = np.hanning(len(audio)).astype(np.float32)
    spec = np.abs(np.fft.rfft(audio * window))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    return float(freqs[int(np.argmax(spec))])


def _best_corr(reference: np.ndarray, rendered: np.ndarray) -> float:
    best = -1.0
    limit = min(len(reference), len(rendered))
    reference = reference[:limit]
    rendered = rendered[:limit]
    for lag in range(0, 2200, 20):
        if lag >= limit - 200:
            break
        left = reference[lag:]
        right = rendered[: limit - lag]
        left = left - left.mean()
        right = right - right.mean()
        denom = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denom < 1e-9:
            continue
        best = max(best, float(np.dot(left, right) / denom))
    return best


def test_bpm() -> None:
    np.random.seed(2)
    audio = make_loop(120.0, 110.0, (220.0, 261.63, 329.63), False)
    bpm, offset = detect_bpm(audio, SR)
    if abs(bpm - 120.0) > 1.5:
        raise AssertionError(f"BPM {bpm:.2f}, offset {offset:.4f}")
    beat = 60.0 / bpm
    wrapped = min(offset % beat, beat - (offset % beat))
    if wrapped > 0.04:
        raise AssertionError(f"beat offset {offset:.4f} is far from the downbeat")


def test_key() -> None:
    sr = SR
    seconds = 3
    t = np.arange(int(sr * seconds)) / sr
    # A minor chord: A C E
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 261.63, 329.63))
    stereo = np.column_stack((chord, chord)).astype(np.float32) * 0.2
    name, camelot, score = detect_key(stereo, sr)
    if camelot != "8A" and name not in ("Am", "C"):
        raise AssertionError(f"key {name} {camelot} score {score:.2f}")


def test_stretch_identity_and_pitch() -> None:
    seconds = 2.0
    t = np.arange(int(SR * seconds)) / SR
    sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    stereo = np.column_stack((sine, sine))
    stretcher = TimeStretcher()
    stretcher.reset(0.0)
    chunks = [stretcher.process(stereo, 256, 1.0, None) for _ in range(int(SR / 256))]
    rendered = np.concatenate(chunks, axis=0)[:, 0]
    score = _best_corr(sine[1024:], rendered[1024:])
    if score < 0.98:
        raise AssertionError(f"stretch identity correlation {score:.3f}")
    stretcher.reset(0.0)
    sped = np.concatenate(
        [stretcher.process(stereo, 256, 1.25, None) for _ in range(int(SR / 256))],
        axis=0,
    )[:, 0]
    heard = _peak_freq(sped[2048:2048 + SR], SR)
    if abs(heard - 440.0) > 8.0:
        raise AssertionError(f"keylock pitch moved to {heard:.1f} Hz")
    plain, _pos = render_resample(stereo, 0.0, SR, 1.25, None)
    plain_freq = _peak_freq(plain[:, 0], SR)
    if abs(plain_freq - 550.0) > 8.0:
        raise AssertionError(f"resample pitch {plain_freq:.1f} Hz, expected about 550")


def test_eq_unity_and_kill() -> None:
    eq = Isolator()
    n = 2048
    t = np.arange(n) / SR
    bass = np.sin(2 * np.pi * 70.0 * t).astype(np.float32)
    stereo = np.column_stack((bass, bass))
    dry = eq.process(stereo, 1, 1, 1)
    if not np.allclose(dry, stereo, atol=1e-5):
        raise AssertionError("EQ at center changed the sound")
    killed = eq.process(stereo, eq_gain(0.0), 1, 1)
    dry_power = float(np.mean(stereo ** 2))
    killed_power = float(np.mean(killed ** 2))
    if killed_power > dry_power * 0.15:
        raise AssertionError(f"low kill left too much energy {killed_power:.4f} vs {dry_power:.4f}")


def test_mix_and_sync() -> None:
    engine = Engine()
    engine.load_practice()
    for _ in range(6):
        engine.render_block()
    left = engine.decks[0]
    right = engine.decks[1]
    if not left.loaded or not right.loaded or not left.playing:
        raise AssertionError("practice tracks did not arm")
    if not left.loop_active or not right.loop_active:
        raise AssertionError("practice loops were not armed")
    audio = np.concatenate([engine.render_block() for _ in range(180)], axis=0)
    peak = float(np.max(np.abs(audio)))
    if peak < 0.05 or not np.isfinite(audio).all():
        raise AssertionError(f"mix peak {peak}")
    right.commands.put(("play",))
    right.commands.put(("sync", True))
    for _ in range(30):
        engine.render_block()
    heard_a = left.bpm * left.rate
    heard_b = right.bpm * right.rate
    if abs(heard_a - heard_b) > 0.8:
        raise AssertionError(f"sync left {heard_a:.2f} right {heard_b:.2f}")
    if right.pos < 0 or right.pos >= len(right.samples):
        raise AssertionError("playhead left the track")


def test_window() -> str:
    from PySide6.QtWidgets import QApplication

    from app.window import MainWindow

    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    window._practice()
    loaded = False
    level = 0.0
    for _ in range(80):
        app.processEvents()
        time.sleep(0.05)
        deck = window.engine.decks[0]
        level = max(level, deck.level, window.engine.master_level)
        if deck.loaded and deck.playing and level > 0.02:
            loaded = True
            break
    app.processEvents()
    path = os.path.join(tempfile.gettempdir(), "dondj_preview.png")
    window.grab().save(path)
    title_a = window.deck_a.title_label.text()
    title_b = window.deck_b.title_label.text()
    tip = window.tip.text()
    window.close()
    app.processEvents()
    if not loaded:
        raise AssertionError(f"window did not play practice audio, level {level:.4f}")
    if "Warm" not in title_a and "연습" not in title_a:
        raise AssertionError(f"deck A title unexpected: {title_a}")
    if not title_b or title_b == "곡 없음":
        raise AssertionError(f"deck B title unexpected: {title_b}")
    if not tip:
        raise AssertionError("tip bar is empty")
    return path


def main() -> None:
    tests = [test_bpm, test_key, test_stretch_identity_and_pitch, test_eq_unity_and_kill, test_mix_and_sync]
    for test in tests:
        test()
        print("ok", test.__name__)
    path = test_window()
    print("ok window", path)


if __name__ == "__main__":
    main()
