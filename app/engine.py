"""Two decks, a mixer, and the sound-output thread."""
from __future__ import annotations

import queue
import threading
import time
import traceback
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from app.dsp import (
    BLOCK,
    SR,
    BeatFX,
    ColorFilter,
    Isolator,
    TimeStretcher,
    cross_gains,
    eq_gain,
    fader_gain,
    render_resample,
    render_scratch,
    soft_limit,
    trim_gain,
)
from app.synth import make_practice_tracks

TARGET_FILL = BLOCK * 5
RING_FRAMES = BLOCK * 48


class RingBuffer:
    def __init__(self, frames: int):
        self.buf = np.zeros((frames, 2), np.float32)
        self.n = frames
        self.r = 0
        self.w = 0
        self.count = 0
        self.lock = threading.Lock()

    def fill_count(self) -> int:
        with self.lock:
            return self.count

    def space(self) -> int:
        with self.lock:
            return self.n - self.count

    def write(self, data: np.ndarray) -> int:
        n = len(data)
        with self.lock:
            n = min(n, self.n - self.count)
            if n <= 0:
                return 0
            first = min(n, self.n - self.w)
            self.buf[self.w:self.w + first] = data[:first]
            second = n - first
            if second:
                self.buf[:second] = data[first:first + second]
            self.w = (self.w + n) % self.n
            self.count += n
            return n

    def read_into(self, out: np.ndarray) -> int:
        n = len(out)
        with self.lock:
            take = min(n, self.count)
            if take > 0:
                first = min(take, self.n - self.r)
                out[:first] = self.buf[self.r:self.r + first]
                second = take - first
                if second:
                    out[first:take] = self.buf[:second]
                self.r = (self.r + take) % self.n
                self.count -= take
        if take < n:
            out[take:] = 0
        return take


class Deck:
    def __init__(self, index: int):
        self.index = index
        self.commands = queue.SimpleQueue()
        self.samples = None
        self.path = ""
        self.title = ""
        self.artist = ""
        self.bpm = 0.0
        self.beat_offset = 0.0
        self.musical_key = ""
        self.camelot = ""
        self.overview = None
        self.detail = None
        self.detail_hz = 40.0
        self.hot_cues = [None] * 8
        self.cue = 0.0
        self.pos = 0.0
        self.rate = 1.0
        self.bend = 0.0
        self.bend_hold = False
        self.playing = False
        self.cue_preview = False
        self.keylock = True
        self.vinyl = True
        self.quantize = True
        self.slip = False
        self.sync_on = False
        self.sync_force_align = False
        self.sync_cooldown = 0
        self.loop_active = False
        self.loop_start = 0.0
        self.loop_end = 0.0
        self.scratching = False
        self.scratch_target = None
        self.slip_action = False
        self.shadow = 0.0
        self.fade = 0
        self.stretcher = TimeStretcher()
        self.stretch_live = False
        self.eq = Isolator()
        self.color = ColorFilter()
        self.eq_low = 0.5
        self.eq_mid = 0.5
        self.eq_high = 0.5
        self.filter_knob = 0.5
        self.trim = 0.5
        self.fader = 0.82
        self.pfl = False
        self.level = 0.0
        self.loaded = False
        self.error = ""
        self.cues_dirty = False
        self.grid_dirty = False
        self.duration = 0.0
        self.revision = 0
        self.tempo_range = 8.0


class Engine:
    def __init__(self):
        self.decks = [Deck(0), Deck(1)]
        self.master_index = 0
        self.crossfader = 0.5
        self.cross_curve = 0.25
        self.cue_mix = 0.0
        self.master = 0.75
        self.fx = BeatFX()
        self.fx_kind = "echo"
        self.fx_depth = 0.0
        self.fx_beats = 1.0
        self.ring = RingBuffer(RING_FRAMES)
        self.running = False
        self.thread = None
        self.stream = None
        self.device_error = ""
        self.device_index = None
        self.underruns = 0
        self.render_error = ""
        self.master_level = 0.0
        self.recording = False
        self.record_chunks = []
        self.record_frames = 0
        self.record_lock = threading.Lock()
        self.record_done = ""
        self.record_note = ""
        self._clocked = False

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="don-dj-audio", daemon=True)
        self.thread.start()
        for _ in range(80):
            if self.ring.fill_count() >= BLOCK * 2:
                break
            time.sleep(0.01)
        self.open_device(None)

    def stop(self) -> None:
        self.running = False
        if self.recording:
            self.stop_recording()
        stream = self.stream
        self.stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

    def list_devices(self) -> list:
        try:
            import sounddevice as sd
        except Exception as exc:
            self.device_error = "소리 장치를 찾지 못했습니다. " + str(exc)
            return []
        found = []
        try:
            devices = sd.query_devices()
        except Exception as exc:
            self.device_error = "소리 장치 목록을 읽지 못했습니다. " + str(exc)
            return []
        for index, dev in enumerate(devices):
            if dev.get("max_output_channels", 0) >= 2:
                found.append((index, dev.get("name") or f"장치 {index}"))
        return found

    def open_device(self, index) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            self.device_error = "sounddevice를 불러오지 못했습니다. " + str(exc)
            self._clocked = True
            return
        stream = self.stream
        self.stream = None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        try:
            self.stream = sd.OutputStream(
                device=index,
                samplerate=SR,
                channels=2,
                dtype="float32",
                blocksize=BLOCK,
                latency="low",
                callback=self._callback,
            )
            self.stream.start()
            self.device_index = index
            self.device_error = ""
            self._clocked = False
        except Exception as exc:
            self.stream = None
            self.device_error = (
                "오디오 장치를 열지 못했습니다. 위의 장치 목록에서 다른 출력을 골라 보세요. "
                + str(exc)
            )
            self._clocked = True

    def _callback(self, outdata, frames, _time, status) -> None:
        if status:
            self.underruns += 1
        got = self.ring.read_into(outdata)
        if got < frames:
            self.underruns += 1

    def _loop(self) -> None:
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, 1)
        except Exception:
            pass
        next_tick = time.perf_counter()
        while self.running:
            try:
                if self._clocked:
                    self.ring.write(self.render_block())
                    next_tick += BLOCK / SR
                    delay = next_tick - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_tick = time.perf_counter()
                    continue
                if self.ring.fill_count() < TARGET_FILL and self.ring.space() >= BLOCK:
                    self.ring.write(self.render_block())
                else:
                    time.sleep(0.0015)
            except Exception:
                self.render_error = traceback.format_exc()
                time.sleep(0.05)

    def render_block(self) -> np.ndarray:
        for deck in self.decks:
            self._drain(deck)
        self._apply_sync_all()
        pres = []
        posts = []
        for deck in self.decks:
            dry = self._render_deck_audio(deck, BLOCK)
            dry = self._apply_fade(deck, dry)
            eqed = deck.eq.process(
                dry,
                eq_gain(deck.eq_low),
                eq_gain(deck.eq_mid),
                eq_gain(deck.eq_high),
            )
            colored = deck.color.process(eqed, deck.filter_knob)
            pre = colored * trim_gain(deck.trim)
            deck.level = float(np.max(np.abs(pre))) if pre.size else 0.0
            post = pre * fader_gain(deck.fader)
            pres.append(pre)
            posts.append(post)
            self._update_slip(deck, BLOCK)
        gain_a, gain_b = cross_gains(self.crossfader, self.cross_curve)
        master = posts[0] * gain_a + posts[1] * gain_b
        master = self.fx.process(master, self.fx_kind, self.fx_depth, self.fx_beats, self._fx_bpm())
        cue = np.zeros_like(master)
        armed = 0
        for deck, pre in zip(self.decks, pres):
            if deck.pfl:
                cue += pre
                armed += 1
        if armed:
            mix = float(np.clip(self.cue_mix, 0.0, 1.0))
            out = master * (1.0 - mix) + cue * mix
        else:
            out = master
        out = out * (float(np.clip(self.master, 0.0, 1.0)) ** 1.35)
        out = soft_limit(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0))
        self.master_level = float(np.max(np.abs(out))) if out.size else 0.0
        self._maybe_record(out)
        return np.ascontiguousarray(out, dtype=np.float32)

    def _drain(self, deck: Deck) -> None:
        while True:
            try:
                cmd = deck.commands.get_nowait()
            except queue.Empty:
                return
            self._apply(deck, cmd)

    def _apply(self, deck: Deck, cmd) -> None:
        name = cmd[0]
        if name == "load":
            self._load(deck, cmd[1])
        elif name == "toggle":
            self._toggle(deck)
        elif name == "play":
            if deck.loaded:
                deck.cue_preview = False
                deck.playing = True
        elif name == "pause":
            deck.playing = False
            deck.cue_preview = False
        elif name == "cue_down":
            self._cue_down(deck)
        elif name == "cue_up":
            self._cue_up(deck)
        elif name == "seek":
            if deck.loaded:
                self._seek(deck, float(cmd[1]) * SR)
        elif name == "rate":
            deck.rate = float(np.clip(cmd[1], 0.5, 1.5))
            deck.sync_on = False
        elif name == "sync":
            deck.sync_on = bool(cmd[1])
            deck.sync_force_align = bool(cmd[1])
        elif name == "keylock":
            deck.keylock = bool(cmd[1])
            deck.stretch_live = False
        elif name == "vinyl":
            deck.vinyl = bool(cmd[1])
        elif name == "quantize":
            deck.quantize = bool(cmd[1])
        elif name == "slip":
            deck.slip = bool(cmd[1])
        elif name == "hot":
            self._hot(deck, int(cmd[1]), bool(cmd[2]))
        elif name == "loop":
            self._loop_beats(deck, float(cmd[1]))
        elif name == "loop_in":
            self._loop_in(deck)
        elif name == "loop_out":
            self._loop_out(deck)
        elif name == "loop_exit":
            deck.loop_active = False
        elif name == "reloop":
            self._reloop(deck)
        elif name == "jump":
            self._jump(deck, float(cmd[1]))
        elif name == "bpm_scale":
            self._scale_bpm(deck, float(cmd[1]))
        elif name == "grid_nudge":
            self._nudge_grid(deck, float(cmd[1]))
        elif name == "range":
            deck.tempo_range = float(cmd[1])

    def _load(self, deck: Deck, track: dict) -> None:
        samples = np.ascontiguousarray(track["samples"], dtype=np.float32)
        if samples.ndim == 1:
            samples = np.column_stack((samples, samples))
        deck.samples = samples
        deck.path = track.get("path") or ""
        deck.title = track.get("title") or "제목 없음"
        deck.artist = track.get("artist") or ""
        deck.bpm = float(track.get("bpm") or 0)
        deck.beat_offset = float(track.get("beat_offset") or 0)
        deck.musical_key = track.get("musical_key") or ""
        deck.camelot = track.get("camelot") or ""
        deck.overview = track.get("overview")
        deck.detail = track.get("detail")
        deck.detail_hz = float(track.get("detail_hz") or 40)
        hot = list(track.get("hot_cues") or [None] * 8)
        while len(hot) < 8:
            hot.append(None)
        deck.hot_cues = hot[:8]
        deck.cue = float(track.get("cue") or 0)
        deck.duration = len(samples) / SR
        deck.loaded = True
        deck.error = ""
        deck.playing = False
        deck.cue_preview = False
        deck.sync_on = False
        deck.loop_active = False
        deck.rate = 1.0
        deck.bend = 0.0
        deck.scratching = False
        deck.eq.reset()
        deck.color.reset()
        deck.revision += 1
        self._seek(deck, deck.cue * SR)
        if track.get("auto_loop"):
            deck.loop_start = 0.0
            deck.loop_end = float(len(samples))
            deck.loop_active = True
        if track.get("auto_play"):
            deck.playing = True

    def _toggle(self, deck: Deck) -> None:
        if not deck.loaded:
            return
        if deck.cue_preview:
            deck.cue_preview = False
            deck.playing = True
            return
        deck.playing = not deck.playing
        if deck.playing:
            deck.fade = max(deck.fade, 32)

    def _cue_down(self, deck: Deck) -> None:
        if not deck.loaded:
            return
        if deck.playing and not deck.cue_preview:
            deck.playing = False
            deck.cue_preview = False
            self._seek(deck, deck.cue * SR)
            return
        pos = self._quantize(deck, deck.pos)
        deck.cue = max(0.0, pos / SR)
        deck.cues_dirty = True
        self._seek(deck, pos)
        deck.cue_preview = True
        deck.playing = True

    def _cue_up(self, deck: Deck) -> None:
        if deck.cue_preview:
            deck.cue_preview = False
            deck.playing = False
            self._seek(deck, deck.cue * SR)

    def _hot(self, deck: Deck, index: int, delete: bool) -> None:
        if not deck.loaded or not 0 <= index < 8:
            return
        if delete:
            deck.hot_cues[index] = None
            deck.cues_dirty = True
            return
        if deck.hot_cues[index] is None:
            pos = self._quantize(deck, deck.pos)
            deck.hot_cues[index] = max(0.0, pos / SR)
            deck.cues_dirty = True
        else:
            self._seek(deck, float(deck.hot_cues[index]) * SR)

    def _loop_beats(self, deck: Deck, beats: float) -> None:
        if not deck.loaded:
            return
        if deck.bpm <= 1:
            deck.error = "BPM을 알아야 박자 루프를 걸 수 있습니다."
            return
        beat = SR * 60.0 / deck.bpm
        if deck.loop_active:
            current = (deck.loop_end - deck.loop_start) / beat
            if abs(current - beats) < 0.25:
                deck.loop_active = False
                return
        start = self._quantize(deck, deck.pos)
        end = start + beats * beat
        end = min(end, float(len(deck.samples) - 2))
        if end <= start + 32:
            return
        deck.loop_start = start
        deck.loop_end = end
        deck.loop_active = True
        deck.error = ""

    def _loop_in(self, deck: Deck) -> None:
        if not deck.loaded:
            return
        deck.loop_start = self._quantize(deck, deck.pos)
        deck.loop_active = False

    def _loop_out(self, deck: Deck) -> None:
        if not deck.loaded:
            return
        end = self._quantize(deck, deck.pos)
        if end <= deck.loop_start + 32:
            end = min(len(deck.samples) - 2, deck.pos)
        if end <= deck.loop_start + 32:
            return
        deck.loop_end = float(end)
        deck.loop_active = True

    def _reloop(self, deck: Deck) -> None:
        if not deck.loaded or deck.loop_end <= deck.loop_start + 32:
            return
        deck.loop_active = True
        self._seek(deck, deck.loop_start)

    def _jump(self, deck: Deck, beats: float) -> None:
        if not deck.loaded:
            return
        if deck.bpm <= 1:
            self._seek(deck, deck.pos + beats * 0.5 * SR)
            return
        dest = deck.pos + beats * (SR * 60.0 / deck.bpm)
        if deck.quantize:
            dest = self._quantize(deck, dest)
        self._seek(deck, dest)

    def _scale_bpm(self, deck: Deck, factor: float) -> None:
        if deck.bpm <= 1:
            return
        deck.bpm = float(np.clip(deck.bpm * factor, 60.0, 200.0))
        deck.grid_dirty = True

    def _nudge_grid(self, deck: Deck, seconds: float) -> None:
        deck.beat_offset = float(deck.beat_offset + seconds)
        if deck.bpm > 1:
            beat = 60.0 / deck.bpm
            deck.beat_offset = deck.beat_offset % beat
        deck.grid_dirty = True

    def _quantize(self, deck: Deck, pos: float) -> float:
        if not deck.quantize or deck.bpm <= 1 or deck.samples is None:
            return float(pos)
        beat = SR * 60.0 / deck.bpm
        offset = deck.beat_offset * SR
        step = round((pos - offset) / beat)
        quant = offset + step * beat
        return float(np.clip(quant, 0, len(deck.samples) - 2))

    def _seek(self, deck: Deck, pos: float) -> None:
        if deck.samples is None or len(deck.samples) < 2:
            deck.pos = 0.0
            return
        pos = float(np.clip(pos, 0, len(deck.samples) - 2))
        deck.pos = pos
        deck.stretcher.reset(pos)
        deck.stretch_live = False
        deck.fade = 96

    def _apply_sync_all(self) -> None:
        master = self.decks[self.master_index if self.master_index in (0, 1) else 0]
        for deck in self.decks:
            if deck is master or not deck.sync_on:
                continue
            self._sync_one(deck, master)

    def _phase(self, deck: Deck):
        if deck.bpm <= 1:
            return None
        beat = SR * 60.0 / deck.bpm
        return ((deck.pos - deck.beat_offset * SR) / beat) % 1.0

    def _sync_one(self, deck: Deck, master: Deck) -> None:
        if not deck.loaded or not master.loaded or deck.bpm <= 1 or master.bpm <= 1:
            return
        if deck.scratching:
            return
        target = (master.bpm * max(0.25, master.rate)) / deck.bpm
        target = float(np.clip(target, 0.5, 1.5))
        master_phase = self._phase(master)
        slave_phase = self._phase(deck)
        if master_phase is None or slave_phase is None:
            deck.rate = target
            return
        err = (master_phase - slave_phase + 0.5) % 1.0 - 0.5
        if deck.sync_cooldown > 0:
            deck.sync_cooldown -= 1
        need_jump = deck.sync_force_align or (abs(err) > 0.35 and deck.sync_cooldown <= 0)
        if need_jump:
            beat = SR * 60.0 / deck.bpm
            self._seek(deck, deck.pos + err * beat)
            deck.rate = target
            deck.sync_force_align = False
            deck.sync_cooldown = int(SR / BLOCK)
            return
        deck.rate = float(np.clip(target * (1.0 + np.clip(err, -0.2, 0.2) * 0.12), 0.5, 1.5))

    def _loop_pair(self, deck: Deck):
        if deck.loop_active and deck.loop_end > deck.loop_start + 16:
            return (deck.loop_start, deck.loop_end)
        return None

    def _render_deck_audio(self, deck: Deck, n: int) -> np.ndarray:
        if not deck.bend_hold:
            deck.bend *= 0.9
            if abs(deck.bend) < 0.0008:
                deck.bend = 0.0
        if deck.samples is None or not deck.loaded:
            deck.level *= 0.6
            return np.zeros((n, 2), np.float32)
        if deck.scratching and deck.scratch_target is not None:
            audio, deck.pos = render_scratch(deck.samples, deck.pos, float(deck.scratch_target), n)
            deck.stretch_live = False
            return audio
        if not deck.playing:
            return np.zeros((n, 2), np.float32)
        rate = float(np.clip(deck.rate + deck.bend, 0.25, 2.0))
        loop = self._loop_pair(deck)
        if loop is None and deck.pos >= len(deck.samples) - 4:
            deck.playing = False
            deck.cue_preview = False
            return np.zeros((n, 2), np.float32)
        use_stretch = deck.keylock and abs(rate - 1.0) >= 0.004
        if use_stretch:
            if not deck.stretch_live:
                deck.stretcher.reset(deck.pos)
                deck.stretch_live = True
            audio = deck.stretcher.process(deck.samples, n, rate, loop)
            deck.pos = float(deck.stretcher.pos)
        else:
            deck.stretch_live = False
            if loop is None:
                remain = len(deck.samples) - 2 - deck.pos
                frames = int(remain / rate) if rate > 0 else n
                if frames <= 0:
                    deck.playing = False
                    return np.zeros((n, 2), np.float32)
                if frames < n:
                    audio = np.zeros((n, 2), np.float32)
                    piece, deck.pos = render_resample(deck.samples, deck.pos, frames, rate, None)
                    audio[:frames] = piece
                    deck.playing = False
                    deck.cue_preview = False
                    return audio
            audio, deck.pos = render_resample(deck.samples, deck.pos, n, rate, loop)
        return audio

    def _apply_fade(self, deck: Deck, audio: np.ndarray) -> np.ndarray:
        if deck.fade > 0 and len(audio):
            amount = min(len(audio), int(deck.fade))
            ramp = np.linspace(0.0, 1.0, amount, dtype=np.float32)[:, None]
            audio = audio.copy()
            audio[:amount] *= ramp
            deck.fade -= amount
        return audio

    def _update_slip(self, deck: Deck, n: int) -> None:
        in_action = bool(deck.scratching or deck.loop_active)
        if deck.slip and deck.playing:
            if in_action:
                if not deck.slip_action:
                    deck.shadow = deck.pos
                deck.shadow += n * max(0.25, deck.rate)
                if deck.samples is not None and deck.shadow >= len(deck.samples) - 2:
                    deck.shadow = float(len(deck.samples) - 2)
            elif deck.slip_action:
                self._seek(deck, deck.shadow)
        else:
            deck.shadow = deck.pos
        deck.slip_action = bool(deck.slip and deck.playing and in_action)

    def _fx_bpm(self) -> float:
        master = self.decks[self.master_index if self.master_index in (0, 1) else 0]
        if master.bpm > 1:
            return master.bpm * max(0.25, master.rate)
        other = self.decks[1 - master.index]
        if other.bpm > 1:
            return other.bpm * max(0.25, other.rate)
        return 120.0

    def _maybe_record(self, audio: np.ndarray) -> None:
        if not self.recording:
            return
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        with self.record_lock:
            if not self.recording:
                return
            self.record_chunks.append(pcm)
            self.record_frames += len(pcm)
            if self.record_frames >= SR * 60 * 20:
                self.recording = False
                self.record_note = "20분이 되어 녹음을 마쳤습니다."

    def start_recording(self) -> None:
        with self.record_lock:
            self.record_chunks = []
            self.record_frames = 0
            self.record_done = ""
            self.record_note = ""
            self.recording = True

    def stop_recording(self) -> str:
        with self.record_lock:
            self.recording = False
            chunks = self.record_chunks
            self.record_chunks = []
        if not chunks:
            self.record_done = ""
            return ""
        audio = np.concatenate(chunks, axis=0)
        folder = Path.home() / "Documents" / "don-dj"
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            folder = Path.home() / "don-dj"
            folder.mkdir(parents=True, exist_ok=True)
        path = folder / datetime.now().strftime("mix-%Y%m%d-%H%M%S.wav")
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(SR)
            handle.writeframes(audio.tobytes())
        self.record_done = str(path)
        return str(path)

    def load_practice(self) -> None:
        left, right = make_practice_tracks()
        self.decks[0].commands.put(("load", left))
        self.decks[1].commands.put(("load", right))
        self.decks[0].commands.put(("play",))
        self.master_index = 0
        self.decks[1].sync_on = False

    def enqueue_load(self, index: int, track: dict) -> None:
        self.decks[index].error = ""
        self.decks[index].commands.put(("load", track))
