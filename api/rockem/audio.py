"""Audio at the edge: everything before the model's encoder, so the node sees only finished
windows (docs/queue/094; tcdent 2026-10-23: "offload any preprocessing to the api").

The pipeline is the reference's (mistral-common and transformers' Voxtral processor): decode
to f32, average the channels, resample to the model's rate with soxr at HQ, pad with zeros to
a whole number of windows, take Whisper's log-mel spectrogram over the whole clip, then cut it
into windows. The spectrogram is normalized by the clip's maximum, so a window depends on the
whole clip and is never computed alone. The mel filterbank is the bundle's own file."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np


class AudioError(ValueError):
    """The bytes are not audio this edge can read."""


@dataclass(frozen=True)
class Spectrogram:
    """A model's spectrogram, from `[model.audio]` and its filterbank."""

    sample_rate: int
    hop: int
    fft_window: int
    bins: int
    frames: int  # a window's frames
    filters: np.ndarray  # [bins][fft_window // 2 + 1] f32

    @property
    def window_samples(self) -> int:
        return self.frames * self.hop

    @classmethod
    def of(cls, audio: dict, filters: bytes) -> Spectrogram:
        bins, frames = audio["window"]
        n_freq = audio["fft_window"] // 2 + 1
        f = np.frombuffer(filters, dtype="<f4")
        if f.size != bins * n_freq:
            raise AudioError(f"mel filters: {f.size} values, the model's are {bins} x {n_freq}")
        return cls(audio["sample_rate"], audio["hop"], audio["fft_window"], bins, frames, f.reshape(bins, n_freq))

    def signal(self, data: bytes) -> np.ndarray:
        """Mono f32 at the model's rate, from any file soundfile reads (WAV, FLAC, OGG, MP3…)."""
        import soundfile
        import soxr

        try:
            with soundfile.SoundFile(io.BytesIO(data)) as f:
                x, rate = f.read(dtype="float32"), f.samplerate
        except (soundfile.LibsndfileError, RuntimeError, TypeError) as e:
            raise AudioError(f"not audio: {e}") from e
        if x.ndim == 2:
            x = x.mean(axis=1)
        if x.size == 0:
            raise AudioError("no samples")
        if rate != self.sample_rate:
            x = soxr.resample(x, rate, self.sample_rate, quality="HQ")
        return np.ascontiguousarray(x, dtype=np.float32)

    def log_mel(self, x: np.ndarray) -> np.ndarray:
        """Whisper's log-mel over the clip padded to whole windows: [bins][frames · windows]."""
        n = -(-x.size // self.window_samples) * self.window_samples
        x = np.pad(x.astype(np.float64), (0, n - x.size))
        half = self.fft_window // 2
        x = np.pad(x, (half, half), mode="reflect")  # centered frames
        count = 1 + (x.size - self.fft_window) // self.hop
        idx = np.arange(self.fft_window)[None, :] + self.hop * np.arange(count)[:, None]
        window = np.hanning(self.fft_window + 1)[:-1]  # periodic Hann
        power = np.abs(np.fft.rfft(x[idx] * window, axis=1)) ** 2  # [frames][freq]
        mel = self.filters.astype(np.float64) @ power[:-1].T  # the last frame dropped, as Whisper does
        log = np.log10(np.maximum(mel, 1e-10))
        log = np.maximum(log, log.max() - 8.0)
        return ((log + 4.0) / 4.0).astype(np.float32)

    def windows(self, data: bytes) -> list[bytes]:
        """The clip as the node takes it: one f32 little-endian [bins][frames] block a window."""
        mel = self.log_mel(self.signal(data))
        return [np.ascontiguousarray(mel[:, i : i + self.frames]).astype("<f4").tobytes() for i in range(0, mel.shape[1], self.frames)]
