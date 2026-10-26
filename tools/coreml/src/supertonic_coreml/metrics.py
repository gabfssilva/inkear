import numpy as np


def log_spectral_distance(actual: np.ndarray, expected: np.ndarray, floor_db: float = 80.0) -> float:
    """Mean log-spectral distance in dB, ignoring bins `floor_db` below the loudest one.

    Phase-insensitive, so two renditions that sound alike score close to zero even when
    their waveforms drift apart.
    """

    def spectrum(wav: np.ndarray) -> np.ndarray:
        frames = np.lib.stride_tricks.sliding_window_view(wav, 2048)[::512] * np.hanning(2048)
        return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=-1)) + 1e-9)

    length = min(len(actual), len(expected))
    a, e = spectrum(actual[:length]), spectrum(expected[:length])
    floor = e.max() - floor_db
    return float(np.sqrt(((np.maximum(a, floor) - np.maximum(e, floor)) ** 2).mean(axis=1)).mean())
