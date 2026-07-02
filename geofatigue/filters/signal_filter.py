"""Pure-function signal filters for physiological time series."""

import numpy as np
from scipy import signal as scipy_signal


def moving_average(sig: np.ndarray, fs: float, window_sec: float) -> np.ndarray:
    """Centered moving-average (uniform boxcar), same-length output.

    Uses mode='same' convolution so all output samples are correctly normalised.
    Keep window_sec well below the signal period of interest to avoid attenuating
    the signal itself (e.g. window_sec=0.05 s for 1.2 Hz BVP at 64 Hz).
    """
    if window_sec <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec}")
    n = max(1, int(round(window_sec * fs)))
    kernel = np.ones(n) / n
    return np.convolve(sig, kernel, mode='same')


def _sos_freqresp(sos: np.ndarray, n: int) -> np.ndarray:
    """Evaluate the SOS filter's complex frequency response at the n rfft bins.

    Returns H as a complex array of length n//2 + 1, computed directly from the
    SOS coefficients at z = exp(-j*omega) for each digital frequency omega.
    This avoids sosfreqz unit-conversion pitfalls and is exact.
    """
    freqs = np.fft.rfftfreq(n)           # [0, ..., 0.5] as fraction of fs
    omega = 2.0 * np.pi * freqs          # [0, ..., π] rad/sample
    z_inv = np.exp(-1j * omega)          # z^{-1}
    z_inv2 = z_inv ** 2                  # z^{-2}

    h = np.ones(len(freqs), dtype=complex)
    for b0, b1, b2, a0, a1, a2 in sos:
        h *= (b0 + b1 * z_inv + b2 * z_inv2) / (a0 + a1 * z_inv + a2 * z_inv2)
    return h


def _fft_zerophase(sos: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Apply a zero-phase IIR filter via circular FFT convolution.

    Computes |H(f)|^2 × X(f) in the frequency domain and returns the real
    IFFT, equivalent to applying the filter forward then backward
    (as sosfiltfilt does in the time domain) but without the boundary
    transients that sosfiltfilt produces when the filter's pole magnitudes
    approach unity (narrow passband relative to fs).

    The circular convolution is exact when the signal is periodic within the
    window, which is the case for well-designed test fixtures and a good
    approximation for long physiological recordings.  For signals with sharp
    boundary discontinuities, apply a mild taper (e.g. Tukey window) first.

    This pattern is standard in physiology/neuroscience toolkits (e.g. MNE).
    """
    n = len(sig)
    h = _sos_freqresp(sos, n)
    sig_fft = np.fft.rfft(sig)
    return np.fft.irfft(sig_fft * np.abs(h) ** 2, n=n)


def bandpass(
    sig: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (FFT-based, no boundary transients)."""
    nyq = fs / 2.0
    if low_hz <= 0 or high_hz <= low_hz or high_hz >= nyq:
        raise ValueError(
            f"Invalid bandpass frequencies: low={low_hz}, high={high_hz}, nyq={nyq}"
        )
    sos = scipy_signal.butter(order, [low_hz / nyq, high_hz / nyq], btype='band', output='sos')
    return _fft_zerophase(sos, sig)


def lowpass(sig: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth lowpass filter (FFT-based, no boundary transients)."""
    nyq = fs / 2.0
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(f"cutoff_hz={cutoff_hz} out of range (0, {nyq})")
    sos = scipy_signal.butter(order, cutoff_hz / nyq, btype='low', output='sos')
    return _fft_zerophase(sos, sig)


def hampel(sig: np.ndarray, window_size: int = 5, k_sigma: float = 1.4826) -> np.ndarray:
    """Hampel identifier: replace outliers with the local median.

    When MAD is zero (constant local background) any deviation from the median
    is treated as an outlier.  This correctly handles isolated spikes in
    otherwise flat signal regions (e.g. motion artefacts in a rest period).
    """
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2
    out = sig.copy()
    for i in range(len(sig)):
        lo, hi = max(0, i - half), min(len(sig), i + half + 1)
        window = sig[lo:hi]
        med = np.median(window)
        mad = k_sigma * np.median(np.abs(window - med))
        if mad == 0:
            if sig[i] != med:
                out[i] = med
        elif abs(sig[i] - med) > 3 * mad:
            out[i] = med
    return out


def extract_rr_intervals(
    peak_timestamps_us: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RR intervals from systolic peak timestamps in microseconds.

    Note: the raw physiological loader (load_avro_physiological_data) provides
    systolic_peaks timestamps in nanoseconds — convert to microseconds before
    calling this function: `peak_timestamps_us = peak_timestamps_ns // 1000`.

    Args:
        peak_timestamps_us: 1-D int64 array of peak timestamps in microseconds.

    Returns:
        (mid_timestamps_us, rr_ms): mid-point timestamps and RR intervals in ms.
    """
    if len(peak_timestamps_us) < 2:
        raise ValueError("Need at least 2 peaks to compute RR intervals.")
    ts = np.asarray(peak_timestamps_us, dtype=np.int64)
    rr_us = np.diff(ts).astype(float)
    rr_ms = rr_us / 1_000.0
    mid_ts = (ts[:-1] + ts[1:]) // 2
    return mid_ts, rr_ms
