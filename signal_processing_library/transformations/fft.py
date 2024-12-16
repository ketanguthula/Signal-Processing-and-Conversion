import numpy as np
from scipy.signal import find_peaks


def compute_fft(signal, sampling_rate):
    """
    Compute the Fast Fourier Transform (FFT) of the signal

    Args:
        signal (ndarray): the input signal. Can be mono (1D) or stereo (2D)
        sampling_rate (int): the sampling rate of the signal in Hz

    Returns:
        tuple: (frequencies, FFT magnitudes)
    """
    # Ensure the signal is mono and process the first channel if stereo
    if signal.ndim == 2:
        signal = signal[:, 0]  # Use the first channel for stereo signals

    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1 / sampling_rate)  # Frequency bins
    fft_values = np.abs(np.fft.rfft(signal))  # Magnitude of FFT
    return freqs, fft_values


def detect_dominant_frequencies(freqs, fft_values, height=0.01):
    """
    Detect dominant frequencies from the FFT result

    Args:
        freqs (ndarray): array of frequencies from FFT
        fft_values (ndarray): FFT magnitudes
        height (float): minimum height of peaks to detect

    Returns:
        ndarray: array of dominant frequencies
    """
    peaks, _ = find_peaks(fft_values, height=height)
    return freqs[peaks]
