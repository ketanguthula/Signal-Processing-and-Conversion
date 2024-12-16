import numpy as np
from scipy.signal import butter, lfilter

def normalize_signal(signal):
    """
    Normalize the signal amplitude to the range [-1, 1].

    Args:
        signal (ndarray): The input signal.

    Returns:
        ndarray: Normalized signal.
    """
    return signal / np.max(np.abs(signal))

def lowpass_filter(signal, cutoff, sampling_rate, order=5):
    """
    Apply a low-pass filter to the signal.

    Args:
        signal (ndarray): The input signal.
        cutoff (float): The cutoff frequency in Hz.
        sampling_rate (int): The sampling rate of the signal in Hz.
        order (int): The order of the filter.

    Returns:
        ndarray: Filtered signal.
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)
