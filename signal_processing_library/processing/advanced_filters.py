from scipy.signal import butter, lfilter
import numpy as np


def highpass_filter(signal, cutoff, sampling_rate, order=5):
    """
    Apply a high-pass filter to the signal

    Args:
        signal (ndarray): the input signal
        cutoff (float): the cutoff frequency in Hz
        sampling_rate (int): the sampling rate of the signal in Hz
        order (int): the order of the filter

    Returns:
        ndarray: high-pass filtered signal
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, signal)


def bandpass_filter(signal, low_cutoff, high_cutoff, sampling_rate, order=5):
    """
    Apply a band-pass filter to the signal

    Args:
        signal (ndarray): the input signal
        low_cutoff (float): the low cutoff frequency in Hz
        high_cutoff (float): The high cutoff frequency in Hz
        sampling_rate (int): The sampling rate of the signal in Hz
        order (int): the order of the filter

    Returns:
        ndarray: band-pass filtered signal
    """
    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return lfilter(b, a, signal)


def dynamic_range_compression(signal, threshold=0.5, ratio=4):
    """
    Apply dynamic range compression to the signal

    Args:
        signal (ndarray): the input signal
        threshold (float): threshold above which compression is applied
        ratio (float): compression ratio

    Returns:
        ndarray: compressed signal
    """
    compressed_signal = np.copy(signal)
    mask = np.abs(signal) > threshold
    compressed_signal[mask] = threshold + (signal[mask] - threshold) / ratio
    return compressed_signal
