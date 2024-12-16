import numpy as np
from scipy.signal import find_peaks, lfilter
from scipy.fftpack import fft
import librosa


def calculate_pitch(signal, sampling_rate):
    """
    Calculate the pitch of a signal using auto- correlation

    Args:
        signal (ndarray): the input signal
        sampling_rate (int): sampling rate of the signal

    Returns:
        float: fundamental frequency (pitch) in Hz
    """
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr) // 2:]
    peaks, _ = find_peaks(corr)
    if len(peaks) > 1:
        lag = peaks[1]
        pitch = sampling_rate / lag
        return pitch
    return 0.0


def short_term_energy(signal, frame_size, hop_size):
    """
    Calculate the short-term energy of the signal

    Args:
        signal (ndarray): The input signal
        frame_size (int): Size of each frame in samples
        hop_size (int): Step size between frames in samples

    Returns:
        ndarray: array of energy values for each frame
    """
    energy = []
    for start in range(0, len(signal) - frame_size, hop_size):
        frame = signal[start:start + frame_size]
        energy.append(np.sum(frame ** 2))
    return np.array(energy)


def compute_snr(signal, noise):
    """
    Compute the Signal-to-Noise Ratio (SNR) in dB

    Args:
        signal (ndarray): the clean signal
        noise (ndarray): the noise signal

    Returns:
        float: SNR value in dB
    """
    # Ensure the signal and noise are mono (collapse stereo to mono if needed)
    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)  # Collapse to mono
    if noise.ndim > 1:
        noise = np.mean(noise, axis=0)  # Collapse to mono

    # Compute signal and noise power
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    # Check to prevent division by zero
    if noise_power == 0:
        return np.inf  # Infinite SNR if there's no noise

    return 10 * np.log10(signal_power / noise_power)


def zero_crossing_rate(signal, frame_size, hop_size):
    """
    Calculate the zero-crossing rate of the signal

    Args:
        signal (ndarray): the input signal
        frame_size (int): size of each frame in samples
        hop_size (int): step size between frames in samples

    Returns:
        ndarray: array of ZCR values for each frame
    """
    zcr = []
    for start in range(0, len(signal) - frame_size, hop_size):
        frame = signal[start:start + frame_size]
        zcr.append(((frame[:-1] * frame[1:]) < 0).sum() / frame_size)
    return np.array(zcr)


def spectral_centroid(signal, sampling_rate):
    """
    Calculate the spectral centroid of the signal

    Args:
        signal (ndarray): the input signal
        sampling_rate (int): sampling rate of the signal

    Returns:
        float: spectral centroid in Hz
    """
    magnitude_spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal), d=1/sampling_rate)
    centroid = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)
    return centroid


def spectral_rolloff(signal, sampling_rate, rolloff_percent=0.85):
    """
    Calculate the spectral rolloff of the signal

    Args:
        signal (ndarray): the input signal
        sampling_rate (int): sampling rate of the signal
        rolloff_percent (float): percentage of spectral energy (default 85%)

    Returns:
        float: spectral rolloff frequency in Hz
    """
    magnitude_spectrum = np.abs(np.fft.rfft(signal))
    total_energy = np.sum(magnitude_spectrum)
    cumulative_energy = np.cumsum(magnitude_spectrum)
    rolloff_index = np.where(cumulative_energy >= rolloff_percent * total_energy)[0][0]
    frequencies = np.fft.rfftfreq(len(signal), d=1/sampling_rate)
    rolloff = frequencies[rolloff_index]
    return rolloff


def compute_mfcc(signal, sampling_rate, n_mfcc=13):
    """
    Compute MFCCs of the signal

    Args:
        signal (ndarray): the input signal
        sampling_rate (int): sampling rate of the signal
        n_mfcc (int): number of MFCCs to compute

    Returns:
        ndarray: Computed MFCCs.
    """
    mfccs = librosa.feature.mfcc(y=signal, sr=sampling_rate, n_mfcc=n_mfcc)
    return mfccs


def formant_frequencies(signal, sampling_rate):
    """
    Estimate formant frequencies using Linear Predictive Coding (LPC)

    Args:
        signal (ndarray): the input signal
        sampling_rate (int): sampling rate of the signal

    Returns:
        list: estimated formant frequencies in Hz
    """
    n_coeffs = 2 + sampling_rate // 1000
    a = lfilter(np.ones(n_coeffs), [1], signal)
    roots = np.roots(a)
    roots = [r for r in roots if np.imag(r) >= 0]
    angles = np.angle(roots)
    frequencies = sorted(angles * (sampling_rate / (2 * np.pi)))
    return frequencies[:4]  # Return the first 4 formants


def fft_based_downsampling(signal, original_rate, target_rate):
    """
    Downsample a signal using FFT-based low-pass filtering

    Args:
        signal (ndarray): input signal
        original_rate (int): original sampling rate of the signal
        target_rate (int): target sampling rate after downsampling

    Returns:
        ndarray: downsampled signal
    """
    fft_signal = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(len(signal), d=1 / original_rate)
    cutoff = target_rate / 2
    fft_signal[frequencies > cutoff] = 0
    filtered_signal = np.fft.irfft(fft_signal)
    downsampled_signal = filtered_signal[::original_rate // target_rate]
    return downsampled_signal


def harmonics_to_noise_ratio(signal, original_rate, target_rate=11025, frame_size=1024, hop_size=512):
    """
    Compute the Harmonics-to-Noise Ratio (HNR) of a signal using FFT-based downsampling

    Args:
        signal (ndarray): the input signal (mono)
        original_rate (int): original sampling rate of the signal
        target_rate (int): target sampling rate for downsampling
        frame_size (int): the size of each frame
        hop_size (int): the hop size between frames

    Returns:
        float: average HNR across all frames in dB
    """
    downsampled_signal = fft_based_downsampling(signal, original_rate, target_rate)
    hnr_values = []

    for frame_index in range(0, len(downsampled_signal) - frame_size, hop_size):
        frame = downsampled_signal[frame_index:frame_index + frame_size]
        autocorr = np.correlate(frame, frame, mode='full')[len(frame) - 1:]
        harmonic_energy = np.max(autocorr)
        noise_energy = np.mean(np.abs(autocorr[1:]))
        if harmonic_energy > 0 and noise_energy > 0:
            hnr = 10 * np.log10(harmonic_energy / (noise_energy + 1e-6))
        else:
            hnr = np.nan
        hnr_values.append(hnr)

    hnr_values = [hnr for hnr in hnr_values if not np.isnan(hnr)]
    return np.mean(hnr_values) if hnr_values else np.nan
