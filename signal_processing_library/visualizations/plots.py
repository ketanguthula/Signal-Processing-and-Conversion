import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram


def plot_time_domain(signal, sampling_rate, title="Time-Domain Signal"):
    """
    Plot the signal in the time domain

    Args:
        signal (ndarray): the input signal (1D or 2D for stereo)
        sampling_rate (int): the sampling rate in Hz
        title (str): title of the plot
    """
    # Ensure the signal is mono
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)  # Convert stereo to mono

    t = np.linspace(0, len(signal) / sampling_rate, len(signal), endpoint=False)
    plt.figure(figsize=(10, 4))
    plt.plot(t, signal, color='blue')
    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


def plot_frequency_domain(freqs, fft_values, title="Frequency-Domain Signal"):
    """
    Plot the frequency spectrum of the signal

    Args:
        freqs (ndarray): array of frequencies from FFT
        fft_values (ndarray): FFT magnitudes
        title (str): title of the plot
    """
    # Plot the freq. spectrum
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_values, color='red')
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show()


def plot_spectrogram(signal, sampling_rate, title="Spectrogram"):
    """
    Plot the spectrogram of the signal

    Args:
        signal (ndarray): the input signal (1D or 2D for stereo)
        sampling_rate (int): the sampling rate in Hz
        title (str): title of the plot
    """
    # Ensure the signal is mono
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)  # Convert stereo to mono

    # Compute the spectrogram
    f, t, Sxx = spectrogram(signal, sampling_rate, nperseg=1024)

    # Plot the spectrogram
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.colorbar(label="Power (dB)")
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.show()
