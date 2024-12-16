import pywt
import numpy as np
import matplotlib.pyplot as plt


def plot_wavelet_transform(signal, sampling_rate, wavelet='cmor1.5-1.0', title="Wavelet Transform"):
    """
    Perform and plot the Wavelet Transform of a signal

    Args:
        signal (ndarray): the signal data (1D for mono or 2D for stereo)
        sampling_rate (int): the sampling rate in Hz
        wavelet (str): the type of wavelet to use (e.g., 'cmor1.5-1.0')
        title (str): title of the plot
    """
    # Ensure the signal is mono
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)  # Convert stereo to mono by averaging channels

    # Define scales for the wavelet transform
    scales = np.arange(1, 128)  # Adjust scales based on desired frequency resolution

    # Perform Continuous Wavelet Transform
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1 / sampling_rate)

    # Plot the Wavelet Transform result
    plt.figure(figsize=(10, 6))
    plt.contourf(np.arange(len(signal)) / sampling_rate, frequencies, np.abs(coefficients), cmap='viridis')
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Magnitude")
    plt.show()
