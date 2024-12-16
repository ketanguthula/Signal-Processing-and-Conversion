import time
from input.file_loader import load_audio, get_signal_metadata
from processing.preprocess import normalize_signal, lowpass_filter
from processing.advanced_filters import dynamic_range_compression
from transformations.fft import compute_fft
from visualizations.plots import plot_time_domain, plot_frequency_domain, plot_spectrogram
from analysis.feature_extraction import (
    short_term_energy,
    zero_crossing_rate,
    spectral_centroid,
    spectral_rolloff,
    compute_mfcc,
    harmonics_to_noise_ratio,
)


def main():
    try:
        file_path = input("Enter the path of the file to analyze: ")

        # Load the signal
        sampling_rate, signal = load_audio(file_path)

        # Print metadata
        metadata = get_signal_metadata(sampling_rate, signal)
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        normalized_signal = normalize_signal(signal)
        filtered_signal = lowpass_filter(normalized_signal, cutoff=1000, sampling_rate=sampling_rate)

        # Compute and display Signal-to-Noise Ratio
        noise = 0.01 * np.random.randn(len(normalized_signal))
        snr = compute_snr(normalized_signal, noise)
        print(f"Signal-to-Noise Ratio: {snr:.2f} dB")

        # Apply dynamic range compression and get plots
        compressed_signal = dynamic_range_compression(filtered_signal, threshold=0.5, ratio=4)

        print("\n")
        plot_time_domain(compressed_signal, sampling_rate, title=f"Compressed Signal (Amplitude vs Time) - {file_path}")

        print("\n")
        freqs, fft_values = compute_fft(compressed_signal, sampling_rate)
        plot_frequency_domain(freqs, fft_values, title=f"Frequency Spectrum of Filtered Signal - {file_path}")

        # Plot spectrogram
        print("\n")
        plot_spectrogram(filtered_signal, sampling_rate, title=f"Spectrogram of Filtered Signal - {file_path}")

        # Testing speech analysis features
        if signal.ndim == 2:  # Convert to mono if necessary
            mean_signal = np.mean(signal, axis=1)
        else:
            mean_signal = signal

        print("\n")

        # Short-term energy
        frame_size = int(0.02 * sampling_rate)  # 20ms frames
        hop_size = int(0.01 * sampling_rate)  # 10ms hop
        energy = short_term_energy(mean_signal, frame_size, hop_size)
        print(f"Short-term Energy (First 10 Values): {energy[:10]}")

        # Zero-crossing rate
        zcr = zero_crossing_rate(mean_signal, frame_size, hop_size)
        print(f"Zero-Crossing Rate (First 10 Values): {zcr[:10]}")

        # Spectral centroid
        centroid = spectral_centroid(mean_signal, sampling_rate)
        print(f"Spectral Centroid: {centroid:.2f} Hz")

        # Spectral rolloff
        rolloff = spectral_rolloff(mean_signal, sampling_rate)
        print(f"Spectral Rolloff: {rolloff:.2f} Hz")

        # MFCC computation
        mfccs = compute_mfcc(mean_signal, sampling_rate)
        print(f"MFCCs (Shape): {mfccs.shape}")

        # Harmonics-to-noise ratio
        downsampled_signal = mean_signal[::8]
        hnr = harmonics_to_noise_ratio(downsampled_signal, original_rate=sampling_rate, target_rate=sampling_rate // 8)
        print(f"Harmonics-to-Noise Ratio (HNR): {hnr:.2f} dB")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
