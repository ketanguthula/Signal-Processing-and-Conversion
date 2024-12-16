import os
import librosa
import pandas as pd
import numpy as np


def load_audio(file_path):
    """
    Load an audio file (WAV, MP3, or CSV)

    Args:
        file_path (str): path to the audio file

    Returns:
        tuple: (sampling_rate, signal)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension in ['.wav', '.mp3']:
        # Load WAV or MP3 using librosa
        try:
            signal, sampling_rate = librosa.load(file_path, sr=None, mono=False)
            # Convert stereo signals to separate channels
            if signal.ndim == 1:
                signal = signal  # Mono
            elif signal.ndim == 2:
                signal = signal.T  # Stereo
        except Exception as e:
            raise ValueError(f"Error loading audio file: {e}")

        print(f"Loaded file: {file_path}")
        print(f"Signal length: {signal.shape[0]} samples")
        print(f"Sampling rate: {sampling_rate} Hz")
        return sampling_rate, signal

    elif file_extension == '.csv':
        # Load CSV using pandas
        try:
            df = pd.read_csv(file_path)
            # Check required columns
            if 'Sampling Rate' not in df.columns or 'Time (s)' not in df.columns:
                raise ValueError("CSV must contain 'Sampling Rate' and 'Time (s)' columns.")
            signal = df.drop(columns=['Sampling Rate', 'Time (s)']).to_numpy()
            sampling_rate = df['Sampling Rate'].iloc[0]
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")

        print(f"Loaded file: {file_path}")
        print(f"Signal length: {signal.shape[0]} samples")
        print(f"Sampling rate: {sampling_rate} Hz")
        return sampling_rate, signal

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def get_signal_metadata(sampling_rate, signal):
    """
    Retrieve metadata about the signal

    Args:
        sampling_rate (int): sampling rate of the signal
        signal (ndarray): the audio signal

    Returns:
        dict: metadata about the signal
    """
    if signal.ndim == 1:
        duration = len(signal) / sampling_rate
        num_channels = 1
    else:
        duration = signal.shape[0] / sampling_rate
        num_channels = signal.shape[1]

    metadata = {
        "duration (seconds)": duration,
        "number of channels": num_channels,
        "sampling rate (Hz)": sampling_rate
    }
    return metadata
