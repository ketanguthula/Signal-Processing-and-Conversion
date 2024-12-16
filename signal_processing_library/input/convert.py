import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write
import librosa
from pathlib import Path
from pydub import AudioSegment


def audio_to_csv(input_path, output_path, file_type='wav'):
    """
    Convert an audio file (.wav or .mp3) to a CSV file

    Args:
        input_path (str): path to the input audio file
        output_path (str): path to save the output CSV file
        file_type (str): type of the audio file ('wav' or 'mp3'). Default is 'wav'

    Returns:
        None
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The file '{input_path}' does not exist.")

    if file_type.lower() == 'wav':
        # Read the WAV file
        sampling_rate, signal = wavfile.read(input_path)
    elif file_type.lower() == 'mp3':
        # Read the MP3 file using librosa
        signal, sampling_rate = librosa.load(input_path, sr=None)
    else:
        raise ValueError(f"Unsupported file type '{file_type}'. Supported types are 'wav' and 'mp3'.")

    # Convert signal to DataFrame
    if signal.ndim == 1:  # Mono
        df = pd.DataFrame(signal, columns=["Mono"])
    else:  # Stereo
        df = pd.DataFrame(signal, columns=["Left", "Right"])

    # Add metadata columns
    df['Sampling Rate'] = sampling_rate
    df['Time (s)'] = np.arange(len(signal)) / sampling_rate

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"CSV saved at {output_path}")


def convert_csv_to_wav(input_csv_path, output_wav_path):
    """
    Convert a CSV file containing audio data to a WAV file

    Args:
        input_csv_path (str): path to the input CSV file
        output_wav_path (str): path to save the output WAV file

    Returns:
        None
    """
    # Read the CSV file
    print(f"Reading CSV file: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # Extract audio data
    if "Mono" in df.columns:
        signal = df["Mono"].to_numpy()
    elif "Left" in df.columns and "Right" in df.columns:
        left = df["Left"].to_numpy()
        right = df["Right"].to_numpy()
        signal = np.column_stack((left, right))  # Combine for stereo
    else:
        raise ValueError("CSV file must contain 'Mono' or 'Left' and 'Right' columns.")

    # Extract sampling rate
    sampling_rate = int(df["Sampling Rate"].iloc[0])  # Assuming uniform sampling rate
    print(f"Detected sampling rate: {sampling_rate} Hz")

    # Write to WAV file
    print(f"Converting CSV to WAV: {output_wav_path}")
    write(output_wav_path, sampling_rate, signal.astype(np.int16))
    print(f"WAV file saved successfully at {output_wav_path}")


def convert_csv_to_mp3(input_csv_path, output_mp3_path):
    """
    Convert a CSV file to an MP3 file

    Args:
        input_csv_path (str): path to the input CSV file containing audio data
        output_mp3_path (str): path to save the output MP3 file

    Returns:
        None
    """
    # Temporary WAV file path
    temp_wav_path = "temp.wav"

    try:
        print(f"Reading CSV file: {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        # Ensure the necessary columns exist
        if 'Sampling Rate' not in df.columns or ('Mono' not in df.columns and 'Left' not in df.columns):
            raise ValueError("CSV file must contain 'Sampling Rate' and audio data columns ('Mono', 'Left', or 'Right').")

        # Extract sampling rate
        sampling_rate = int(df['Sampling Rate'].iloc[0])
        print(f"Detected sampling rate: {sampling_rate} Hz")

        # Prepare signal
        if 'Mono' in df.columns:
            signal = df['Mono'].values
        else:
            signal = df[['Left', 'Right']].values

        # Save signal as WAV for now
        print(f"Converting CSV to WAV: {temp_wav_path}")
        write(temp_wav_path, sampling_rate, signal.astype(np.int16))

        # Convert WAV to MP3
        print(f"Converting WAV to MP3: {output_mp3_path}")
        audio = AudioSegment.from_wav(temp_wav_path)
        audio.export(output_mp3_path, format="mp3")
        print(f"MP3 file saved successfully at {output_mp3_path}")

    finally:
        # Cleanup the temporary WAV file
        import os
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

