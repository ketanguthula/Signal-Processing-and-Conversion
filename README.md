# Signal-Processing-and-Conversion

## Overview
The following is a Python-based signal processing library designed to analyze and process audio files in various formats, including WAV, MP3, and CSV. It provides functionalities for calculating common audio metrics, generating visualizations, and converting between the file formats listed above.

## Features
### Audio Metrics Computation:
- Signal-to-Noise Ratio (SNR)
- Zero-Crossing Rate (ZCR)
- Short-Term Energy
- Spectral Centroid
- Spectral Rolloff
- MFCC (Mel Frequency Cepstral Coefficients)
- Harmonics-to-Noise Ratio (HNR)

### File Format Conversions:
- Convert audio files to CSV.
- Convert from CSV to WAV/MP3.

### Visualizations:
- Time-domain plots
- Frequency-domain plots (FFT)
- Spectrograms

## Directory Structure
signal_processing_library/
|-- examples/                 # Example audio files (WAV, MP3, CSV)
|-- input/                        # File loading and conversion utilities
|   |-- file_loader.py        # Loading audio files
|   |-- convert.py            # Conversion functions (e.g., WAV to CSV, CSV to MP3)
|-- processing/               # Preprocessing and filtering
|   |-- preprocess.py         # Normalization and basic filters
|   |-- advanced_filters.py   # High-pass, band-pass filters, dynamic range compression
|-- analysis/                 # Feature extraction and metrics computation
|   |-- feature_extraction.py # All metric functions (e.g., SNR, ZCR, MFCC)
|-- transformations/          # Signal transformations (e.g., FFT)
|   |-- fft.py                # FFT and dominant frequency detection
|-- visualizations/           # Plotting utilities
|   |-- plots.py              # Time-domain, frequency-domain, and spectrogram plots
|-- metrics_main.py           # Main program for calculating metrics and handling visualizations
|-- convert_main.py           # Main program for file format conversions
|-- requirements.txt          # Project dependencies
|-- README.md                 # Project documentation
|-- LICENSE                   # License file

## Installation

Clone the repository:git clone https://github.com/yourusername/signal\_processing\_library.git

1.  cd signal\_processing\_library
    
2.  Install the required dependencies:pip install -r requirements.txt
    
3.  Ensure you have the following additional software installed:
    

FFmpeg (required for MP3 conversions):

\*sudo apt install ffmpeg # for Linux

\*brew install ffmpeg # for macOS

## Usage

### Running the Metrics Tool

The metrics\_main.py script allows you to compute metrics and generate visualizations for a given audio file.

You can run it using the common python metrics\_main.py (Note, you will have to use !python if running in a notebook environment).

You will then be prompted to enter the path of the audio file you want to analyze. As mentioned, the script supports WAV, MP3, and CSV files.

### Running the Conversion Tool

The convert\_main.py script facilitates file format conversions.

You can run it using the common !python convert\_main.py (.

Enter the file path of the file you want to convert and then the file path for where you want to save the conversion.

### Example

python metrics\_main.py

\# Enter the path: examples/Testing\_Example\_1.wav

## Example Files 
Sample audio files are provided in the examples directory for testing purposes.

## Dependencies
-   Python >= 3.7
    
-   numpy
    
-   scipy
    
-   matplotlib
    
-   librosa
    
-   pandas
    
-   pydub
    
-   FFmpeg (for MP3 conversions)
    

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing and Contact
Contributions are welcome. If you have suggestions for improvements or potential enhancements, do feel free to open an issue or submit a pull request. For any questions or feedback, please reach out to durgaketans@gmail.com or open an issue on GitHub.

