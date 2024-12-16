import sys
from input.convert import audio_to_csv, convert_csv_to_wav, convert_csv_to_mp3


def main():
    print("Welcome to the Audio Conversion Tool!")
    print("Choose an option:")
    print("1. Convert audio to CSV")
    print("2. Convert CSV to WAV")
    print("3. Convert CSV to MP3")

    try:
        choice = int(input("Enter your choice (1/2/3): "))
        if choice == 1:
            input_path = input("Enter the path of the audio file: ")
            output_path = input("Enter the path to save the CSV file: ")
            file_type = input("Enter the audio file type ('wav' or 'mp3'): ").lower()
            audio_to_csv(input_path, output_path, file_type)
        elif choice == 2:
            input_path = input("Enter the path of the CSV file: ")
            output_path = input("Enter the path to save the WAV file: ")
            convert_csv_to_wav(input_path, output_path)
        elif choice == 3:
            input_path = input("Enter the path of the CSV file: ")
            output_path = input("Enter the path to save the MP3 file: ")
            convert_csv_to_mp3(input_path, output_path)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
