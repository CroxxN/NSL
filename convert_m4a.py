from pydub import AudioSegment
import os

def convert_mp3_m4a_to_mp3(file_path):
    # Load the mp3.m4a file
    sound = AudioSegment.from_file(file_path, format="m4a")

    # Change the file extension to mp3
    mp3_file_path = os.path.splitext(file_path)[0] + ".mp3"

    # Export the audio as mp3
    sound.export(mp3_file_path, format="mp3")

    return mp3_file_path

def convert_all_mp3_m4a_to_mp3(directory_path):
    # Get all mp3.m4a files in the specified directory
    mp3_m4a_files = [file for file in os.listdir(directory_path) if file.endswith(".mp3.m4a")]

    # Convert each mp3.m4a file to mp3
    for mp3_m4a_file in mp3_m4a_files:
        mp3_m4a_file_path = os.path.join(directory_path, mp3_m4a_file)
        new_mp3_file_path = convert_mp3_m4a_to_mp3(mp3_m4a_file_path)

        # Rename the file to remove the ".m4a" extension
        os.rename(new_mp3_file_path, os.path.splitext(new_mp3_file_path)[0] + ".mp3")

    print(f"Conversion completed for {len(mp3_m4a_files)} files.")

if __name__ == "__main__":
    # Specify the directory path where mp3.m4a files are located
    directory_path = "./"

    # Convert all mp3.m4a files to mp3
    convert_all_mp3_m4a_to_mp3(directory_path)