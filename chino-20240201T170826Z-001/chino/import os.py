import os
from pydub import AudioSegment

def convert_m4a_to_mp3(m4a_file, mp3_file):
    audio = AudioSegment.from_file(m4a_file, format="m4a")
    audio.export(mp3_file, format="mp3")

def batch_convert_m4a_to_mp3(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".m4a"):
            m4a_file = os.path.join(directory, filename)
            mp3_file = os.path.join(directory, filename[:-4] + ".mp3")
            convert_m4a_to_mp3(m4a_file, mp3_file)

# Replace '/path/to/your/directory' with the path to your directory containing M4A files
batch_convert_m4a_to_mp3(r'C:\Users\lenvo\Downloads\Rec-20240201T043211Z-001\Rec')

