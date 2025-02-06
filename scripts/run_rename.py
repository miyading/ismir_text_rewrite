# # find /Users/dingmeiying/Documents/GitHub/music_caps_dl/audio  -type f | sed -n 's/.*\.//p' | sort | uniq -c | sort -nr
# #    5369 wav
# #    3 mp4
# #    1 part
# Step 0: Rename all audio filesfrom [ytid]-[start,end].wav to ytid.wav
import os
import re

def rename_audio_files(audio_folder):
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav"):
            # Match the pattern [ytid]-[start, end].wav
            match = re.match(r"\[([^\]]+)\]-\[[^\]]+\]\.wav", filename)
            if match:
                ytid = match.group(1)  # Extract the YouTube ID inside the first set of brackets
                new_filename = f"{ytid}.wav"  # Create the new filename
                old_file_path = os.path.join(audio_folder, filename)
                new_file_path = os.path.join(audio_folder, new_filename)

                # Rename the file
                os.rename(old_file_path, new_file_path)
                # print(f"Renamed {filename} to {new_filename}")
            else:
                print(f"Filename {filename} does not match the expected format.")
                
if __name__ == "__main__":
    # audio_folder = "/content/drive/MyDrive/Colab Notebooks/MusicCaps/audio"
    audio_folder = "/Users/dingmeiying/Documents/GitHub/music_caps_dl/audio"
    rename_audio_files(audio_folder)