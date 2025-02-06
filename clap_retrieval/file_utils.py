import json
import os
import pandas as pd

def save_embeddings(embeddings, file_path):
    with open(file_path, 'w') as f:
        json.dump(embeddings, f)

def load_embeddings(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

# Step 1: Read Captions and find Available Audio Files

def get_text_prompts_from_csv(csv_file):
    """
    Read the caption column from the CSV file and return a list of text prompts.

    Args:
    -- csv_file: Path to the CSV file (e.g., 'musiccaps-public.csv').

    Returns:
    -- List of text prompts and corresponding YouTube IDs (ytid).
    """
    df = pd.read_csv(csv_file)
    text_prompts = df['caption'].tolist()  # Convert the 'caption' column to a list
    youtube_ids = df['ytid'].tolist()      # Extract YouTube IDs
    return youtube_ids, text_prompts

def get_available_audio_files(audio_folder, youtube_ids):
    available_audio_files = []
    found_ytids = set()  # Keep track of found YouTube IDs

    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav"):
            # Extract the YouTube ID and remove any surrounding brackets
            ytid = os.path.splitext(filename)[0].strip("[]")
            if ytid in youtube_ids:
                file_path = os.path.join(audio_folder, filename)
                available_audio_files.append((ytid, file_path))
                found_ytids.add(ytid)
            else:
                print(f"Filename {filename} has a YouTube ID not in the provided list: {ytid}")

    # Find the missing YouTube IDs by checking those not found in the folder
    missing_ytids = [ytid for ytid in youtube_ids if ytid not in found_ytids]

    # print(f"Available Audio Files: {available_audio_files}")
    print(f"Number of Available Audio Files: {len(available_audio_files)}")
    # print(f"Missing YouTube IDs: {missing_ytids}")
    print(f"Number of Missing YouTube IDs: {len(missing_ytids)}")

    return available_audio_files, missing_ytids