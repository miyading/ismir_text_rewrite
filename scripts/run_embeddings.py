import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clap_retrieval.embeddings import compute_text_embeddings, compute_audio_embeddings
from clap_retrieval.file_utils import load_embeddings, save_embeddings
from clap_retrieval.clap_model import initialize_clap_model

# Paths and variables
csv_file = "data/musiccaps.csv"  # Path to CSV file
audio_folder = "/Users/dingmeiying/Documents/GitHub/music_caps_dl/audio"  # Folder where audio files are stored
text_embedding_save_path = "data/text_embeddings.json"  # Where to save text embeddings
audio_embedding_save_path = "data/audio_embeddings.json"  # Where to save audio embeddings
batch_size = 16  # Set the batch size

# Initialize the CLAP model
model = initialize_clap_model()

# Load embeddings if the files exist, otherwise initialize empty dictionaries
if os.path.exists(text_embedding_save_path):
    text_embeddings = load_embeddings(text_embedding_save_path)
    print(f"Loaded existing text embeddings from {text_embedding_save_path}")
else:
    text_embeddings = {}
    print(f"No text embeddings file found. Starting with an empty dictionary.")

if os.path.exists(audio_embedding_save_path):
    audio_embeddings = load_embeddings(audio_embedding_save_path)
    print(f"Loaded existing audio embeddings from {audio_embedding_save_path}")
else:
    audio_embeddings = {}
    print(f"No audio embeddings file found. Starting with an empty dictionary.")

# Load text prompts and YouTube IDs from CSV
def get_text_prompts_from_csv(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    text_prompts = df['caption'].tolist()  # Convert the 'caption' column to a list
    youtube_ids = df['ytid'].tolist()      # Extract YouTube IDs
    return youtube_ids, text_prompts

# Load YouTube IDs and text prompts
youtube_ids, text_prompts = get_text_prompts_from_csv(csv_file)

# Compute CLAP embeddings for text prompts (if not already done)
compute_text_embeddings(text_prompts, youtube_ids, audio_folder, model, text_embeddings, text_embedding_save_path, batch_size=batch_size)

# Compute CLAP embeddings for audio files (if not already done)
compute_audio_embeddings(audio_folder, youtube_ids, model, audio_embeddings, audio_embedding_save_path, batch_size=batch_size)

# Save embeddings to disk
# save_embeddings(text_embeddings, text_embedding_save_path)
# save_embeddings(audio_embeddings, audio_embedding_save_path)

print("Embedding computation complete!")
