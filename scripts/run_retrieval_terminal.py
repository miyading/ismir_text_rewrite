import os
from clap_retrieval.retrieval import find_top_k_matches_with_keywords
from clap_retrieval.file_utils import load_embeddings
from clap_retrieval.clap_model import initialize_clap_model

# Paths and variables
text_embedding_save_path = "data/text_embeddings.json"  # Path to saved text embeddings
audio_embedding_save_path = "data/audio_embeddings.json"  # Path to saved audio embeddings

# Initialize the CLAP model
model = initialize_clap_model()

# Load text embeddings if the file exists, otherwise raise an error
if os.path.exists(text_embedding_save_path):
    text_embeddings = load_embeddings(text_embedding_save_path)
    print(f"Loaded existing text embeddings from {text_embedding_save_path}")
else:
    raise FileNotFoundError(f"Text embeddings file not found: {text_embedding_save_path}. Please run 'run_embeddings.py' first to compute embeddings.")

# Load audio embeddings if the file exists, otherwise raise an error
if os.path.exists(audio_embedding_save_path):
    audio_embeddings = load_embeddings(audio_embedding_save_path)
    print(f"Loaded existing audio embeddings from {audio_embedding_save_path}")
else:
    raise FileNotFoundError(f"Audio embeddings file not found: {audio_embedding_save_path}. Please run 'run_embeddings.py' first to compute embeddings.")

def get_text_prompts_and_keywords_from_csv(csv_file):
    """
    Load YouTube IDs, text prompts, and aspect list (keywords) from a CSV file.

    Args:
    -- csv_file: Path to the CSV file containing 'ytid', 'caption', and 'aspect_list'.

    Returns:
    -- youtube_ids: A list of YouTube IDs.
    -- text_prompts: A list of text prompts corresponding to YouTube IDs.
    -- aspect_lists: A list of keywords (aspect_list) corresponding to YouTube IDs.
    """
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    youtube_ids = df['ytid'].tolist()         # Extract YouTube IDs
    text_prompts = df['caption'].tolist()     # Extract text prompts (captions)
    aspect_lists = df['aspect_list'].tolist() # Extract aspect lists (keywords)

    return youtube_ids, text_prompts, aspect_lists
# Load YouTube IDs, Text Prompts, and Aspect List (Keywords) from CSV
csv_file = "data/musiccaps.csv"  # Path to your CSV file
youtube_ids, text_prompts, aspect_lists = get_text_prompts_and_keywords_from_csv(csv_file)

# Input new text prompt
new_text_prompt = input("Enter a new text prompt: ")

# Find top k matches for the new text prompt
top_k_text_matches, top_k_audio_matches = find_top_k_matches_with_keywords(
    new_text_prompt, text_prompts, youtube_ids, text_embeddings, audio_embeddings, aspect_lists, model, k=5
)

# Output the top k text matches
print("\nTop k text matches:")
for score, ytid, prompt, keywords in top_k_text_matches:
    print(f"Similarity: {score:.4f}, YouTube ID: {ytid}, Prompt: {prompt}, Keywords: {keywords}")

# Output the top k audio matches
print("\nTop k audio matches:")
for score, ytid, prompt, keywords in top_k_audio_matches:
    print(f"Similarity: {score:.4f}, YouTube ID: {ytid}, Prompt: {prompt}, Keywords: {keywords}")

print("Retrieval complete!")
