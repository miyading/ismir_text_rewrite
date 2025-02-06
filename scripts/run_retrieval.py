import os
from clap_retrieval.retrieval import find_top_k_matches_with_keywords
from clap_retrieval.file_utils import load_embeddings
from clap_retrieval.clap_model import initialize_clap_model


# Paths and variables
text_embedding_save_path = "data/text_embeddings.json"  # Path to saved text embeddings
audio_embedding_save_path = "data/audio_embeddings.json"  # Path to saved audio embeddings
csv_file = "data/musiccaps.csv"  # Path to your CSV file

# Initialize the CLAP model
model = initialize_clap_model()

# Load text embeddings
if os.path.exists(text_embedding_save_path):
    text_embeddings = load_embeddings(text_embedding_save_path)
else:
    raise FileNotFoundError(f"Text embeddings file not found: {text_embedding_save_path}. Please run 'run_embeddings.py' first to compute embeddings.")

# Load audio embeddings
if os.path.exists(audio_embedding_save_path):
    audio_embeddings = load_embeddings(audio_embedding_save_path)
else:
    raise FileNotFoundError(f"Audio embeddings file not found: {audio_embedding_save_path}. Please run 'run_embeddings.py' first to compute embeddings.")

# Load YouTube IDs, Text Prompts, and Aspect List (Keywords) from CSV
def get_text_prompts_and_keywords_from_csv(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    youtube_ids = df['ytid'].tolist()
    text_prompts = df['caption'].tolist()
    aspect_lists = df['aspect_list'].tolist()
    return youtube_ids, text_prompts, aspect_lists

youtube_ids, text_prompts, aspect_lists = get_text_prompts_and_keywords_from_csv(csv_file)

def retrieve_prompts(new_text_prompt, k=5):
    """
    Retrieve top k text and audio matches for the given text prompt.

    Args:
    -- new_text_prompt: The user-provided text prompt.
    -- k: Number of top matches to return.

    Returns:
    -- top_k_text_matches: Top k text matches with scores, YouTube IDs, prompts, and keywords.
    -- top_k_audio_matches: Top k audio matches with scores, YouTube IDs, prompts, and keywords.
    """
    top_k_text_matches, top_k_audio_matches = find_top_k_matches_with_keywords(
        new_text_prompt, text_prompts, youtube_ids, text_embeddings, audio_embeddings, aspect_lists, model, k
    )
    return top_k_text_matches, top_k_audio_matches

if __name__ == "__main__":
    # Input new text prompt
    new_text_prompt = input("Enter a new text prompt: ")

    # Retrieve prompts
    top_k_text_matches, top_k_audio_matches = retrieve_prompts(new_text_prompt)

    # Print results
    print("\nTop k text matches:")
    for score, ytid, prompt, keywords in top_k_text_matches:
        print(f"Similarity: {score:.4f}, YouTube ID: {ytid}, Prompt: {prompt}, Keywords: {keywords}")

    print("\nTop k audio matches:")
    for score, ytid, prompt, keywords in top_k_audio_matches:
        print(f"Similarity: {score:.4f}, YouTube ID: {ytid}, Prompt: {prompt}, Keywords: {keywords}")
