from clap_retrieval.retrieval import find_top_k_matches_with_keywords
from clap_retrieval.file_utils import load_embeddings
from clap_retrieval.clap_model import initialize_clap_model
import pandas as pd

def get_text_prompts_and_keywords_from_csv(csv_file):
    """Load YouTube IDs, text prompts, and aspect lists (keywords) from CSV."""
    df = pd.read_csv(csv_file)
    youtube_ids = df['ytid'].tolist()
    text_prompts = df['caption'].tolist()
    aspect_lists = df['aspect_list'].tolist()
    return youtube_ids, text_prompts, aspect_lists

def retrieve_matches(new_text_prompt, csv_file, text_embedding_save_path, audio_embedding_save_path, k=5):
    """
    Retrieve top k matches for a given text prompt.

    Args:
    -- new_text_prompt: The text prompt input from the user.
    -- csv_file: Path to the CSV file containing 'ytid', 'caption', and 'aspect_list'.
    -- text_embedding_save_path: Path to saved text embeddings.
    -- audio_embedding_save_path: Path to saved audio embeddings.
    -- k: Number of top matches to retrieve.

    Returns:
    -- top_k_text_matches: List of top k text matches with keywords [(similarity_score, ytid, prompt, keywords)].
    -- top_k_audio_matches: List of top k audio matches with keywords [(similarity_score, ytid, prompt, keywords)].
    """
    # Initialize the CLAP model
    model = initialize_clap_model()

    # Load the text, audio embeddings, and keywords (aspect_list)
    youtube_ids, text_prompts, aspect_lists = get_text_prompts_and_keywords_from_csv(csv_file)
    text_embeddings = load_embeddings(text_embedding_save_path)
    audio_embeddings = load_embeddings(audio_embedding_save_path)

    # Find top k matches for the new text prompt
    top_k_text_matches, top_k_audio_matches = find_top_k_matches_with_keywords(
        new_text_prompt, text_prompts, youtube_ids, text_embeddings, audio_embeddings, aspect_lists, model, k
    )
    
    return top_k_text_matches, top_k_audio_matches
