import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# def get_k_highest_similarities(new_embedding, all_embeddings, k=5):
#     """
#     Calculate cosine similarity between the new embedding and all saved embeddings,
#     and return the top k similar YouTube IDs.

#     Args:
#     -- new_embedding: The new embedding to compare against saved embeddings.
#     -- all_embeddings: Dictionary of saved embeddings {ytid: embedding}.
#     -- k: Number of top similar items to return.

#     Returns:
#     -- List of tuples [(similarity_score, ytid)], sorted by similarity score.
#     """
#     ytids = list(all_embeddings.keys())
#     embeddings = np.array(list(all_embeddings.values()))  # Convert all embeddings to a numpy array

#     # Ensure new_embedding is also a numpy array
#     new_embedding = np.array(new_embedding)

#     # Compute cosine similarities
#     similarities = cosine_similarity([new_embedding], embeddings)[0]

#     # Get top k similar embeddings
#     top_k_indices = similarities.argsort()[-k:][::-1]  # Get the k largest values and reverse

#     # Return the similarity score and corresponding YouTube ID
#     return [(similarities[i], ytids[i]) for i in top_k_indices]

def find_top_k_matches_with_keywords(new_text, text_prompts, youtube_ids, text_embeddings, audio_embeddings, aspect_lists, model, k=5):
    """
    Find the top k matches for the new text prompt and include aspect lists (keywords).

    Args:
    -- new_text: The new text prompt to compare against saved embeddings.
    -- text_prompts: List of text prompts.
    -- youtube_ids: List of YouTube IDs.
    -- text_embeddings: Dictionary of saved text embeddings {ytid: embedding}.
    -- audio_embeddings: Dictionary of saved audio embeddings {ytid: embedding}.
    -- aspect_lists: List of aspect lists (keywords) corresponding to YouTube IDs.
    -- model: The CLAP model used to compute embeddings.
    -- k: Number of top matches to return.

    Returns:
    -- top_k_text_matches: List of top k matches in text embeddings [(similarity_score, ytid, text_prompt, keywords)].
    -- top_k_audio_matches: List of top k matches in audio embeddings [(similarity_score, ytid, text_prompt, keywords)].
    """
    # Step 1: Compute the embedding for the new text prompt
    new_text_embedding = model.get_text_embedding([new_text])[0]

    # Step 2: Find top k matches in the text embeddings
    top_k_text_matches = get_k_highest_similarities_with_prompts_and_keywords(
        new_text_embedding, text_embeddings, text_prompts, youtube_ids, aspect_lists, k=k
    )

    # Step 3: Find top k matches in the audio embeddings
    top_k_audio_matches = get_k_highest_similarities_with_prompts_and_keywords(
        new_text_embedding, audio_embeddings, text_prompts, youtube_ids, aspect_lists, k=k
    )

    return top_k_text_matches, top_k_audio_matches

def get_k_highest_similarities_with_prompts_and_keywords(new_embedding, all_embeddings, text_prompts, youtube_ids, aspect_lists, k=5):
    """
    Calculate cosine similarity and return similarity score, ytid, text_prompt, and keywords.

    Args:
    -- new_embedding: The new embedding to compare against saved embeddings.
    -- all_embeddings: Dictionary of saved embeddings {ytid: embedding}.
    -- text_prompts: List of text prompts corresponding to YouTube IDs.
    -- youtube_ids: List of YouTube IDs corresponding to embeddings.
    -- aspect_lists: List of aspect lists (keywords) corresponding to YouTube IDs.
    -- k: Number of top similar items to return.

    Returns:
    -- List of tuples [(similarity_score, ytid, text_prompt, keywords)], sorted by similarity score.
    """
    ytids = list(all_embeddings.keys())
    embeddings = np.array(list(all_embeddings.values()))  # Convert all embeddings to a numpy array

    # Ensure new_embedding is also a numpy array
    new_embedding = np.array(new_embedding)

    # Compute cosine similarities
    similarities = cosine_similarity([new_embedding], embeddings)[0]

    # Get top k similar embeddings
    top_k_indices = similarities.argsort()[-k:][::-1]  # Get the k largest values and reverse

    # Return the similarity score, YouTube ID, corresponding text prompt, and keywords
    return [
        (
            similarities[i], 
            ytids[i], 
            text_prompts[youtube_ids.index(ytids[i])], 
            aspect_lists[youtube_ids.index(ytids[i])]  # Get keywords for the YouTube ID
        ) 
        for i in top_k_indices
    ]
