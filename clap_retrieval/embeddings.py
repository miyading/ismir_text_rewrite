import os
from tqdm import tqdm
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from clap_retrieval.dataset import TextDataset, AudioDataset
from clap_retrieval.file_utils import save_embeddings

def compute_text_embeddings(text_prompts, youtube_ids, audio_folder, model, text_embeddings, text_embedding_save_path, batch_size=16):
    """
    Compute and save embeddings for text prompts where the corresponding audio file exists.

    Args:
    -- text_prompts: A list of text prompts (in the same order as youtube_ids).
    -- youtube_ids: A list of YouTube IDs.
    -- audio_folder: The path to the folder where audio files are located.
    -- model: The CLAP model used to compute embeddings.
    -- text_embeddings: Dictionary to store text embeddings.
    -- text_embedding_save_path: Path to save the updated text embeddings.
    -- batch_size: Size of the batch for processing text prompts.
    """
    # Filter only YouTube IDs that have a corresponding audio file in the folder
    available_ytids_and_prompts = [(ytid, prompt) for ytid, prompt in zip(youtube_ids, text_prompts)
                                   if os.path.isfile(os.path.join(audio_folder, f"{ytid}.wav"))]

    # Unpack filtered YTIDs and corresponding text prompts
    available_ytids, available_prompts = zip(*available_ytids_and_prompts) if available_ytids_and_prompts else ([], [])

    # Filter out already embedded text prompts
    unseen_ytids = [ytid for ytid in available_ytids if ytid not in text_embeddings]
    unseen_prompts = [prompt for ytid, prompt in zip(available_ytids, available_prompts) if ytid not in text_embeddings]

    if unseen_ytids:
        print(f"Calculating text embeddings for {len(unseen_ytids)} available YouTube IDs...")

        # Initialize the dataset and dataloader
        text_dataset = TextDataset(unseen_prompts)
        text_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

        with torch.no_grad():
            # Iterate over batches of text prompts
            for i, batch in enumerate(tqdm(text_loader, total=len(text_loader))):
                # Batch contains text prompts
                text_batch = batch

                # Compute text embeddings using the model
                new_text_embeddings = model.get_text_embedding(text_batch)

                # Update the text_embeddings dictionary
                batch_ytids = unseen_ytids[i * batch_size: (i + 1) * batch_size]
                for ytid, emb in zip(batch_ytids, new_text_embeddings):
                    text_embeddings[ytid] = emb.tolist()

        # Save the updated text embeddings
        save_embeddings(text_embeddings, text_embedding_save_path)
    else:
        print("All text embeddings are already computed for the available files.")


def compute_audio_embeddings(audio_folder, youtube_ids, model, audio_embeddings, audio_embedding_save_path, batch_size=16):
    """
    Compute and save embeddings for audio files that exist in the folder.

    Args:
    -- audio_folder: The path to the folder where audio files are located.
    -- youtube_ids: A list of YouTube IDs.
    -- model: The CLAP model used to compute embeddings.
    -- audio_embeddings: Dictionary to store audio embeddings.
    -- audio_embedding_save_path: Path to save the updated audio embeddings.
    -- batch_size: Size of the batch for processing audio files.
    """
    # Filter only YouTube IDs for which the audio file exists in the folder
    available_audio_files = [ytid for ytid in youtube_ids if os.path.isfile(os.path.join(audio_folder, f"{ytid}.wav"))]

    # Filter out already embedded audios
    unseen_ytids = [ytid for ytid in available_audio_files if ytid not in audio_embeddings]

    if unseen_ytids:
        print(f"Calculating audio embeddings for {len(unseen_ytids)} available files...")

        # Initialize the dataset and dataloader
        audio_dataset = AudioDataset(unseen_ytids)
        audio_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: x)

        with torch.no_grad():
            first_iteration = True  # Initialize the flag

            for batch in tqdm(audio_loader, total=len(audio_loader)):
                # `batch` is a list of YouTube IDs
                ytid_batch = batch  # No need to unpack

                # Dynamically create file paths for each YouTube ID
                audio_paths_batch = [os.path.join(audio_folder, f"{ytid}.wav") for ytid in ytid_batch]

                if first_iteration:
                    print(f"Batch size: {len(ytid_batch)}")  # Print batch size
                    print("YouTube IDs:", ytid_batch)  # Print ytids for the first iteration
                    print("File Paths:", audio_paths_batch)  # Print file paths for the first iteration
                    first_iteration = False

                valid_ytids = []
                valid_audio_paths = []

                for ytid, file_path in zip(ytid_batch, audio_paths_batch):
                    if os.path.isfile(file_path):
                        valid_ytids.append(ytid)
                        valid_audio_paths.append(file_path)
                    else:
                        print(f"Warning: File does not exist for ytid '{ytid}': {file_path}")

                if valid_audio_paths:
                    # Compute audio embeddings using the model
                    new_audio_embeddings = model.get_audio_embedding_from_filelist(x=valid_audio_paths, use_tensor=True)

                    # Update the audio_embeddings dictionary
                    for ytid, emb in zip(valid_ytids, new_audio_embeddings):
                        audio_embeddings[ytid] = emb.tolist()

        # Save the updated embeddings
        save_embeddings(audio_embeddings, audio_embedding_save_path)
    else:
        print("All audio embeddings are already computed for the available files.")
