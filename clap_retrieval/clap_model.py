import os
import requests
from tqdm import tqdm
import torch
import laion_clap
from clap_module.factory import load_state_dict

CLAP_MODEL_PATH = 'load/clap_score/630k-audioset-fusion-best.pt'

def download_clap_model():
    """
    Downloads the CLAP model if it doesn't exist locally.
    """
    url = 'https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt'

    if not os.path.exists(CLAP_MODEL_PATH):
        print('Downloading CLAP model...')
        os.makedirs(os.path.dirname(CLAP_MODEL_PATH), exist_ok=True)
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(CLAP_MODEL_PATH, 'wb') as file:
            with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
                for data in response.iter_content(chunk_size=8192):
                    file.write(data)
                    progress_bar.update(len(data))
    else:
        print('CLAP model already exists. Skipping download.')

def initialize_clap_model():
    """
    Initializes and loads the CLAP model with the correct state dictionary.
    Returns:
    -- model: CLAP_Module object with pretrained weights loaded, using CUDA if available, otherwise CPU.
    """
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Download the model if needed
    download_clap_model()

    # Initialize the CLAP model
    model = laion_clap.CLAP_Module(enable_fusion=True, device=device)

    # Load model state dictionary and fix any known issues with the checkpoint
    pkg = load_state_dict(CLAP_MODEL_PATH)
    pkg.pop('text_branch.embeddings.position_ids', None)  # Remove problematic key
    model.model.load_state_dict(pkg)

    # Put the model in evaluation mode
    model.eval()

    return model
