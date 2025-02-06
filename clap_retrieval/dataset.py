from torch.utils.data import Dataset

# Custom Dataset for batching text prompts
class TextDataset(Dataset):
    def __init__(self, text_prompts):
        self.text_prompts = text_prompts

    def __len__(self):
        return len(self.text_prompts)

    def __getitem__(self, idx):
        return self.text_prompts[idx]

class AudioDataset(Dataset):
    def __init__(self, youtube_ids):
        self.youtube_ids = youtube_ids  # List of YouTube IDs

    def __len__(self):
        return len(self.youtube_ids)

    def __getitem__(self, idx):
        return self.youtube_ids[idx]