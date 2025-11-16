import torch
import os
from models import Encoder
from dataset import ProcessorSequenceDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./data/chb-mit-scalp-eeg-database-1.0.0" # CHANGE THIS
ENCODER_PATH = "./encoder.pth"
SAVE_DIR = "./preprocessed_mamba_data/"
CHANNELS = 23
SEQ_LEN = 256
LATENT_DIM = 64
MAMBA_SEQ_LEN = 3600 # 1 hour of 1Hz z-vectors

def preprocess():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- 1. Load Pre-trained Encoder ---
    print("Loading pre-trained encoder...")
    encoder = Encoder(in_channels=CHANNELS, seq_len=SEQ_LEN, latent_dim=LATENT_DIM)
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    encoder.to(DEVICE).eval() # Set to eval mode!

    # --- 2. Data ---
    print("Loading sequence index...")
    # This dataset is a generator for *raw* sequences
    dataset = ProcessorSequenceDataset(root_dir=DATA_DIR, seq_len=MAMBA_SEQ_LEN)
    # Use batch_size=1 because each item is huge and processed sequentially
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # --- 3. Preprocessing Loop ---
    print(f"Starting preprocessing. Saving files to {SAVE_DIR}")
    
    for i, (raw_sequence, label) in enumerate(tqdm(loader)):
        # raw_sequence: (1, L, C, Samples) -> (1, 3600, 23, 256)
        # label: (1,)
        
        if label.item() == -1: # Skip dummy data
            continue
            
        raw_sequence = raw_sequence.squeeze(0).to(DEVICE) # (L, C, Samples)
        label = label.to(DEVICE)
        
        z_sequence = []
        # Process 1 second at a time (in batches for speed)
        with torch.no_grad():
            for chunk in torch.split(raw_sequence, batch_size=64):
                # chunk: (64, C, Samples)
                z = encoder(chunk) # (64, z_dim)
                z_sequence.append(z)
        
        # z_sequence: (L, z_dim) -> (3600, 64)
        z_sequence = torch.cat(z_sequence, dim=0)
        
        # Save the preprocessed (tensor, label) pair
        save_path = os.path.join(SAVE_DIR, f"seq_{i:05d}.pt")
        torch.save((z_sequence.cpu(), label.cpu()), save_path)
        
    print(f"Preprocessing complete. {len(loader)} sequences saved.")

if __name__ == "__main__":
    preprocess()