import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models import Autoencoder
from dataset import AutoencoderDataset
import argparse

# --- Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 128
NUM_EPOCHS = 20
DATA_DIR = "./data/chb-mit-scalp-eeg-database-1.0.0" # CHANGE THIS
MODEL_SAVE_PATH = "./encoder.pth"
CHANNELS = 23 # Based on our preprocessing
SEQ_LEN = 256 # 1 second at 256Hz
LATENT_DIM = 64

def train_autoencoder():
    # --- 1. Data ---
    print("Loading data...")
    dataset = AutoencoderDataset(root_dir=DATA_DIR, chunk_size=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # --- 2. Model ---
    print("Initializing model...")
    model = Autoencoder(in_channels=CHANNELS, seq_len=SEQ_LEN, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- 3. Training Loop ---
    print(f"Starting autoencoder training on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        
        for i, batch in enumerate(loader):
            batch = batch.to(DEVICE) # (B, C, L)
            
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i}/{len(loader)}, Loss: {loss.item():.6f}")
                
        avg_loss = total_loss / len(loader)
        print(f"--- Epoch {epoch+1} Complete --- Avg. Loss: {avg_loss:.6f} ---")

    # --- 4. Save the Encoder ---
    print("Training complete. Saving encoder model...")
    torch.save(model.encoder.state_dict(), MODEL_SAVE_PATH)
    print(f"Encoder saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_autoencoder()