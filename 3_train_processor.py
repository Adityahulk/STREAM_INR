import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models import StreamProcessor
from dataset import PreprocessedMambaDataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# --- Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 # Can be larger now
NUM_EPOCHS = 50
PREPROCESSED_DIR = "./preprocessed_mamba_data/"
LATENT_DIM = 64
MAMBA_SEQ_LEN = 3600
MODEL_SAVE_PATH = "./processor.pth"

def train_processor():
    # --- 1. Data ---
    print("Loading preprocessed data...")
    full_dataset = PreprocessedMambaDataset(data_dir=PREPROCESSED_DIR)
    
    # Create train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 2. Model ---
    print("Initializing Mamba model...")
    model = StreamProcessor(seq_len=MAMBA_SEQ_LEN, latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # We must use BCEWithLogitsLoss for binary classification
    # This also handles the unbalanced dataset better
    criterion = nn.BCEWithLogitsLoss() 

    # --- 3. Training Loop ---
    print(f"Starting Mamba processor training on {DEVICE}...")
    best_val_f1 = -1.0
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        
        for i, (z_sequence, label) in enumerate(train_loader):
            z_sequence = z_sequence.to(DEVICE)
            label = label.float().to(DEVICE) # Ensure label is float
            
            optimizer.zero_grad()
            logits = model(z_sequence)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for z_sequence, label in val_loader:
                z_sequence = z_sequence.to(DEVICE)
                logits = model(z_sequence)
                
                preds = torch.sigmoid(logits).cpu().numpy() > 0.5
                all_preds.extend(preds)
                all_labels.extend(label.numpy())
        
        # Calculate metrics
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        if f1 > best_val_f1:
            print(f"New best F1 score! Saving model to {MODEL_SAVE_PATH}")
            best_val_f1 = f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Training complete.")

if __name__ == "__main__":
    train_processor()