import torch
import numpy as np
import mne
from collections import deque
from models import Encoder, StreamProcessor
from dataset import z_score_normalize, preprocess_eeg # Re-use preprocessing

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER_PATH = "./encoder.pth"
PROCESSOR_PATH = "./processor.pth"
CHANNELS = 23
SEQ_LEN = 256
LATENT_DIM = 64
MAMBA_SEQ_LEN = 3600 # 1 hour of 1Hz z-vectors

class StreamingDemo:
    def __init__(self):
        print("Loading models...")
        # Load Encoder
        self.encoder = Encoder(in_channels=CHANNELS, seq_len=SEQ_LEN, latent_dim=LATENT_DIM)
        self.encoder.load_state_dict(torch.load(ENCODER_PATH))
        self.encoder.to(DEVICE).eval()
        
        # Load Processor
        self.processor = StreamProcessor(seq_len=MAMBA_SEQ_LEN, latent_dim=LATENT_DIM)
        self.processor.load_state_dict(torch.load(PROCESSOR_PATH))
        self.processor.to(DEVICE).eval()
        
        # Create a buffer to hold the last 1-hour of z-vectors
        self.z_buffer = deque(maxlen=MAMBA_SEQ_LEN)
        print("Models loaded. Ready to stream.")

    def process_chunk(self, raw_1sec_chunk):
        """
        Process a single 1-second chunk of raw EEG data.
        Input: raw_1sec_chunk (numpy array, shape (23, 256))
        """
        
        # 1. Preprocess the chunk
        chunk = z_score_normalize(raw_1sec_chunk)
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        # 2. Get z-vector from Encoder
        with torch.no_grad():
            z = self.encoder(chunk_tensor) # (1, z_dim)
            
        # 3. Add to buffer and remove old
        self.z_buffer.append(z)
        
        # 4. Check if buffer is full
        if len(self.z_buffer) < MAMBA_SEQ_LEN:
            return f"Buffering... ({len(self.z_buffer)} / {MAMBA_SEQ_LEN})"
            
        # 5. Run Mamba Processor
        # Stack all z-vectors into a sequence
        # (L, 1, z_dim) -> (1, L, z_dim)
        z_sequence = torch.cat(list(self.z_buffer), dim=0).unsqueeze(0)
        
        with torch.no_grad():
            logit = self.processor(z_sequence)
            prob = torch.sigmoid(logit).item()
            
        return f"Prediction (Seizure in next 60 min): {prob * 100:.2f}%"


def run_demo():
    # --- This is a FAKE demo. In a real app, you'd get this from a live sensor. ---
    print("\n--- Starting Real-Time Streaming Demo ---")
    
    # 1. Load the demo class
    demo = StreamingDemo()
    
    # 2. Load a real EEG file to simulate a stream
    # !! YOU MUST CHANGE THIS to a real file path !!
    TEST_FILE = "./data/chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_03.edf"
    
    try:
        raw = mne.io.read_raw_edf(TEST_FILE, preload=True, verbose=False)
        data = preprocess_eeg(raw)
        total_seconds = int(data.shape[1] / SEQ_LEN)
    except Exception as e:
        print(f"Could not load demo file: {e}")
        return

    # 3. Simulate the stream, 1 second at a time
    print(f"Simulating stream from {TEST_FILE}...")
    for sec in range(total_seconds):
        start = sec * SEQ_LEN
        stop = start + SEQ_LEN
        
        chunk = data[:, start:stop]
        
        if chunk.shape[1] < SEQ_LEN:
            continue
            
        # Get the prediction for this 1-second chunk
        prediction = demo.process_chunk(chunk)
        
        # Print an update every 10 seconds
        if sec % 10 == 0:
            print(f"Time: {sec} seconds | Status: {prediction}")
            
        # For the demo, we'll just run for a bit
        if sec > (MAMBA_SEQ_LEN + 100):
            print("Demo finished.")
            break

if __name__ == "__main__":
    # In a real paper, you would add a `test()` function here that
    # loads a test set from `PreprocessedMambaDataset` and
    # computes the final F1, Precision, Recall, and AUC-ROC metrics.
    
    run_demo()