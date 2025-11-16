import torch
import mne
import numpy as np
import os
import re
from torch.utils.data import Dataset
from glob import glob

# --- EEG Preprocessing Helper ---
def preprocess_eeg(raw, sampling_rate=256):
    """Applies basic filtering and resampling."""
    # CHB-MIT is already 256Hz, so no resampling needed.
    # Apply a band-pass filter (e.g., 1Hz to 45Hz)
    raw.filter(1., 45., fir_design='firwin', verbose=False)
    
    # Select only EEG channels (e.g., exclude 'ECG', 'VAB')
    # This list is from CHB-MIT's website. You may need to adjust.
    eeg_channels = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8', 'T8-P8'
    ]
    
    # Find which of these channels are *actually* in the current file
    available_channels = [ch for ch in eeg_channels if ch in raw.ch_names]
    
    # If we have too few channels, we might skip the file
    if len(available_channels) < 18: # Hyperparameter
        return None 
        
    raw.pick_channels(available_channels)
    
    # Some files (like chb24) are missing channels. We need a fixed size.
    # We will pad with zeros up to 23 channels.
    data = raw.get_data()
    if data.shape[0] < 23:
        pad = np.zeros((23 - data.shape[0], data.shape[1]))
        data = np.concatenate([data, pad], axis=0)
    
    # Make sure it's exactly 23 channels (truncate if more, pad if less)
    data = data[:23, :]
    
    return data

def z_score_normalize(chunk):
    """Applies z-score normalization per channel."""
    mean = np.mean(chunk, axis=1, keepdims=True)
    std = np.std(chunk, axis=1, keepdims=True)
    return (chunk - mean) / (std + 1e-6)

# --- Part 1: Dataset for Autoencoder ---
# This dataset just returns 1-second chunks of *any* non-seizure EEG data.

class AutoencoderDataset(Dataset):
    def __init__(self, root_dir, chunk_size=256, sampling_rate=256):
        self.chunk_size = chunk_size
        self.sampling_rate = sampling_rate
        self.file_list = glob(os.path.join(root_dir, 'chb*/*.edf'))
        
        # Load seizure times so we can *avoid* them
        self.seizure_times = self._parse_all_summaries(root_dir)
        
        self.chunks_index = self._build_index()

    def _parse_all_summaries(self, root_dir):
        # This is a simplification. You'd parse the summary.txt files.
        # For this example, we'll assume a dummy function.
        # In a real paper, this parsing is a critical step.
        print("Parsing seizure summaries (dummy)...")
        # Format: { 'chb01_01.edf': [(start_sec, end_sec), ...], ... }
        return {
            'chb01_01.edf': [(2996, 3036)],
            'chb01_02.edf': [(1467, 1494)]
        } # !! YOU MUST IMPLEMENT THIS PARSING !!

    def _build_index(self):
        print("Building autoencoder chunk index...")
        index = []
        for filepath in self.file_list:
            filename = os.path.basename(filepath)
            seizures = self.seizure_times.get(filename, [])
            
            try:
                raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
            except:
                continue # Skip broken files
                
            total_seconds = int(raw.n_times / self.sampling_rate)
            
            for sec in range(total_seconds):
                # Check if this second is inside a seizure (or 1 min buffer)
                is_safe = True
                for (start, end) in seizures:
                    if sec >= (start - 60) and sec <= (end + 60):
                        is_safe = False
                        break
                
                if is_safe:
                    index.append((filepath, sec))
        print(f"Found {len(index)} 1-second 'safe' chunks.")
        return index

    def __len__(self):
        return len(self.chunks_index)

    def __getitem__(self, idx):
        filepath, second = self.chunks_index[idx]
        
        start_sample = second * self.sampling_rate
        stop_sample = start_sample + self.chunk_size
        
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        data = preprocess_eeg(raw) # (23, Total_Samples)
        
        if data is None:
            # Return a dummy chunk if preprocessing fails
            return torch.zeros((23, self.chunk_size), dtype=torch.float32)
        
        chunk = data[:, start_sample:stop_sample]
        
        # Ensure chunk is the right size (e.g., if it's at the end of the file)
        if chunk.shape[1] < self.chunk_size:
            pad = np.zeros((chunk.shape[0], self.chunk_size - chunk.shape[1]))
            chunk = np.concatenate([chunk, pad], axis=1)
            
        chunk = z_score_normalize(chunk)
        return torch.tensor(chunk, dtype=torch.float32)


# --- Part 2: Dataset for Mamba Processor ---
# This dataset returns *sequences* of z-vectors and their labels.

class ProcessorSequenceDataset(Dataset):
    def __init__(self, root_dir, pre_ictal_window=3600, safe_window=4*3600, seq_len=3600):
        """
        pre_ictal_window: Seconds before seizure to label as "pre-ictal" (1 hour)
        safe_window: Seconds of buffer around seizure to label "inter-ictal" (4 hours)
        seq_len: Length of sequence to return (1 hour)
        """
        self.root_dir = root_dir
        self.pre_ictal_window = pre_ictal_window
        self.safe_window = safe_window
        self.seq_len = seq_len
        self.sampling_rate = 256
        
        # This is the critical, hard part. You must parse this.
        self.seizure_times = self._parse_all_summaries(root_dir)
        
        self.sequence_index = self._build_index()

    def _parse_all_summaries(self, root_dir):
        # !! THIS IS A DUMMY. YOU MUST IMPLEMENT THIS. !!
        # Find all summary.txt files, parse them to get file names and seizure times.
        print("Parsing seizure summaries (dummy)...")
        # Format: { 'chb01': { 'chb01_01.edf': [(start, end)], 
        #                     'chb01_02.edf': [(start, end)] }, ... }
        return {
            'chb01': {
                'chb01_03.edf': [(1000, 1050)],
                'chb01_04.edf': [(2000, 2050)]
            },
            'chb02': {
                'chb02_16.edf': [(8000, 8050)],
                'chb02_19.edf': [(9000, 9050)]
            }
        }

    def _build_index(self):
        print("Building processor sequence index...")
        index = []
        for patient, files in self.seizure_times.items():
            all_patient_seizures = []
            for filename, seizures in files.items():
                all_patient_seizures.extend(seizures)
            
            for filename, seizures in files.items():
                filepath = os.path.join(self.root_dir, patient, filename)
                if not os.path.exists(filepath):
                    continue
                    
                try:
                    raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
                    total_seconds = int(raw.n_times / self.sampling_rate)
                except:
                    continue
                
                # --- 1. Add Pre-ictal sequences (Label=1) ---
                for (start_sec, end_sec) in seizures:
                    # We need a full `seq_len` window *ending* at the seizure
                    if start_sec >= self.seq_len:
                        seq_start = start_sec - self.seq_len
                        index.append((filepath, seq_start, 1))
                
                # --- 2. Add Inter-ictal sequences (Label=0) ---
                # This is harder. We need to find "safe" windows.
                # We can just sample random start times and check if they are "safe".
                for _ in range(20): # Try to find 20 safe windows per file
                    seq_start = np.random.randint(0, total_seconds - self.seq_len)
                    seq_end = seq_start + self.seq_len
                    
                    is_safe = True
                    for (sz_start, sz_end) in all_patient_seizures:
                        # Check if the sequence overlaps with a "danger zone"
                        if seq_end > (sz_start - self.safe_window) and \
                           seq_start < (sz_end + self.safe_window):
                            is_safe = False
                            break
                    
                    if is_safe:
                        index.append((filepath, seq_start, 0))

        print(f"Found {len(index)} total sequences.")
        print(f"  Pre-ictal (1): {sum(1 for _,_,L in index if L==1)}")
        print(f"  Inter-ictal (0): {sum(1 for _,_,L in index if L==0)}")
        return index

    def __len__(self):
        return len(self.sequence_index)
    
    def __getitem__(self, idx):
        # This function returns the *raw* data for a sequence.
        # It will be *very slow* to use directly for training.
        # We will use `2_preprocess_mamba.py` to pre-compute these.
        
        filepath, start_second, label = self.sequence_index[idx]
        
        try:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            data = preprocess_eeg(raw) # (23, Total_Samples)
        except:
            # Return dummy data on error
            return torch.zeros((self.seq_len, 23, 256)), -1
            
        if data is None:
            return torch.zeros((self.seq_len, 23, 256)), -1

        sequence_chunks = []
        for i in range(self.seq_len):
            second = start_second + i
            start_sample = second * self.sampling_rate
            stop_sample = start_sample + self.sampling_rate
            
            if stop_sample > data.shape[1]:
                # Not enough data, break early
                break
                
            chunk = data[:, start_sample:stop_sample]
            chunk = z_score_normalize(chunk)
            sequence_chunks.append(chunk)
        
        # If we didn't get a full sequence, pad with zeros
        while len(sequence_chunks) < self.seq_len:
            sequence_chunks.append(np.zeros((23, 256)))
            
        # (L, C, Samples) -> (3600, 23, 256)
        sequence = np.stack(sequence_chunks, axis=0)
        
        return torch.tensor(sequence, dtype=torch.float32), label


# --- Part 3: Dataset for *FAST* Mamba Training ---
# This dataset just loads the pre-processed .pt files.

class PreprocessedMambaDataset(Dataset):
    def __init__(self, data_dir):
        self.file_list = glob(os.path.join(data_dir, "*.pt"))
    
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        # Loads a tuple of (z_sequence_tensor, label_tensor)
        return torch.load(self.file_list[idx])