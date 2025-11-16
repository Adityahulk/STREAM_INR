## 🔬 Roadmap to A* Publication (To-Do List)

This project has the potential for a high-impact, A* publication. To get there, we must be rigorous. This list tracks the key experiments and milestones required.

### Phase 1: Model & Data Engineering (Completed)
- [x] **Define Core Architecture:** `Stream-INR` (SIREN Autoencoder + Mamba Processor)
- [x] **Select Benchmark Dataset:** CHB-MIT Scalp EEG Database
- [x] **Setup Project:** Create modular code for `models.py`, `dataset.py`, and training scripts.
- [x] **Implement Data Pipeline:** Write `mne`-based loaders for `.edf` files and parsers for `chbXX-summary.txt` annotation files.
- [x] **Implement `1_train_autoencoder.py`:** Script to train the SIREN compressor.
- [x] **Implement `2_preprocess_mamba.py`:** Script to generate the $z$-vector sequences.
- [x] **Implement `3_train_processor.py`:** Script to train the Mamba processor with a full validation loop.

### Phase 2: Core Experimentation (In Progress)
- [ ] **Run `1_train_autoencoder.py`:**
    - [ ] Train the SIREN model.
    - [ ] **Crucial Check:** Visually inspect the reconstructed waveforms. Do they look like the originals? This proves the $z$-vector is a high-fidelity representation.
- [ ] **Run `2_preprocess_mamba.py`:**
    - [ ] Generate the complete `preprocessed_mamba_data/` directory for all patients.
- [ ] **Run `3_train_processor.py`:**
    - [ ] Train the main `Stream-INR` model.
    - [ ] **Tune Hyperparameters:** Adjust Mamba's `d_model`, `d_state`, and `learning_rate` to get the best possible validation F1-score.
    - [ ] **Save Best Model:** `processor.pth` is ready.

### Phase 3: A* Ablation Studies (To-Do)
This is the **most critical** phase for an A* paper. We must *prove* our architecture is better *and* understand *why*.

- [ ] **Baseline 1: "Dumb Compression"**
    - [ ] Modify `dataset.py` to create $z$-vectors by simple *averaging* (e.g., `torch.mean(chunk, dim=-1)`) instead of using the SIREN encoder.
    - [ ] Re-run `3_train_processor.py` on this "dumb" data.
    - [ ] **Hypothesis:** The F1-score will be very low (e.g., ~0.55), proving that *intelligent* compression (SIREN) is necessary to preserve the signal.
- [ ] **Baseline 2: "No Long-Term Memory"**
    - [ ] Implement a standard **CNN-LSTM** baseline model.
    - [ ] Train it on the *same* preprocessed $z$-vector sequences.
    - [ ] **Hypothesis:** The F1-score will be mediocre (e.g., ~0.65), proving that Mamba's superior long-range memory is necessary to find the pre-ictal patterns.
- [ ] **Baseline 3: "SOTA Transformer"**
    - [ ] Implement a **Patch-TST (Transformer)** baseline.
    - [ ] Train it on the *same* preprocessed $z$-vector sequences.
    - [ ] **Hypothesis:** This will be our main competitor. It may achieve a high F1-score (e.g., ~0.80), but we will beat it on the *next* step.
- [ ] **Baseline 4: "End-to-End Mamba" (Optional but good)**
    - [ ] Modify the Mamba model to take the raw, non-compressed `(B, 3600, 23*256)` data directly.
    - [ ] **Hypothesis:** The model will fail to train, be confused by the noise, or perform poorly, proving that our SIREN compressor is not a gimmick but a vital denoising step.

### Phase 4: Final Results & Paper (To-Do)
- [ ] **Implement `test.py`:**
    - [ ] Create a final, held-out test set (e.g., Patients `chb20-24`).
    - [ ] Run all models (Ours, CNN-LSTM, Patch-TST) *once* on this test set.
- [ ] **Generate the "Money Table" (The A* Result):**
    - [ ] Create the final table comparing all models on **F1-Score, Precision, Recall, AUC-ROC**.
    - [ ] **This table proves our accuracy.**
- [ ] **Generate the "Efficiency Table":**
    - [ ] Profile all models on a single GPU.
    - [ ] Compare: **`Parameters (M)`**, **`Training VRAM (GB)`**, and **`Inference Latency (ms/step)`**.
    - [ ] **This table proves our edge-device feasibility.**
- [ ] **Write the Paper:**
    - [ ] Draft the abstract and methodology.
    - [ ] Use the ablation studies to write the core "Results" section.
    - [ ] Use the efficiency table to prove the "Edge Device" claims.
    - [Example chart for paper: Mamba accuracy vs Transformer accuracy vs compute]
    - [ ] Submit to NeurIPS / ICML / ICLR.