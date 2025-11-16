import torch
import torch.nn as nn
from mamba_ssm import Mamba

# --- Part 1.1: The SIREN (Sine) Layer ---
# This is the "secret sauce" of the SIREN autoencoder.
# It uses sin() activation and a special initialization.

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # This initialization is CRITICAL for SIREN to work.
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer uniform distribution
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)
            else:
                # Subsequent layers uniform distribution
                limit = torch.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-limit, limit)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

# --- Part 1.2: The SIREN-based Autoencoder ---

class Encoder(nn.Module):
    """
    Compresses a 1-second EEG chunk.
    Input: (B, C, L) -> (B, 24, 256)
    Output: (B, z_dim) -> (B, 64)
    """
    def __init__(self, in_channels=24, seq_len=256, latent_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(2), # seq_len = 128
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(2), # seq_len = 64
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool1d(4), # seq_len = 16
            nn.Flatten(),
            nn.Linear(128 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    """
    Reconstructs the 1-second EEG chunk from the latent vector z.
    Input: (B, z_dim) -> (B, 64)
    Output: (B, C, L) -> (B, 24, 256)
    """
    def __init__(self, latent_dim=64, out_channels=24, seq_len=256, hidden_dim=256):
        super().__init__()
        self.seq_len = seq_len
        self.out_channels = out_channels

        # This SIREN MLP maps the latent vector z to the full signal
        self.net = nn.Sequential(
            SineLayer(latent_dim, hidden_dim, is_first=True, omega_0=30.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0),
            SineLayer(hidden_dim, hidden_dim, omega_0=30.0),
            nn.Linear(hidden_dim, out_channels * seq_len) # Final layer is linear
        )
        
    def forward(self, z):
        # (B, 64) -> (B, 24 * 256)
        signal_flat = self.net(z)
        # (B, 24 * 256) -> (B, 24, 256)
        return signal_flat.view(-1, self.out_channels, self.seq_len)

class Autoencoder(nn.Module):
    def __init__(self, in_channels=24, seq_len=256, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(in_channels, seq_len, latent_dim)
        self.decoder = Decoder(latent_dim, in_channels, seq_len)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# --- Part 2: The Mamba Processor ---
# This model takes the *sequence* of z-vectors.

class StreamProcessor(nn.Module):
    """
    Takes a sequence of latent vectors and predicts pre-ictal state.
    Input: (B, L, z_dim) -> (B, 3600, 64)
    Output: (B, 1) -> (B, 1) (Logit for pre-ictal vs. inter-ictal)
    """
    def __init__(self, seq_len=3600, latent_dim=64, mamba_dim=128, num_classes=1):
        super().__init__()
        self.seq_len = seq_len
        
        # 1. Project latent vector z (64) to Mamba's dimension (128)
        self.embedding = nn.Linear(latent_dim, mamba_dim)
        
        # 2. The Mamba Block
        self.mamba = Mamba(
            d_model=mamba_dim,
            d_state=16, # This is a key hyperparameter
            d_conv=4,
            expand=2,
        )
        
        # 3. Classification Head
        self.norm = nn.LayerNorm(mamba_dim)
        self.classifier = nn.Linear(mamba_dim, num_classes)
        
    def forward(self, z_sequence):
        # z_sequence shape: (B, L, z_dim) -> (B, 3600, 64)
        x = self.embedding(z_sequence) # (B, L, mamba_dim)
        x = self.mamba(x) # (B, L, mamba_dim)
        
        # We only care about the prediction at the *last* time step
        x_last = x[:, -1, :] # (B, mamba_dim)
        
        x_last = self.norm(x_last)
        logit = self.classifier(x_last)
        
        # (B, 1) -> squeeze to (B,) if using BCEWithLogitsLoss
        return logit.squeeze(-1)