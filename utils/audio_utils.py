"""
Audio preprocessing utilities for Cantonese SER.
Converts MP3 bytes to Mel Spectrograms.
"""

import torch
import torchaudio
import numpy as np
from io import BytesIO
import soundfile as sf


class AudioPreprocessor:
    """
    Preprocess audio from MP3 bytes to Mel Spectrogram.
    
    Augmentations (noise injection, SpecAugment) are applied only in
    training mode. Use train() / eval() to toggle.
    
    Args:
        sample_rate: Target sample rate (default: 22050)
        n_mels: Number of mel filter banks (default: 80)
        n_fft: FFT window size (default: 1024)
        hop_length: Hop length for STFT (default: 256)
        win_length: Window length for STFT (default: 1024)
        f_min: Minimum frequency (default: 0)
        f_max: Maximum frequency (default: 11025)
        noise_factor: Std of Gaussian noise added to waveform in train mode (default: 0.005)
        freq_mask_max: Max mel bands to mask in SpecAugment (default: 15)
        time_mask_max: Max time frames to mask in SpecAugment (default: 30)
        num_spec_mask: Number of freq+time mask rounds in SpecAugment (default: 2)
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        f_min: float = 0.0,
        f_max: float = 11025.0,
        noise_factor: float = 0.005,
        freq_mask_max: int = 15,
        time_mask_max: int = 30,
        num_spec_mask: int = 2,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        
        # Augmentation parameters
        self.noise_factor = noise_factor
        self.freq_mask_max = freq_mask_max
        self.time_mask_max = time_mask_max
        self.num_spec_mask = num_spec_mask
        
        # Training mode flag (default: eval — no augmentations)
        self.training = False
        
        # Mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0  # Power spectrogram
        )
    
    def train(self):
        """Enable training augmentations (noise + SpecAugment)."""
        self.training = True
    
    def eval(self):
        """Disable augmentations for validation/inference."""
        self.training = False
    
    def __call__(self, audio_bytes: bytes) -> tuple:
        """
        Convert MP3 bytes to Mel Spectrogram.
        
        When self.training is True, applies noise injection on waveform
        and SpecAugment (frequency + time masking) on the dB spectrogram.
        
        Args:
            audio_bytes: Raw MP3 bytes from parquet file
            
        Returns:
            mel_spec: Tensor of shape (n_mels, time_steps)
            length: int, number of time frames
        """
        # 1. Decode MP3 bytes to waveform
        with BytesIO(audio_bytes) as f:
            waveform, sr = sf.read(f)
        
        # Convert to torch tensor
        waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # 2. Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.sample_rate
            )
        
        # 3. Ensure mono (shape: [time])
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=-1)
        
        # 4. Normalize waveform (z-score)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
        
        # 5. Noise injection (train only)
        if self.training and self.noise_factor > 0:
            waveform = waveform + self.noise_factor * torch.randn_like(waveform)
        
        # 6. Compute Mel Spectrogram (power)
        mel_spec = self.mel_transform(waveform.unsqueeze(0))  # (1, n_mels, time)
        
        # 7. Peak-referenced dB conversion
        # Equivalent to librosa.power_to_db(ref=np.max)
        mel_max = mel_spec.max() + 1e-10
        mel_spec = 10.0 * torch.log10(mel_spec / mel_max + 1e-10)
        
        # 8. SpecAugment: frequency + time masking (train only)
        if self.training and self.num_spec_mask > 0:
            mel_spec = self._apply_spec_augment(mel_spec)
        
        # 9. Normalize mel spectrogram (z-score)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        # Remove batch dimension: (1, n_mels, time) -> (n_mels, time)
        mel_spec = mel_spec.squeeze(0)
        
        # Get length (number of time frames)
        length = mel_spec.shape[-1]
        
        return mel_spec, length
    
    def _apply_spec_augment(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment: frequency masking and time masking.
        
        Operates on a single spectrogram of shape (1, n_mels, time).
        Args:
            mel_spec: Tensor of shape (1, n_mels, time)
        Returns:
            augmented: Tensor of same shape
        """
        _, n_mels, n_steps = mel_spec.shape
        fill_value = mel_spec.min()
        
        for _ in range(self.num_spec_mask):
            # Frequency mask
            f_size = torch.randint(1, self.freq_mask_max + 1, (1,)).item()
            f_start = torch.randint(0, max(1, n_mels - f_size), (1,)).item()
            mel_spec[:, f_start:f_start + f_size, :] = fill_value
            
            # Time mask
            t_size = torch.randint(1, self.time_mask_max + 1, (1,)).item()
            t_start = torch.randint(0, max(1, n_steps - t_size), (1,)).item()
            mel_spec[:, :, t_start:t_start + t_size] = fill_value
        
        return mel_spec


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    print("Testing AudioPreprocessor...")
    
    # Load sample parquet
    df = pd.read_parquet("data/train-00000-of-00045.parquet")
    
    # Get first audio
    audio_bytes = df['audio_file'].iloc[0]['bytes']
    
    # Preprocess
    preprocessor = AudioPreprocessor()
    mel_spec, length = preprocessor(audio_bytes)
    
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"Number of time frames: {length}")
    print(f"Value range: [{mel_spec.min():.2f}, {mel_spec.max():.2f}]")
    print(f"Has NaN: {torch.isnan(mel_spec).any()}")
    print(f"Has Inf: {torch.isinf(mel_spec).any()}")
    
    # Test with a few samples to check variable lengths
    print("\nTesting variable lengths...")
    for i in range(5):
        audio_bytes = df['audio_file'].iloc[i]['bytes']
        mel_spec, length = preprocessor(audio_bytes)
        duration = df['duration'].iloc[i] / 22050  # Convert to seconds
        print(f"Sample {i}: duration={duration:.2f}s, frames={length}")
