from __future__ import annotations

import torch
from torch import nn
import torchaudio


class CNNArtifactDetector(nn.Module):
    def __init__(self, sample_rate: int = 32000, n_mels: int = 128) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=320,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80.0)

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3:
            raise ValueError(f"Expected waveform [B, 1, T], got {waveform.shape}")

        x = waveform.squeeze(1)
        x = self.mel(x)
        x = self.to_db(x)

        # Standardize per-sample for stable optimization on varied loudness.
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        x = (x - mean) / std

        x = x.unsqueeze(1)
        x = self.net(x)
        x = x.flatten(start_dim=1)
        logits = self.classifier(x).squeeze(1)
        return logits
