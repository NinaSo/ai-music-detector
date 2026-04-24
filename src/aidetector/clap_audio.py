from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from transformers import ClapModel, ClapProcessor

from aidetector.audio_io import load_audio
from aidetector.device import get_best_device


@dataclass
class CLAPAudioConfig:
    model_name: str = "laion/clap-htsat-unfused"
    sample_rate: int = 48000
    batch_size: int = 8


class CLAPAudioEmbedder:
    def __init__(self, cfg: CLAPAudioConfig) -> None:
        self.cfg = cfg
        self.device = get_best_device()
        self.processor = ClapProcessor.from_pretrained(cfg.model_name)
        self.model = ClapModel.from_pretrained(cfg.model_name).to(self.device)
        self.model.eval()

    def load_audio_array(self, path: str) -> np.ndarray:
        wav, sr = load_audio(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.cfg.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.cfg.sample_rate)
        return wav.squeeze(0).numpy()

    @torch.inference_mode()
    def embed_audio_arrays(self, audio_arrays: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(
            audio=audio_arrays,
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        allowed = {"input_features", "is_longer", "attention_mask"}
        kwargs = {k: v for k, v in inputs.items() if k in allowed}
        emb_out = self.model.get_audio_features(**kwargs)
        if isinstance(emb_out, torch.Tensor):
            emb = emb_out
        elif hasattr(emb_out, "pooler_output") and emb_out.pooler_output is not None:
            emb = emb_out.pooler_output
        elif hasattr(emb_out, "last_hidden_state") and emb_out.last_hidden_state is not None:
            emb = emb_out.last_hidden_state.mean(dim=1)
        else:
            raise TypeError(f"Unsupported CLAP audio output type: {type(emb_out)}")

        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb.detach().cpu().numpy().astype(np.float32)
