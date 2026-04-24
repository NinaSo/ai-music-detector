from __future__ import annotations

import torchaudio


TORCHCODEC_HELP = (
    "Audio decode failed because torchaudio requires `torchcodec` in this environment. "
    "Fix by running: `pip install torchcodec`. "
    "If install fails, pin to older stack: `pip install \"torch==2.5.1\" \"torchaudio==2.5.1\"`."
)


def load_audio(path: str):
    try:
        return torchaudio.load(path)
    except ImportError as e:
        msg = str(e)
        if "TorchCodec" in msg or "torchcodec" in msg.lower():
            raise ImportError(TORCHCODEC_HELP) from e
        raise
