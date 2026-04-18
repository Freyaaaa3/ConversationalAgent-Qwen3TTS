import io
import os
from typing import Optional

import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel

import torch

# Ensure local package import works regardless of CWD.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
import sys

sys.path.insert(0, os.path.join(REPO_ROOT, "backend", "pkg"))

from qwen_tts import Qwen3TTSModel


MODEL_PATH = os.environ.get("QWEN_TTS_MODEL_PATH", os.path.join(REPO_ROOT, "backend", "model", "Qwen3_TTS_12Hz_0.6B_Base"))
DEVICE = os.environ.get("QWEN_TTS_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")


class TTSRequest(BaseModel):
  text: str
  language: Optional[str] = "Chinese"


app = FastAPI(title="Qwen3-TTS HTTP Server")


@torch.inference_mode()
def _load_model() -> Qwen3TTSModel:
  global _model
  if "_model" not in globals():
    _model = Qwen3TTSModel.from_pretrained(
      MODEL_PATH,
      device_map=DEVICE,
      dtype=torch.bfloat16 if str(DEVICE).startswith("cuda") else torch.float32,
      attn_implementation="flash_attention_2" if str(DEVICE).startswith("cuda") else "sdpa",
    )
  return _model


@app.post("/tts", response_class=None)
@torch.inference_mode()
def tts(req: TTSRequest):
  model = _load_model()

  wavs, sr = model.generate_voice_clone(
    text=req.text,
    language=req.language or "Chinese",
    ref_audio=None,
    ref_text=None,
    x_vector_only_mode=True,
  )

  wav = wavs[0]
  buffer = io.BytesIO()
  sf.write(buffer, wav, sr, format="WAV")
  buffer.seek(0)

  from fastapi.responses import Response

  return Response(content=buffer.read(), media_type="audio/wav")


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("QWEN_TTS_PORT", "8001")))

