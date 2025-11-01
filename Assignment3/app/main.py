from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import io, base64
from fastapi.responses import Response, HTMLResponse
from typing import Optional
from fastapi import Query

import torch
from torchvision.utils import make_grid
from PIL import Image

# Import your GAN generator + latent size
from helper_lib.model import GANGenerator, LATENT_DIM

app = FastAPI(title="Assignment3 GAN API", version="1.0.0")

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "outputs" / "generator.pth"


def _load_generator(device: str = "cpu") -> GANGenerator:
    G = GANGenerator().to(device)
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {WEIGHTS_PATH}. "
            "Train first and save weights to Assignment3/outputs/generator.pth"
        )
    state = torch.load(str(WEIGHTS_PATH), map_location=device)
    G.load_state_dict(state)
    G.eval()
    return G


def _tensor_grid_to_base64_png(t):
    """
    t: (N,1,28,28) in [-1,1] -> base64 PNG grid
    """
    grid = make_grid(t, nrow=int(t.size(0) ** 0.5), normalize=True, value_range=(-1, 1))
    # CHW -> HWC uint8
    arr = (grid.mul(255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy())
    if arr.shape[2] == 1:
        img = Image.fromarray(arr.squeeze(-1), mode="L")
    else:
        img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class GenerateResponse(BaseModel):
    n: int
    image_base64_png: str


@app.get("/")
def health():
    return {
        "status": "ok",
        "component": "gan",
        "weights": str(WEIGHTS_PATH),
        "hint": "Call /gan/generate?n=16 to get a base64 PNG grid."
    }


@app.get("/gan/generate", response_model=GenerateResponse)
def gan_generate(
    n: int = Query(16, ge=1, le=64, description="Number of samples"),
    seed: Optional[int] = Query(None, description="Optional RNG seed"),
):
    device = "cpu"  # API is CPU-only
    if seed is not None:
        torch.manual_seed(seed)

    try:
        G = _load_generator(device=device)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    z = torch.randn(n, LATENT_DIM, device=device)  # (N,100)
    with torch.no_grad():
        samples = G(z).cpu()  # (N,1,28,28) in [-1,1]
    b64 = _tensor_grid_to_base64_png(samples)
    return {"n": n, "image_base64_png": b64}


# --- helper to generate PNG bytes directly ---
def _generate_png_bytes(n: int, seed: Optional[int] = None) -> bytes:
    device = "cpu"
    if seed is not None:
        torch.manual_seed(seed)
    G = _load_generator(device=device)
    z = torch.randn(n, LATENT_DIM, device=device)
    with torch.no_grad():
        samples = G(z).cpu()
    # make a grid and encode to PNG bytes (same logic as _tensor_grid_to_base64_png)
    grid = make_grid(samples, nrow=int(n ** 0.5), normalize=True, value_range=(-1, 1))
    arr = (grid.mul(255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy())
    img = Image.fromarray(arr.squeeze(-1), mode="L") if arr.shape[2] == 1 else Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- 1) Raw PNG endpoint ---
@app.get("/gan/generate.png")
def gan_generate_png(
    n: int = Query(16, ge=1, le=64, description="Number of samples"),
    seed: Optional[int] = Query(None, description="Optional RNG seed"),
):
    try:
        png = _generate_png_bytes(n, seed)
    except FileNotFoundError as e:
        # align with your existing behavior
        raise HTTPException(status_code=503, detail=str(e))
    return Response(content=png, media_type="image/png")

# --- 2) Simple HTML page that shows the image ---
@app.get("/gan/generate.html")
def gan_generate_html(
    n: int = Query(16, ge=1, le=64, description="Number of samples"),
    seed: Optional[int] = Query(None, description="Optional RNG seed"),
):
    try:
        png = _generate_png_bytes(n, seed)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    b64 = base64.b64encode(png).decode("utf-8")
    html = f"""
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>GAN Samples (n={n})</title></head>
      <body style="margin:0;background:#111;display:grid;place-items:center;height:100vh;">
        <img alt="GAN samples" src="data:image/png;base64,{b64}" style="image-rendering: pixelated; max-width:95vw; height:auto;"/>
      </body>
    </html>
    """
    return HTMLResponse(content=html)