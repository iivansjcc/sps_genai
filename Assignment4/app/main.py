from pathlib import Path
from typing import Optional

import base64
import io

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

from helper_lib.energy_model import EnergyCNN, compute_energy_from_logits
from helper_lib.diffusion_model import SimpleUNet, get_diffusion_schedule, p_sample_loop
from helper_lib.data import CIFAR10_MEAN, CIFAR10_STD

app = FastAPI(title="Assignment4 CIFAR-10 Energy & Diffusion API")

# Paths
ROOT = Path(__file__).resolve().parents[1]  # Assignment4
OUTPUTS = ROOT / "outputs"
ENERGY_WEIGHTS = OUTPUTS / "energy_cifar10.pth"
DIFF_WEIGHTS = OUTPUTS / "diffusion_cifar10.pth"

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Preprocess for energy model (classifier)
energy_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Preprocess for diffusion model: images are in [-1,1]
diffusion_inverse_transform = transforms.Compose([
    transforms.Lambda(lambda x: (x + 1.0) / 2.0),  # [-1,1] -> [0,1]
])


def load_energy_model(device: str = "cpu") -> EnergyCNN:
    if not ENERGY_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Energy model weights not found at {ENERGY_WEIGHTS}. "
            "Train and save them before calling the API."
        )
    model = EnergyCNN(num_classes=10).to(device)
    state = torch.load(str(ENERGY_WEIGHTS), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_diffusion_model(device: str = "cpu", T: int = 100):
    if not DIFF_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Diffusion model weights not found at {DIFF_WEIGHTS}. "
            "Train and save them before calling the API."
        )
    model = SimpleUNet().to(device)
    state = torch.load(str(DIFF_WEIGHTS), map_location=device)
    model.load_state_dict(state)
    model.eval()
    sched = get_diffusion_schedule(T=T, device=device)
    return model, sched


def tensor_to_png_bytes(t: torch.Tensor) -> bytes:
    """
    t: (N,C,H,W) in [0,1] or [-1,1]
    we normalize to [0,1] and make a grid, then encode as PNG bytes.
    """
    if t.min() < 0.0:
        t = (t + 1.0) / 2.0
    grid = make_grid(t, nrow=int(t.size(0) ** 0.5), padding=2)
    grid = grid.clamp(0.0, 1.0)
    arr = (grid.mul(255).permute(1, 2, 0).byte().cpu().numpy())
    if arr.shape[2] == 1:
        img = Image.fromarray(arr.squeeze(-1), mode="L")
    else:
        img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class EnergyResponse(BaseModel):
    class_idx: int
    class_name: str
    energy: float
    probs: Optional[list[float]] = None


class DiffusionResponse(BaseModel):
    n: int
    image_base64_png: str


@app.get("/")
def health():
    return {
        "status": "ok",
        "components": ["energy_model", "diffusion_model"],
        "energy_weights": str(ENERGY_WEIGHTS),
        "diffusion_weights": str(DIFF_WEIGHTS),
    }


# -------- Energy model endpoint --------

@app.post("/energy/predict", response_model=EnergyResponse)
async def energy_predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    x = energy_transform(img).unsqueeze(0)  # (1,3,32,32)
    device = "cpu"
    try:
        model = load_energy_model(device=device)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    with torch.no_grad():
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1)
        class_idx = int(probs.argmax(1).item())
        energy = float(compute_energy_from_logits(logits)[0].item())

    class_name = CIFAR10_CLASSES[class_idx]
    return EnergyResponse(
        class_idx=class_idx,
        class_name=class_name,
        energy=energy,
        probs=probs[0].tolist(),
    )


# -------- Diffusion endpoints --------

@app.get("/diffusion/generate", response_model=DiffusionResponse)
def diffusion_generate(
    n: int = Query(16, ge=1, le=64),
    seed: Optional[int] = Query(None),
    T: int = Query(100, ge=10, le=1000),
):
    device = "cpu"
    if seed is not None:
        torch.manual_seed(seed)

    try:
        model, sched = load_diffusion_model(device=device, T=T)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    with torch.no_grad():
        samples = p_sample_loop(model, (n, 3, 32, 32), sched, device=device)
        # convert to [0,1] for visualization
        samples_vis = diffusion_inverse_transform(samples)
    png_bytes = tensor_to_png_bytes(samples_vis)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    return DiffusionResponse(n=n, image_base64_png=b64)


@app.get("/diffusion/generate.png")
def diffusion_generate_png(
    n: int = Query(16, ge=1, le=64),
    seed: Optional[int] = Query(None),
    T: int = Query(100, ge=10, le=1000),
):
    device = "cpu"
    if seed is not None:
        torch.manual_seed(seed)

    try:
        model, sched = load_diffusion_model(device=device, T=T)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    with torch.no_grad():
        samples = p_sample_loop(model, (n, 3, 32, 32), sched, device=device)
        samples_vis = diffusion_inverse_transform(samples)
    png_bytes = tensor_to_png_bytes(samples_vis)
    return Response(content=png_bytes, media_type="image/png")


@app.get("/diffusion/generate.html")
def diffusion_generate_html(
    n: int = Query(16, ge=1, le=64),
    seed: Optional[int] = Query(None),
    T: int = Query(100, ge=10, le=1000),
):
    device = "cpu"
    if seed is not None:
        torch.manual_seed(seed)

    try:
        model, sched = load_diffusion_model(device=device, T=T)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    with torch.no_grad():
        samples = p_sample_loop(model, (n, 3, 32, 32), sched, device=device)
        samples_vis = diffusion_inverse_transform(samples)
    png_bytes = tensor_to_png_bytes(samples_vis)
    b64 = base64.b64encode(png_bytes).decode("utf-8")
    html = f"""
    <!doctype html>
    <html>
      <head><meta charset="utf-8"><title>Diffusion Samples (n={n})</title></head>
      <body style="margin:0;background:#111;display:grid;place-items:center;height:100vh;">
        <img alt="diffusion samples" src="data:image/png;base64,{b64}"
             style="image-rendering: pixelated; max-width:95vw; height:auto;"/>
      </body>
    </html>
    """
    return HTMLResponse(content=html)