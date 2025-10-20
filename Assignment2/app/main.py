from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import io
import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from helper_lib import get_model  # uses your model.py

app = FastAPI(title="Assignment2 CNN Inference")

# ----- Model / preprocessing config -----
MODEL_PATH = "checkpoints/model.pth"
NUM_CLASSES = 10           # CIFAR-10
IMG_SIZE = 64
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# CIFAR-10 class labels
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Build model and load weights
model = get_model("enhancedcnn", num_classes=NUM_CLASSES)
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class Prediction(BaseModel):
    top1_class_idx: int
    top1_class_name: str
    logits: List[float]

@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "enhancedcnn",
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "version": "with-class-name-and-robust-image-decode"
    }

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    # Read & decode image robustly
    img_bytes = await file.read()
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.load()  # force decode to catch errors early
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Unsupported or corrupt image. Please upload a JPG or PNG.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    x = preprocess(img).unsqueeze(0)  # (1,3,H,W)

    with torch.no_grad():
        logits_tensor = model(x)[0]
        logits = logits_tensor.tolist()
        top1 = int(logits_tensor.argmax().item())

    pred_name = CLASS_NAMES[top1] if 0 <= top1 < len(CLASS_NAMES) else f"class_{top1}"
    return {"top1_class_idx": top1, "top1_class_name": pred_name, "logits": logits}