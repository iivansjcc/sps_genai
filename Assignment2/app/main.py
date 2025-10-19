from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import torch
from torchvision import transforms
from PIL import Image
import io

from helper_lib import get_model  # uses your model.py

app = FastAPI(title="Assignment2 CNN Inference")

# ----- Model load -----
MODEL_PATH = "checkpoints/model.pth"
NUM_CLASSES = 10
IMG_SIZE = 64

model = get_model("enhancedcnn", num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ----- Preprocess -----
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

@app.get("/")
def health():
    return {"status": "ok", "model": "enhancedcnn", "img_size": IMG_SIZE, "num_classes": NUM_CLASSES}

class Prediction(BaseModel):
    top1_class_idx: int
    logits: List[float]

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes))
    x = preprocess(img).unsqueeze(0)  # (1,3,64,64)
    with torch.no_grad():
        logits = model(x)[0].tolist()
        top1 = int(torch.tensor(logits).argmax().item())
    return {"top1_class_idx": top1, "logits": logits}