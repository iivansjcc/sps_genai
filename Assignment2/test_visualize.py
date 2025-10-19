import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from helper_lib import get_model

# ----- config -----
CKPT_PATH = "checkpoints/model.pth"
IMG_SIZE = 64
BATCH_SIZE = 12 
 # Auto-select device: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU
def _pick_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

DEVICE = _pick_device()
print(f"Using device: {DEVICE}")
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

# Build CIFAR-10 test loader 
tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
test_ds = datasets.CIFAR10(root="data/cifar10", train=False, download=True, transform=tfm)
class_names = test_ds.classes
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load model (num_classes matches CIFAR-10 = 10)
model = get_model("enhancedcnn", num_classes=10)
state = torch.load(CKPT_PATH, map_location="cpu")
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# Get one batch
images, labels = next(iter(test_loader))
images, labels = images.to(DEVICE), labels.to(DEVICE)

with torch.no_grad():
    logits = model(images)
    preds = logits.argmax(1)

# Helper: unnormalize & show
def imshow(img_tensor):
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img * np.array(STD)) + np.array(MEAN)
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis("off")

# Plot grid
n = images.size(0)
cols = 6
rows = (n + cols - 1) // cols
plt.figure(figsize=(3*cols, 3*rows))
for i in range(n):
    plt.subplot(rows, cols, i+1)
    imshow(images[i])
    p, t = int(preds[i]), int(labels[i])
    title = f"Pred: {class_names[p]}\nTrue: {class_names[t]}"
    plt.title(title, fontsize=9)
plt.tight_layout()
plt.show()