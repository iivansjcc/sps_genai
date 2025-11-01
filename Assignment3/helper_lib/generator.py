import io, base64
from typing import Optional
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
import matplotlib.pyplot as plt

def generate_samples(
    model: dict,
    device: str = "cpu",
    num_samples: int = 16,
    seed: Optional[int] = None,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Generate 'num_samples' images from model['G'] and either show a grid or save it.
    - model: {'G': Generator, 'D': Discriminator} (only G is used)
    - returns the grid image as a PIL.Image if show=False and save_path is None
    """
    assert isinstance(model, dict) and "G" in model, "model must be dict with 'G'"
    G = model["G"].to(device).eval()
    if seed is not None:
        torch.manual_seed(seed)

    z = torch.randn(num_samples, 100, 1, 1, device=device)
    with torch.no_grad():
        samples = G(z).cpu()  # expected in [-1,1] due to Tanh

    grid = make_grid(samples, nrow=int(num_samples**0.5), normalize=True, value_range=(-1, 1))

    if save_path:
        save_image(grid, save_path)

    if show:
        # convert to HWC for imshow
        arr = grid.permute(1,2,0).numpy()
        if arr.shape[2] == 1:
            arr = arr.squeeze(-1)
            plt.imshow(arr, cmap="gray")
        else:
            plt.imshow(arr)
        plt.axis("off")
        plt.show()
        return None

    # return a PIL Image if not showing or saving
    arr = (grid.mul(255).clamp(0,255).permute(1,2,0).byte().numpy())
    mode = "L" if arr.shape[2] == 1 else "RGB"
    return Image.fromarray(arr.squeeze() if mode=="L" else arr, mode=mode)