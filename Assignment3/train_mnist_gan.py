import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from pathlib import Path
from torchvision.utils import make_grid, save_image

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch.backends.mps as mps
        if mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"

def main():
    device = pick_device()
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # scale to [-1,1]
    ])
    ds = datasets.MNIST(root="data/mnist", train=True, download=True, transform=tfm)
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model("gan")
    model = train_gan(model, dl, device=device, epochs=15)

    # quick sanity check
    z = torch.randn(16, 100, device=device)
    with torch.no_grad():
        samples = model["G"](z).cpu()
    print("Generated:", samples.shape, samples.min().item(), samples.max().item())

    # save sample grid
    grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
    save_image(grid, out_dir / "gan_samples_last.png")
    print(f"Saved sample grid -> {out_dir / 'gan_samples_last.png'}")

    # save generator checkpoint
    torch.save(model["G"].state_dict(), out_dir / "generator.pth")
    print(f"Saved generator weights -> {out_dir / 'generator.pth'}")

if __name__ == "__main__":
    main()