

from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from helper_lib.data import get_cifar10_diffusion_loader
from helper_lib.diffusion_model import SimpleUNet, get_diffusion_schedule
from helper_lib.trainer import train_diffusion


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
    print(f"Using device: {device}")

    train_loader = get_cifar10_diffusion_loader(batch_size=128, num_workers=0)

    T = 100  # number of diffusion steps
    sched = get_diffusion_schedule(T=T, device=device)

    model = SimpleUNet()
    here = Path(__file__).resolve()
    assignment_dir = here.parents[0]   # Assignment4
    ckpt_path = assignment_dir / "outputs" / "diffusion_cifar10.pth"

    model = train_diffusion(
        model=model,
        train_loader=train_loader,
        schedule=sched,
        device=device,
        epochs=5,       # keep small so it finishes; increase if you want better samples
        lr=2e-4,
        ckpt_path=ckpt_path,
    )

    # Generate and save a quick sample grid for inspection
    with torch.no_grad():
        from helper_lib.diffusion_model import p_sample_loop
        samples = p_sample_loop(model, (16, 3, 32, 32), sched, device=device)
        # convert from [-1,1] to [0,1]
        samples = (samples + 1.0) / 2.0
        samples = samples.clamp(0.0, 1.0)
        grid = make_grid(samples, nrow=4)
    out_dir = assignment_dir / "outputs" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "diffusion_samples_last.png"
    save_image(grid, save_path)
    print(f"Saved diffusion sample grid to {save_path}")


if __name__ == "__main__":
    main()