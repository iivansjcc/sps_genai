import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional

LATENT_DIM = 100

def train_gan(
    model: Dict[str, torch.nn.Module],
    data_loader,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[Dict[str, optim.Optimizer]] = None,
    device: str = "cpu",
    epochs: int = 10,
    lr: float = 2e-4,
    beta1: float = 0.5,
):
    """
    Train a GAN with BCEWithLogitsLoss.
      model: {'G': Generator, 'D': Discriminator}
      data_loader: MNIST in [-1,1]
    """
    assert isinstance(model, dict) and "G" in model and "D" in model, "model must be dict with 'G' and 'D'"
    G, D = model["G"].to(device), model["D"].to(device)

    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    if optimizer is None:
        optG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
        optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
    else:
        optG, optD = optimizer["G"], optimizer["D"]

    for ep in range(1, epochs + 1):
        G.train(); D.train()
        g_sum = d_sum = n = 0
        for real, _ in data_loader:
            real = real.to(device)
            b = real.size(0)
            valid = torch.ones(b, device=device)
            fakev = torch.zeros(b, device=device)

            # ---- Train D ----
            D.zero_grad(set_to_none=True)
            d_real = D(real)
            loss_real = criterion(d_real, valid)

            with torch.no_grad():
                z = torch.randn(b, LATENT_DIM, device=device)  # (N,100)
                fake_imgs = G(z)
            d_fake = D(fake_imgs)
            loss_fake = criterion(d_fake, fakev)

            lossD = loss_real + loss_fake
            lossD.backward(); optD.step()

            # ---- Train G ----
            G.zero_grad(set_to_none=True)
            z2 = torch.randn(b, LATENT_DIM, device=device)     # (N,100)
            gen_imgs = G(z2)
            d_gen = D(gen_imgs)
            lossG = criterion(d_gen, valid)
            lossG.backward(); optG.step()

            d_sum += lossD.item(); g_sum += lossG.item(); n += 1

        print(f"Epoch {ep:03d} | D: {d_sum/n:.3f} | G: {g_sum/n:.3f}")

    return {"G": G, "D": D}