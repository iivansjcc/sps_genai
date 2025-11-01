import torch
import torch.nn as nn

# ---- Constants for MNIST ----
LATENT_DIM = 100    # noise dim
IMG_CH = 1          # MNIST grayscale
IMG_SIZE = 28

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d, nn.Linear)):
        if getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias.data)

class GANGenerator(nn.Module):
    """
    Generator:
      z: (N,100) -> FC -> 7*7*128 -> reshape (N,128,7,7)
      ConvT 128->64 (k4,s2,p1) + BN + ReLU -> (N,64,14,14)
      ConvT 64->1   (k4,s2,p1) + Tanh      -> (N,1,28,28)
    """
    def __init__(self, z_dim: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(z_dim, 7 * 7 * 128)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, IMG_CH, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # accept (N,100) or (N,100,1,1)
        if z.dim() == 4 and z.size(2) == 1 and z.size(3) == 1:
            z = z.view(z.size(0), -1)
        x = self.fc(z)                       # (N, 7*7*128)
        x = x.view(z.size(0), 128, 7, 7)     # (N,128,7,7)
        x = self.deconv(x)                   # (N,1,28,28)
        return x

class GANDiscriminator(nn.Module):
    """
    Discriminator:
      x: (N,1,28,28)
      Conv 1->64   (k4,s2,p1) + LeakyReLU(0.2)     -> (N,64,14,14)
      Conv 64->128 (k4,s2,p1) + BN + LeakyReLU(0.2)-> (N,128,7,7)
      Flatten -> Linear(128*7*7 -> 1)  => logit
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(IMG_CH, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128 * 7 * 7, 1)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)                 # (N,128,7,7)
        h = h.view(h.size(0), -1)            # (N, 128*7*7)
        logit = self.classifier(h).view(-1)  # (N,)
        return logit

def get_model(model_name: str):
    """
    Return {'G': Generator, 'D': Discriminator} for 'gan'.
    """
    name = (model_name or "").strip().lower()
    if name == "gan":
        return {"G": GANGenerator(), "D": GANDiscriminator()}
    raise ValueError(f"Unknown model_name: {model_name}")