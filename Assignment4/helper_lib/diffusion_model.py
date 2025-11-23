

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Time embedding ----------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embedding (as in many diffusion implementations).
    t: (N,) integer timesteps
    """
    half = dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half - 1)
    emb = torch.exp(torch.arange(half, device=t.device) * -emb)
    emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb


class SimpleUNet(nn.Module):
    """
    Very small UNet-like epsilon model for CIFAR-10 (3x32x32).
    Predicts noise given noisy image and timestep.
    """

    def __init__(self, time_dim: int = 128):
        super().__init__()
        self.time_dim = time_dim

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
        )

        # Down
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)  # 16x16
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1, stride=2)  # 8x8

        # Up
        self.tconv1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)  # 16x16
        self.tconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)   # 32x32
        self.conv_out = nn.Conv2d(64, 3, 3, padding=1)

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (N,3,32,32), t: (N,)
        N = x.size(0)
        temb = timestep_embedding(t, self.time_dim)  # (N, time_dim)
        temb = self.time_mlp(temb)                   # (N, time_dim)
        temb = temb[:, :, None, None]                # (N, time_dim,1,1)

        # Down
        h1 = self.act(self.conv1(x))                 # (N,64,32,32)
        h2 = self.act(self.conv2(h1))                # (N,128,16,16)
        h3 = self.act(self.conv3(h2))                # (N,128,8,8)

        # Add time embedding at bottleneck
        h3 = h3 + temb[:, :128, :, :]                # broadcast

        # Up
        u1 = self.act(self.tconv1(h3))               # (N,128,16,16)
        u1 = u1 + h2
        u2 = self.act(self.tconv2(u1))               # (N,64,32,32)
        u2 = u2 + h1
        out = self.conv_out(u2)                      # (N,3,32,32)
        return out


# ---------- Diffusion schedule utilities ----------

def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


def get_diffusion_schedule(T: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Precompute betas, alphas, and derived terms for DDPM-style training/sampling.
    Returns a dict with all necessary tensors on the given device.
    """
    betas = make_beta_schedule(T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return dict(
        T=T,
        betas=betas,
        alphas=alphas,
        alphas_cumprod=alphas_cumprod,
        alphas_cumprod_prev=alphas_cumprod_prev,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
        sqrt_recip_alphas=sqrt_recip_alphas,
        posterior_variance=posterior_variance,
    )


def q_sample(x0: torch.Tensor,
             t: torch.Tensor,
             noise: torch.Tensor,
             sqrt_alphas_cumprod: torch.Tensor,
             sqrt_one_minus_alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    Forward diffusion: q(x_t | x_0) = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise
    x0, noise: (N,3,32,32)
    t: (N,) int64
    """
    device = x0.device
    # gather corresponding sqrt(alpha_bar_t)
    sqrt_a = sqrt_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
    sqrt_oma = sqrt_one_minus_alphas_cumprod[t].to(device).view(-1, 1, 1, 1)
    return sqrt_a * x0 + sqrt_oma * noise


def p_sample(model: nn.Module,
             x_t: torch.Tensor,
             t: torch.Tensor,
             schedule: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    One reverse step p(x_{t-1} | x_t).
    """
    betas = schedule["betas"]
    sqrt_one_minus_alphas_cumprod = schedule["sqrt_one_minus_alphas_cumprod"]
    sqrt_recip_alphas = schedule["sqrt_recip_alphas"]
    posterior_variance = schedule["posterior_variance"]

    # predict noise
    eps_theta = model(x_t, t)
    # DDPM formula (simplified)
    betas_t = betas[t].view(-1, 1, 1, 1)
    sqrt_one_minus_ac_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
    sqrt_recip_alphas_t = sqrt_recip_alphas[t].view(-1, 1, 1, 1)

    model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_ac_t * eps_theta)

    # for t > 0, sample from N(model_mean, posterior_variance)
    noise = torch.randn_like(x_t)
    posterior_var_t = posterior_variance[t].view(-1, 1, 1, 1)
    nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)  # no noise at t=0
    x_prev = model_mean + nonzero_mask * torch.sqrt(posterior_var_t) * noise
    return x_prev


def p_sample_loop(model: nn.Module,
                  shape,
                  schedule: Dict[str, torch.Tensor],
                  device: str = "cpu") -> torch.Tensor:
    """
    Generate samples starting from pure noise at time T-1 down to 0.
    shape: (N,3,32,32)
    Returns x_0 in [-1,1].
    """
    T = schedule["T"]
    x = torch.randn(shape, device=device)
    for t_step in reversed(range(T)):
        t = torch.full((shape[0],), t_step, device=device, dtype=torch.long)
        x = p_sample(model, x, t, schedule)
    return x