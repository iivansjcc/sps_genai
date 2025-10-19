import torch
from pathlib import Path
from datetime import datetime

def save_model(model: torch.nn.Module, path: str, with_timestamp: bool = False) -> str:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if with_timestamp:
        stem = path_obj.stem
        suffix = path_obj.suffix or ".pth"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        path_obj = path_obj.with_name(f"{stem}_{ts}{suffix}")

    torch.save(model.state_dict(), path_obj)
    return str(path_obj)

def load_model(model: torch.nn.Module, path: str, map_location: str = "cpu") -> torch.nn.Module:
    state_dict = torch.load(path, map_location=map_location)
    model.load_state_dict(state_dict)
    return model

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: torch.nn.Module):
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")