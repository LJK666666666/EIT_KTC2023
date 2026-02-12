"""
Test-time optimization with differentiable physics backend.
"""
from pathlib import Path
from typing import Dict, Tuple
import json
import torch
import matplotlib.pyplot as plt


def tv_loss(sigma: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation regularization."""
    grad_x = sigma[:, :, :, 1:] - sigma[:, :, :, :-1]
    grad_y = sigma[:, :, 1:, :] - sigma[:, :, :-1, :]
    return (grad_x.abs().mean() + grad_y.abs().mean())


def optimize_sigma_with_backend(
    sigma_init: torch.Tensor,
    measurements: torch.Tensor,
    backend,
    steps: int = 20,
    lr: float = 1e-2,
    lambda_smooth: float = 1e-4
) -> Tuple[torch.Tensor, Dict[str, list]]:
    """
    Optimize sigma at test time:
    L = ||y_pred - y_obs||^2 + lambda * TV(sigma)
    """
    if sigma_init.shape[0] != 1:
        raise ValueError(f"Only single-sample optimization is supported, got {sigma_init.shape}")

    measurements = measurements.to(sigma_init.device)
    backend.prepare(measurements, sigma_init)

    sigma = sigma_init.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([sigma], lr=lr)

    history = {
        "total_loss": [],
        "data_loss": [],
        "tv_loss": []
    }

    for _ in range(steps):
        optimizer.zero_grad()
        pred_measurements = backend.predict(sigma)
        data_loss = torch.mean((pred_measurements - measurements) ** 2)
        current_tv_loss = tv_loss(sigma)
        total_loss = data_loss + lambda_smooth * current_tv_loss
        total_loss.backward()
        optimizer.step()

        history["total_loss"].append(float(total_loss.detach().cpu().item()))
        history["data_loss"].append(float(data_loss.detach().cpu().item()))
        history["tv_loss"].append(float(current_tv_loss.detach().cpu().item()))

    return sigma.detach(), history


def save_loss_curve(history: Dict[str, list], save_json: Path, save_png: Path) -> None:
    """Save optimization loss history as json and plot."""
    save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w") as f:
        json.dump(history, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(history["total_loss"], label="total_loss")
    ax.plot(history["data_loss"], label="data_loss")
    ax.plot(history["tv_loss"], label="tv_loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(save_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
