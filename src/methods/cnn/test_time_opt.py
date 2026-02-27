"""
Test-time optimization with differentiable physics backend.
"""
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from .contour_param import build_initial_labels, build_distance_maps, synthesize_sigma
from .contour_postprocess import ternary_from_sigma, cleanup_ternary


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
    lambda_smooth: float = 1e-4,
    lambda_anchor: float = 5e-4,
    relinearize_every: int = 20,
    max_delta: float = 0.25,
    lr_min_factor: float = 0.1
) -> Tuple[torch.Tensor, Dict[str, list]]:
    """
    Optimize sigma at test time:
    L = ||y_pred - y_obs||^2 + lambda * TV(sigma)
    """
    if sigma_init.shape[0] != 1:
        raise ValueError(f"Only single-sample optimization is supported, got {sigma_init.shape}")

    measurements = measurements.to(sigma_init.device)
    sigma_start = sigma_init.detach().clone()
    backend.prepare(measurements, sigma_start)
    sigma = sigma_start.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([sigma], lr=lr)
    anchor = sigma_start.detach().clone()
    min_lr = float(lr) * float(lr_min_factor)
    if min_lr > lr:
        min_lr = float(lr)

    history = {
        "total_loss": [],
        "data_loss": [],
        "tv_loss": [],
        "anchor_loss": [],
        "lr": []
    }

    for step in range(steps):
        # cosine annealing from lr -> lr * lr_min_factor
        if steps > 1:
            phase = float(step) / float(steps - 1)
        else:
            phase = 1.0
        current_lr = min_lr + 0.5 * (float(lr) - min_lr) * (1.0 + torch.cos(torch.tensor(phase * 3.141592653589793)).item())
        optimizer.param_groups[0]["lr"] = current_lr

        if relinearize_every > 0 and step > 0 and (step % relinearize_every == 0):
            with torch.no_grad():
                sigma_detached = sigma.detach().clone()
            backend.prepare(measurements, sigma_detached)
            anchor = sigma_detached.detach().clone()

        optimizer.zero_grad()
        pred_measurements = backend.predict(sigma)
        data_loss = torch.mean((pred_measurements - measurements) ** 2)
        current_tv_loss = tv_loss(sigma)
        current_anchor_loss = torch.mean((sigma - anchor) ** 2)
        total_loss = data_loss + lambda_smooth * current_tv_loss + lambda_anchor * current_anchor_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            if max_delta > 0:
                lower = sigma_start - max_delta
                upper = sigma_start + max_delta
                sigma.clamp_(min=lower, max=upper)

        history["total_loss"].append(float(total_loss.detach().cpu().item()))
        history["data_loss"].append(float(data_loss.detach().cpu().item()))
        history["tv_loss"].append(float(current_tv_loss.detach().cpu().item()))
        history["anchor_loss"].append(float(current_anchor_loss.detach().cpu().item()))
        history["lr"].append(float(current_lr))

    return sigma.detach(), history


def optimize_sigma_contour_step(
    sigma_init: torch.Tensor,
    measurements: torch.Tensor,
    backend,
    steps: int = 280,
    lr: float = 1e-2,
    seg_threshold: float = 0.25,
    bspline_grid: int = 12,
    tau_start: float = 0.8,
    tau_end: float = 0.35,
    max_shift_px: float = 8.0,
    lambda_length: float = 2e-3,
    lambda_area: float = 5e-3,
    lambda_anchor_shape: float = 1e-3,
    lambda_speckle: float = 3e-3,
    lambda_anchor_sigma: float = 1e-3,
    relinearize_every: int = 25,
    min_component_ratio: float = 0.0015,
    lr_min_factor: float = 0.1,
    value_bounds: Tuple[float, float, float, float, float, float] = (-1.3, -0.2, -0.2, 0.2, 0.2, 1.3),
    stage2_steps: int = 60
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, list]]:
    """
    Contour-first optimization:
    stage A: optimize boundary displacement + class values.
    stage B: cleanup ternary then short pixel-space refinement.
    """
    if sigma_init.shape[0] != 1:
        raise ValueError(f"Only single-sample optimization is supported, got {sigma_init.shape}")
    measurements = measurements.to(sigma_init.device)
    device = sigma_init.device
    sigma_start = sigma_init.detach().clone()
    h = int(sigma_start.shape[-2])
    w = int(sigma_start.shape[-1])
    sigma_np = sigma_start[0, 0].detach().cpu().numpy()
    labels = build_initial_labels(sigma_np, threshold=seg_threshold)
    d_neg_np, d_pos_np, fg_dist_np = build_distance_maps(labels)
    d_neg = torch.from_numpy(d_neg_np).to(device=device, dtype=sigma_start.dtype).unsqueeze(0).unsqueeze(0)
    d_pos = torch.from_numpy(d_pos_np).to(device=device, dtype=sigma_start.dtype).unsqueeze(0).unsqueeze(0)
    far_bg = (fg_dist_np > 8.0).astype(np.float32)
    far_bg_t = torch.from_numpy(far_bg).to(device=device, dtype=sigma_start.dtype).unsqueeze(0).unsqueeze(0)
    init_neg_area = float((labels == -1).mean())
    init_pos_area = float((labels == 1).mean())
    theta_neg = torch.zeros((1, 1, int(bspline_grid), int(bspline_grid)), device=device, dtype=sigma_start.dtype, requires_grad=True)
    theta_pos = torch.zeros((1, 1, int(bspline_grid), int(bspline_grid)), device=device, dtype=sigma_start.dtype, requires_grad=True)
    values = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=sigma_start.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([theta_neg, theta_pos, values], lr=lr)
    min_lr = float(lr) * float(lr_min_factor)
    if min_lr > lr:
        min_lr = float(lr)
    history = {
        "total_loss": [],
        "data_loss": [],
        "tv_loss": [],
        "area_loss": [],
        "shape_anchor_loss": [],
        "speckle_loss": [],
        "sigma_anchor_loss": [],
        "lr": [],
        "tau": [],
        "stage": []
    }
    # stage A
    for step in range(max(1, int(steps))):
        if steps > 1:
            phase = float(step) / float(steps - 1)
        else:
            phase = 1.0
        current_lr = min_lr + 0.5 * (float(lr) - min_lr) * (1.0 + torch.cos(torch.tensor(phase * np.pi)).item())
        optimizer.param_groups[0]["lr"] = current_lr
        tau = float(tau_start) + (float(tau_end) - float(tau_start)) * phase
        sigma_pred, m_neg, _, m_pos = synthesize_sigma(
            theta_neg=theta_neg,
            theta_pos=theta_pos,
            d_neg=d_neg,
            d_pos=d_pos,
            values=values,
            tau=tau,
            max_shift_px=max_shift_px
        )
        sigma_pred = sigma_pred.unsqueeze(0) if sigma_pred.dim() == 3 else sigma_pred
        if relinearize_every > 0 and (step == 0 or step % int(relinearize_every) == 0):
            backend.prepare(measurements, sigma_pred.detach())
        optimizer.zero_grad()
        pred_meas = backend.predict(sigma_pred)
        data_loss = torch.mean((pred_meas - measurements) ** 2)
        tv_field = tv_loss(m_neg) + tv_loss(m_pos)
        area_loss = (m_neg.mean() - init_neg_area) ** 2 + (m_pos.mean() - init_pos_area) ** 2
        shape_anchor_loss = torch.mean(theta_neg ** 2) + torch.mean(theta_pos ** 2)
        speckle_loss = torch.mean((m_neg + m_pos) * far_bg_t)
        sigma_anchor_loss = torch.tensor(0.0, device=device, dtype=sigma_start.dtype)
        total_loss = (
            data_loss
            + lambda_length * tv_field
            + lambda_area * area_loss
            + lambda_anchor_shape * shape_anchor_loss
            + lambda_speckle * speckle_loss
        )
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            values[0].clamp_(min=float(value_bounds[0]), max=float(value_bounds[1]))
            values[1].clamp_(min=float(value_bounds[2]), max=float(value_bounds[3]))
            values[2].clamp_(min=float(value_bounds[4]), max=float(value_bounds[5]))
        history["total_loss"].append(float(total_loss.detach().cpu().item()))
        history["data_loss"].append(float(data_loss.detach().cpu().item()))
        history["tv_loss"].append(float(tv_field.detach().cpu().item()))
        history["area_loss"].append(float(area_loss.detach().cpu().item()))
        history["shape_anchor_loss"].append(float(shape_anchor_loss.detach().cpu().item()))
        history["speckle_loss"].append(float(speckle_loss.detach().cpu().item()))
        history["sigma_anchor_loss"].append(float(sigma_anchor_loss.detach().cpu().item()))
        history["lr"].append(float(current_lr))
        history["tau"].append(float(tau))
        history["stage"].append("A")
    with torch.no_grad():
        sigma_a = sigma_pred.detach().clone()
    ternary_a_np = ternary_from_sigma(sigma_a[0, 0].detach().cpu().numpy(), threshold=seg_threshold)
    ternary_b_np = cleanup_ternary(ternary_a_np, min_component_ratio=min_component_ratio)
    sigma_clean = torch.from_numpy(ternary_b_np).to(device=device, dtype=sigma_start.dtype).unsqueeze(0).unsqueeze(0)
    # stage B
    sigma = sigma_clean.detach().clone().requires_grad_(True)
    anchor_sigma = sigma_clean.detach().clone()
    optimizer_b = torch.optim.Adam([sigma], lr=max(min_lr, float(lr) * 0.4))
    stage2_steps = int(max(0, stage2_steps))
    for step in range(stage2_steps):
        if relinearize_every > 0 and (step == 0 or step % int(relinearize_every) == 0):
            backend.prepare(measurements, sigma.detach())
        optimizer_b.zero_grad()
        pred_meas = backend.predict(sigma)
        data_loss = torch.mean((pred_meas - measurements) ** 2)
        tv_field = tv_loss(sigma)
        area_loss = torch.tensor(0.0, device=device, dtype=sigma_start.dtype)
        shape_anchor_loss = torch.tensor(0.0, device=device, dtype=sigma_start.dtype)
        speckle_loss = torch.tensor(0.0, device=device, dtype=sigma_start.dtype)
        sigma_anchor_loss = torch.mean((sigma - anchor_sigma) ** 2)
        total_loss = data_loss + (0.5 * lambda_length) * tv_field + lambda_anchor_sigma * sigma_anchor_loss
        total_loss.backward()
        optimizer_b.step()
        with torch.no_grad():
            sigma.clamp_(min=-1.3, max=1.3)
        history["total_loss"].append(float(total_loss.detach().cpu().item()))
        history["data_loss"].append(float(data_loss.detach().cpu().item()))
        history["tv_loss"].append(float(tv_field.detach().cpu().item()))
        history["area_loss"].append(float(area_loss.detach().cpu().item()))
        history["shape_anchor_loss"].append(float(shape_anchor_loss.detach().cpu().item()))
        history["speckle_loss"].append(float(speckle_loss.detach().cpu().item()))
        history["sigma_anchor_loss"].append(float(sigma_anchor_loss.detach().cpu().item()))
        history["lr"].append(float(optimizer_b.param_groups[0]["lr"]))
        history["tau"].append(float(tau_end))
        history["stage"].append("B")
    sigma_final = sigma.detach() if stage2_steps > 0 else sigma_clean.detach()
    ternary_final = torch.from_numpy(ternary_from_sigma(sigma_final[0, 0].detach().cpu().numpy(), threshold=seg_threshold)).to(
        device=device, dtype=sigma_start.dtype
    ).unsqueeze(0).unsqueeze(0)
    return sigma_final, ternary_final, history


def save_loss_curve(history: Dict[str, list], save_json: Path, save_png: Path) -> None:
    """Save optimization loss history as json and plot."""
    save_json.parent.mkdir(parents=True, exist_ok=True)
    with open(save_json, "w") as f:
        json.dump(history, f, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for key in ["total_loss", "data_loss", "tv_loss", "anchor_loss", "area_loss", "shape_anchor_loss", "speckle_loss", "sigma_anchor_loss"]:
        if key in history:
            ax.plot(history[key], label=key)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(save_png), dpi=150, bbox_inches="tight")
    plt.close(fig)
