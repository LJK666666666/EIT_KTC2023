"""
验证 GroundTruths 与 evaluation_datasets 的前向物理一致性。

流程:
1) 用 ref.mat 估计背景电导率 sigma_bg（匹配 Uelref）
2) 在 sigma_bg 线性化，用 Jacobian 拟合标签1/2对应电导率增量
3) 用完整前向模型重模拟每个样本，计算 Relative Error 与 Corr
4) 输出图像与 summary.json 到 results/forward_validation_{num}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.optimize as opt
from tqdm import tqdm

from sim_dataset import load_mesh

import sys

project_root = Path(__file__).resolve().parent.parent
ktc_path = project_root / "src" / "ktc_methods"
if str(ktc_path) not in sys.path:
    sys.path.insert(0, str(ktc_path))

import KTCFwd  # type: ignore


def next_indexed_dir(base_dir: Path) -> Path:
    parent = base_dir.parent
    base_name = base_dir.name
    max_idx = -1
    if parent.exists():
        pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{2}})$")
        for item in parent.iterdir():
            if item.is_dir():
                m = pattern.match(item.name)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
    return parent / f"{base_name}_{max_idx + 1:02d}"


def load_level_sample(level: int, sample_id: int) -> Dict[str, np.ndarray]:
    data_path = project_root / "EvaluationData_full" / "evaluation_datasets" / f"level{level}" / f"data{sample_id}.mat"
    ref_path = project_root / "EvaluationData_full" / "evaluation_datasets" / f"level{level}" / "ref.mat"
    gt_path = project_root / "EvaluationData_full" / "GroundTruths" / f"level_{level}" / f"{sample_id}_true.mat"

    data = sio.loadmat(str(data_path))
    ref = sio.loadmat(str(ref_path))
    gt = sio.loadmat(str(gt_path))

    return {
        "Uel": np.asarray(data["Uel"], dtype=np.float64).reshape(-1, 1),
        "Uelref": np.asarray(ref["Uelref"], dtype=np.float64).reshape(-1, 1),
        "Inj": np.asarray(ref["Injref"], dtype=np.float64),
        "Mpat": np.asarray(ref["Mpat"], dtype=np.float64),
        "truth": np.asarray(gt["truth"], dtype=np.uint8),
    }


def load_training_sample(sample_id: int) -> Dict[str, np.ndarray]:
    data_path = project_root / "KTC2023" / "TrainingData" / f"data{sample_id}.mat"
    ref_path = project_root / "KTC2023" / "TrainingData" / "ref.mat"
    gt_path = project_root / "KTC2023" / "GroundTruths" / f"true{sample_id}.mat"

    data = sio.loadmat(str(data_path))
    ref = sio.loadmat(str(ref_path))
    gt = sio.loadmat(str(gt_path))
    return {
        "Uel": np.asarray(data["Uel"], dtype=np.float64).reshape(-1, 1),
        "Uelref": np.asarray(ref["Uelref"], dtype=np.float64).reshape(-1, 1),
        "Inj": np.asarray(ref["Injref"], dtype=np.float64),
        "Mpat": np.asarray(ref["Mpat"], dtype=np.float64),
        "truth": np.asarray(gt["truth"], dtype=np.uint8),
    }


def build_vincl_flat(level: int, injref: np.ndarray, nel: int = 32) -> np.ndarray:
    """
    官方 main.py 的难度掩码逻辑，返回 shape [76*31] 的 flatten mask。
    """
    vincl = np.ones(((nel - 1), 76), dtype=bool)
    rmind = np.arange(0, 2 * (level - 1), 1)
    for ii in range(0, 75):
        for jj in rmind:
            if injref[jj, ii]:
                vincl[:, ii] = 0
            vincl[jj, :] = 0
    return vincl.T.flatten()


def node_to_pixel_indices(mesh_nodes: np.ndarray, h: int = 256, w: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    x = mesh_nodes[:, 0]
    y = mesh_nodes[:, 1]
    col = np.rint((x + 0.115) / 0.23 * (w - 1)).astype(np.int32)
    row = np.rint((0.115 - y) / 0.23 * (h - 1)).astype(np.int32)
    col = np.clip(col, 0, w - 1)
    row = np.clip(row, 0, h - 1)
    return row, col


def apply_truth_transform(truth: np.ndarray, transform: str) -> np.ndarray:
    if transform == "id":
        return truth
    if transform == "flipud":
        return np.flipud(truth)
    if transform == "fliplr":
        return np.fliplr(truth)
    if transform == "rot90":
        return np.rot90(truth, 1)
    if transform == "rot180":
        return np.rot90(truth, 2)
    if transform == "rot270":
        return np.rot90(truth, 3)
    if transform == "rot90_flipud":
        return np.flipud(np.rot90(truth, 1))
    if transform == "rot270_flipud":
        return np.flipud(np.rot90(truth, 3))
    raise ValueError(f"Unknown transform: {transform}")


def truth_to_templates(truth: np.ndarray, row_idx: np.ndarray, col_idx: np.ndarray, transform: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = apply_truth_transform(truth, transform)
    labels = arr[row_idx, col_idx]
    t1 = (labels == 1).astype(np.float64).reshape(-1, 1)
    t2 = (labels == 2).astype(np.float64).reshape(-1, 1)
    return t1, t2


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(b)
    if denom < 1e-12:
        return float(np.linalg.norm(a - b))
    return float(np.linalg.norm(a - b) / denom)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a1 = a.reshape(-1)
    b1 = b.reshape(-1)
    sa = np.std(a1)
    sb = np.std(b1)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a1, b1)[0, 1])


def estimate_sigma_bg_and_z(solver, uref: np.ndarray, n_nodes: int) -> Tuple[float, float, float]:
    sigma_candidates = np.linspace(0.55, 1.05, 11)
    logz_candidates = np.linspace(-8.0, -4.0, 9)
    best_sigma = float(sigma_candidates[0])
    best_logz = float(logz_candidates[0])
    best_err = 1e9
    for sigma_bg in tqdm(sigma_candidates, desc="Estimating sigma_bg/z", ncols=100):
        sigma = np.full((n_nodes, 1), float(sigma_bg), dtype=np.float64)
        for logz in logz_candidates:
            z = (10.0 ** float(logz)) * np.ones((32, 1), dtype=np.float64)
            u = np.asarray(solver.SolveForward(sigma.copy(), z), dtype=np.float64).reshape(-1, 1)
            err = relative_error(u, uref)
            if err < best_err:
                best_err = err
                best_sigma = float(sigma_bg)
                best_logz = float(logz)
    return best_sigma, best_logz, best_err


def build_sample_templates(
    samples: List[Dict[str, np.ndarray]],
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    transform: str,
    swap: bool
) -> List[Dict[str, np.ndarray]]:
    out = []
    for s in samples:
        t1, t2 = truth_to_templates(s["truth"], row_idx, col_idx, transform=transform)
        if swap:
            t1, t2 = t2, t1
        out.append({
            "level": int(s["level"]),
            "sample_id": int(s["sample_id"]),
            "t1": t1,
            "t2": t2,
            "Uel": s["Uel"],
            "Uelref": s["Uelref"],
            "vincl_flat": s["vincl_flat"],
            "truth": s["truth"],
        })
    return out


def evaluate_global_params(
    solver,
    templates: List[Dict[str, np.ndarray]],
    sigma_bg: float,
    sigma1: float,
    sigma2: float,
    log10_z: float
) -> Tuple[float, List[Dict[str, float]], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    z = (10.0 ** float(log10_z)) * np.ones((32, 1), dtype=np.float64)
    details = []
    all_pred = []
    all_meas = []
    rep = {}

    for t in templates:
        sigma_nodes = sigma_bg + (sigma1 - sigma_bg) * t["t1"] + (sigma2 - sigma_bg) * t["t2"]
        u_pred = np.asarray(solver.SolveForward(sigma_nodes.copy(), z.copy()), dtype=np.float64).reshape(-1, 1)
        du_pred = u_pred - t["Uelref"]
        du_meas = t["Uel"] - t["Uelref"]
        mask = t["vincl_flat"]
        du_pred_mask = du_pred[mask]
        du_meas_mask = du_meas[mask]
        rel = relative_error(du_pred_mask, du_meas_mask)
        corr = corrcoef(du_pred_mask, du_meas_mask)
        details.append({
            "level": t["level"],
            "sample_id": t["sample_id"],
            "relative_error": rel,
            "corr": corr
        })
        all_pred.append(du_pred_mask.reshape(-1))
        all_meas.append(du_meas_mask.reshape(-1))
        if not rep:
            rep = {
                "du_meas": du_meas.copy(),
                "du_pred": du_pred.copy(),
                "mask": mask.copy(),
                "truth": t["truth"].copy()
            }

    pred_cat = np.concatenate(all_pred, axis=0)
    meas_cat = np.concatenate(all_meas, axis=0)
    agg_rel = relative_error(pred_cat, meas_cat)
    return agg_rel, details, pred_cat, meas_cat, rep


def optimize_global_params(
    solver,
    templates: List[Dict[str, np.ndarray]],
    init_sigma_bg: float,
    init_sigma1: float,
    init_sigma2: float,
    init_log10_z: float,
    max_iter: int = 36
) -> Dict[str, float]:
    x0 = np.array([init_sigma_bg, init_sigma1, init_sigma2, init_log10_z], dtype=np.float64)
    bounds = [(0.45, 1.20), (0.05, 1.80), (0.10, 3.20), (-8.0, -3.0)]

    def objective(x):
        sigma_bg = float(x[0])
        sigma1 = float(x[1])
        sigma2 = float(x[2])
        log10_z = float(x[3])
        agg_rel, _, _, _, _ = evaluate_global_params(
            solver=solver,
            templates=templates,
            sigma_bg=sigma_bg,
            sigma1=sigma1,
            sigma2=sigma2,
            log10_z=log10_z
        )
        penalty = 0.0
        if sigma1 >= sigma2:
            penalty += 0.5 * (sigma1 - sigma2 + 1e-3) ** 2
        return agg_rel + penalty

    result = opt.minimize(
        objective,
        x0=x0,
        method="Powell",
        bounds=bounds,
        options={
            "maxiter": int(max_iter),
            "maxfev": int(max_iter * 24),
            "xtol": 1e-3,
            "ftol": 1e-4,
            "disp": False
        }
    )

    x = result.x
    return {
        "sigma_bg": float(x[0]),
        "sigma1": float(x[1]),
        "sigma2": float(x[2]),
        "log10_z": float(x[3]),
        "objective": float(result.fun),
        "nfev": int(result.nfev),
        "nit": int(getattr(result, "nit", -1)),
        "success": bool(result.success),
        "message": str(result.message),
    }


def fit_label_contrast_linear(
    samples: List[Dict[str, np.ndarray]],
    j_full: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    sigma_bg: float
) -> Dict:
    transforms = ["id", "flipud", "fliplr", "rot90", "rot180", "rot270", "rot90_flipud", "rot270_flipud"]
    combos = []
    for tr in transforms:
        combos.append((f"{tr}_noswap", tr, False))
        combos.append((f"{tr}_swap", tr, True))
    best = None

    for name, transform, swap in combos:
        x_rows = []
        y_rows = []
        rel_list = []
        for s in samples:
            t1, t2 = truth_to_templates(s["truth"], row_idx, col_idx, transform=transform)
            if swap:
                t1, t2 = t2, t1
            phi1 = j_full @ t1
            phi2 = j_full @ t2
            x = np.hstack([phi1, phi2])  # [M,2]
            y = s["Uel"] - s["Uelref"]    # [M,1]
            mask = s["vincl_flat"]
            x = x[mask]
            y = y[mask]
            x_rows.append(x)
            y_rows.append(y)
        x_all = np.vstack(x_rows)
        y_all = np.vstack(y_rows)
        lower = np.array([0.05 - sigma_bg, 0.05 - sigma_bg], dtype=np.float64)
        upper = np.array([2.50 - sigma_bg, 2.50 - sigma_bg], dtype=np.float64)
        sol = opt.lsq_linear(x_all, y_all.reshape(-1), bounds=(lower, upper), lsmr_tol="auto", verbose=0)
        d1 = float(sol.x[0])
        d2 = float(sol.x[1])

        for s in samples:
            t1, t2 = truth_to_templates(s["truth"], row_idx, col_idx, transform=transform)
            if swap:
                t1, t2 = t2, t1
            pred = (j_full @ (d1 * t1 + d2 * t2)).reshape(-1, 1)
            y = (s["Uel"] - s["Uelref"]).reshape(-1, 1)
            mask = s["vincl_flat"]
            rel_list.append(relative_error(pred[mask], y[mask]))
        mean_rel = float(np.mean(rel_list))

        item = {
            "name": name,
            "transform": transform,
            "swap": swap,
            "delta1": d1,
            "delta2": d2,
            "sigma1": sigma_bg + d1,
            "sigma2": sigma_bg + d2,
            "linear_mean_rel": mean_rel,
        }
        if best is None or item["linear_mean_rel"] < best["linear_mean_rel"]:
            best = item

    if best is None:
        raise ValueError("Linear fitting failed.")
    return best


def main():
    parser = argparse.ArgumentParser(description="Validate forward model using EvaluationData_full mapping")
    parser.add_argument("--levels", type=str, default="1,2,3,4,5,6,7", help="Levels, comma-separated")
    parser.add_argument("--samples_per_level", type=int, default=3, help="Samples per level")
    parser.add_argument("--mesh_name", type=str, default="Mesh_sparse.mat", help="KTC mesh for forward simulation")
    parser.add_argument("--z_contact", type=float, default=1e-6, help="Contact impedance")
    parser.add_argument("--result_dir", type=str, default="forward_validation", help="Result base directory name")
    parser.add_argument("--opt_max_iter", type=int, default=36, help="Global nonlinear optimization max iterations")
    parser.add_argument("--skip_global_opt", action="store_true", help="Skip slow global nonlinear optimization")
    parser.add_argument(
        "--use_full_measurements",
        action="store_true",
        help="Use all measurements (recommended for EvaluationData_full)"
    )
    parser.add_argument(
        "--calibration_source",
        type=str,
        default="trainingdata",
        choices=["trainingdata", "evaluation"],
        help="参数标定数据源：trainingdata=KTC2023/TrainingData，evaluation=当前评估集"
    )
    parser.add_argument(
        "--target_source",
        type=str,
        default="evaluation",
        choices=["evaluation", "trainingdata"],
        help="待检验目标数据源"
    )
    args = parser.parse_args()

    samples: List[Dict[str, np.ndarray]] = []
    if args.target_source == "evaluation":
        level_ids = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
        for level in level_ids:
            for sid in range(1, args.samples_per_level + 1):
                item = load_level_sample(level, sid)
                item["level"] = level
                item["sample_id"] = sid
                if args.use_full_measurements:
                    item["vincl_flat"] = np.ones((item["Uel"].shape[0],), dtype=bool)
                else:
                    item["vincl_flat"] = build_vincl_flat(level, item["Inj"])
                samples.append(item)
    else:
        for sid in range(1, args.samples_per_level + 1):
            item = load_training_sample(sid)
            item["level"] = 0
            item["sample_id"] = sid
            item["vincl_flat"] = np.ones((item["Uel"].shape[0],), dtype=bool)
            samples.append(item)

    calib_samples: List[Dict[str, np.ndarray]] = []
    if args.calibration_source == "trainingdata":
        for sid in range(1, 5):
            item = load_training_sample(sid)
            item["level"] = 0
            item["sample_id"] = sid
            item["vincl_flat"] = np.ones((item["Uel"].shape[0],), dtype=bool)
            calib_samples.append(item)
    else:
        calib_samples = samples

    out_dir = next_indexed_dir(project_root / "results" / args.result_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    mesh, mesh2 = load_mesh(args.mesh_name)
    row_idx, col_idx = node_to_pixel_indices(mesh.g, 256, 256)

    inj = samples[0]["Inj"]
    mpat = samples[0]["Mpat"]
    vincl = np.ones((mpat.shape[1], inj.shape[1]), dtype=bool)
    solver = KTCFwd.EITFEM(mesh2, inj, mpat, vincl)
    n_nodes = mesh.g.shape[0]

    sigma_bg, log10_z_init, bg_ref_err = estimate_sigma_bg_and_z(solver, samples[0]["Uelref"], n_nodes)
    z_init = (10.0 ** log10_z_init) * np.ones((32, 1), dtype=np.float64)
    sigma_bg_nodes = np.full((n_nodes, 1), sigma_bg, dtype=np.float64)

    j_full = np.asarray(solver.Jacobian(sigma_bg_nodes, z_init), dtype=np.float64)

    best_linear = fit_label_contrast_linear(
        samples=calib_samples,
        j_full=j_full,
        row_idx=row_idx,
        col_idx=col_idx,
        sigma_bg=sigma_bg
    )

    sigma1_init = float(best_linear["sigma1"])
    sigma2_init = float(best_linear["sigma2"])
    transform = str(best_linear["transform"])
    swap = bool(best_linear["swap"])

    templates = build_sample_templates(
        samples=samples,
        row_idx=row_idx,
        col_idx=col_idx,
        transform=transform,
        swap=swap
    )

    if args.skip_global_opt:
        opt_result = {
            "sigma_bg": float(sigma_bg),
            "sigma1": float(sigma1_init),
            "sigma2": float(sigma2_init),
            "log10_z": float(log10_z_init),
            "objective": None,
            "nfev": 0,
            "nit": 0,
            "success": True,
            "message": "Skipped by --skip_global_opt",
        }
    else:
        opt_result = optimize_global_params(
            solver=solver,
            templates=templates,
            init_sigma_bg=sigma_bg,
            init_sigma1=sigma1_init,
            init_sigma2=sigma2_init,
            init_log10_z=log10_z_init,
            max_iter=args.opt_max_iter
        )

    sigma_bg = opt_result["sigma_bg"]
    sigma1 = opt_result["sigma1"]
    sigma2 = opt_result["sigma2"]
    log10_z = opt_result["log10_z"]

    _, detail_rows, pred_cat, meas_cat, rep = evaluate_global_params(
        solver=solver,
        templates=templates,
        sigma_bg=sigma_bg,
        sigma1=sigma1,
        sigma2=sigma2,
        log10_z=log10_z
    )

    per_sample_calibrated = []
    for s in tqdm(samples, desc="Running per-sample linear fit", ncols=100):
        t1_s, t2_s = truth_to_templates(s["truth"], row_idx, col_idx, transform=transform)
        if swap:
            t1_s, t2_s = t2_s, t1_s
        x_s = np.hstack([j_full @ t1_s, j_full @ t2_s])
        y_s = (s["Uel"] - s["Uelref"]).reshape(-1)
        mask = s["vincl_flat"]
        x_s = x_s[mask]
        y_s = y_s[mask]
        du_meas_mask = y_s.reshape(-1, 1)

        lower = np.array([0.05 - sigma_bg, 0.05 - sigma_bg], dtype=np.float64)
        upper = np.array([2.50 - sigma_bg, 2.50 - sigma_bg], dtype=np.float64)
        sol_s = opt.lsq_linear(x_s, y_s, bounds=(lower, upper), lsmr_tol="auto", verbose=0)
        d1_s = float(sol_s.x[0])
        d2_s = float(sol_s.x[1])
        sigma1_s = sigma_bg + d1_s
        sigma2_s = sigma_bg + d2_s

        z_eval = (10.0 ** log10_z) * np.ones((32, 1), dtype=np.float64)
        sigma_nodes_s = sigma_bg + (sigma1_s - sigma_bg) * t1_s + (sigma2_s - sigma_bg) * t2_s
        u_pred_s = np.asarray(solver.SolveForward(sigma_nodes_s, z_eval), dtype=np.float64).reshape(-1, 1)
        du_pred_s = u_pred_s - s["Uelref"]
        du_pred_s_mask = du_pred_s[mask]

        rel_s = relative_error(du_pred_s_mask, du_meas_mask)
        corr_s = corrcoef(du_pred_s_mask, du_meas_mask)
        per_sample_calibrated.append({
            "level": int(s["level"]),
            "sample_id": int(s["sample_id"]),
            "sigma1": float(sigma1_s),
            "sigma2": float(sigma2_s),
            "relative_error": float(rel_s),
            "corr": float(corr_s),
        })

    rel_all = np.array([r["relative_error"] for r in detail_rows], dtype=np.float64)
    corr_all = np.array([r["corr"] for r in detail_rows], dtype=np.float64)
    rel_cal = np.array([r["relative_error"] for r in per_sample_calibrated], dtype=np.float64)
    corr_cal = np.array([r["corr"] for r in per_sample_calibrated], dtype=np.float64)

    # Global scaling/affine diagnostics
    denom = float(np.dot(pred_cat, pred_cat)) + 1e-12
    alpha = float(np.dot(pred_cat, meas_cat) / denom)
    rel_scaled = relative_error(alpha * pred_cat, meas_cat)
    x_aff = np.column_stack([pred_cat, np.ones_like(pred_cat)])
    beta, _, _, _ = np.linalg.lstsq(x_aff, meas_cat, rcond=None)
    rel_affine = relative_error(x_aff @ beta, meas_cat)

    # Plot 1: per-sample relative error
    labels = [f"L{r['level']}-S{r['sample_id']}" for r in detail_rows]
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(len(detail_rows)), rel_all)
    plt.plot(np.arange(len(detail_rows)), np.full_like(rel_all, rel_all.mean()), "r--")
    plt.xticks(np.arange(len(detail_rows)), labels, rotation=45, ha="right")
    plt.ylabel("Relative Error")
    plt.tight_layout()
    plt.savefig(out_dir / "relative_error_per_sample.png", dpi=150)
    plt.close()

    # Plot 2: representative curve and scatter
    if rep is not None:
        du_meas = rep["du_meas"].reshape(-1)
        du_pred = rep["du_pred"].reshape(-1)
        mask = rep["mask"].reshape(-1).astype(bool)
        du_meas_mask = du_meas[mask]
        du_pred_mask = du_pred[mask]
        n_show = min(600, du_meas_mask.size)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].imshow(rep["truth"], cmap="gray")
        axes[0].set_xticks([])
        axes[0].set_yticks([])

        axes[1].plot(du_meas_mask[:n_show], label="Measured")
        axes[1].plot(du_pred_mask[:n_show], label="Simulated")
        axes[1].set_xlabel("Measurement Index")
        axes[1].set_ylabel("DeltaU")
        axes[1].legend()

        axes[2].scatter(du_meas_mask, du_pred_mask, s=6, alpha=0.5)
        vmin = min(float(du_meas_mask.min()), float(du_pred_mask.min()))
        vmax = max(float(du_meas_mask.max()), float(du_pred_mask.max()))
        axes[2].plot([vmin, vmax], [vmin, vmax], "r--")
        axes[2].set_xlabel("Measured DeltaU")
        axes[2].set_ylabel("Simulated DeltaU")
        plt.tight_layout()
        plt.savefig(out_dir / "representative_comparison.png", dpi=150)
        plt.close()

    # Plot 3: corr distribution
    plt.figure(figsize=(6, 4))
    plt.hist(corr_all, bins=10)
    plt.xlabel("Correlation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "corr_histogram.png", dpi=150)
    plt.close()

    # Plot 4: global-fit vs per-sample-fit relative error
    plt.figure(figsize=(7, 4))
    plt.hist(rel_all, bins=10, alpha=0.6, label="Global mapping")
    plt.hist(rel_cal, bins=10, alpha=0.6, label="Per-sample mapping")
    plt.xlabel("Relative Error")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "relative_error_global_vs_sample_fit.png", dpi=150)
    plt.close()

    summary = {
        "num_samples": len(detail_rows),
        "mesh_name": args.mesh_name,
        "target_source": args.target_source,
        "calibration_source": args.calibration_source,
        "num_calibration_samples": len(calib_samples),
        "sigma_bg_estimated": sigma_bg,
        "z_contact_estimated": float(10.0 ** log10_z),
        "log10_z_estimated": float(log10_z),
        "sigma_bg_ref_relative_error": bg_ref_err,
        "best_linear_mapping": best_linear,
        "global_nonlinear_optimization": opt_result,
        "nonlinear_metrics": {
            "relative_error_mean": float(rel_all.mean()),
            "relative_error_std": float(rel_all.std()),
            "relative_error_min": float(rel_all.min()),
            "relative_error_max": float(rel_all.max()),
            "corr_mean": float(corr_all.mean()),
            "corr_std": float(corr_all.std()),
            "corr_min": float(corr_all.min()),
            "corr_max": float(corr_all.max())
        },
        "per_sample": detail_rows,
        "per_sample_calibrated_metrics": {
            "relative_error_mean": float(rel_cal.mean()),
            "relative_error_std": float(rel_cal.std()),
            "relative_error_min": float(rel_cal.min()),
            "relative_error_max": float(rel_cal.max()),
            "corr_mean": float(corr_cal.mean()),
            "corr_std": float(corr_cal.std()),
            "corr_min": float(corr_cal.min()),
            "corr_max": float(corr_cal.max())
        },
        "per_sample_calibrated": per_sample_calibrated,
        "global_scale_diagnostics": {
            "alpha_only": alpha,
            "relative_error_after_alpha": rel_scaled,
            "affine_a": float(beta[0]),
            "affine_b": float(beta[1]),
            "relative_error_after_affine": rel_affine
        }
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Validation results saved to: {out_dir}")
    print(f"Mean Relative Error: {summary['nonlinear_metrics']['relative_error_mean']:.6f}")
    print(f"Mean Corr: {summary['nonlinear_metrics']['corr_mean']:.6f}")


if __name__ == "__main__":
    main()
