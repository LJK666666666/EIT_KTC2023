"""
Linearized KTC physics backend for test-time optimization.
Uses full 32-electrode KTC forward/Jacobian, then maps deltaU(2356) to compact
208-dim ys protocol (16x13) via sparse linear calibration inferred from
data/test2023 and corresponding .mat files.
"""
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import sys
import re

import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch

from .physics_backend import PhysicsBackend


ktc_path = Path(__file__).resolve().parents[2] / "ktc_methods"
if str(ktc_path) not in sys.path:
    sys.path.insert(0, str(ktc_path))

import KTCFwd  # type: ignore
from ...sim_dataset import load_mesh


def _valid_eim_columns(inj_idx: int, num_electrodes: int = 16) -> List[int]:
    skip = {(inj_idx - 1) % num_electrodes, inj_idx, (inj_idx + 1) % num_electrodes}
    return [j for j in range(num_electrodes) if j not in skip]


def _build_expand_index() -> List[Tuple[int, int, int]]:
    idx_map = []
    compact_idx = 0
    for row in range(16):
        for col in _valid_eim_columns(row, 16):
            idx_map.append((compact_idx, row, col))
            compact_idx += 1
    return idx_map


def _build_node_sampling_matrix(node_xy: np.ndarray, h: int, w: int) -> sp.csr_matrix:
    n_nodes = node_xy.shape[0]
    rows = []
    cols = []
    vals = []
    for i in range(n_nodes):
        x = float(node_xy[i, 0])
        y = float(node_xy[i, 1])
        col_f = (x + 0.115) / 0.23 * (w - 1)
        row_f = (0.115 - y) / 0.23 * (h - 1)
        col_f = min(max(col_f, 0.0), w - 1.0)
        row_f = min(max(row_f, 0.0), h - 1.0)
        c0 = int(np.floor(col_f))
        r0 = int(np.floor(row_f))
        c1 = min(c0 + 1, w - 1)
        r1 = min(r0 + 1, h - 1)
        wc = col_f - c0
        wr = row_f - r0
        w00 = (1.0 - wr) * (1.0 - wc)
        w01 = (1.0 - wr) * wc
        w10 = wr * (1.0 - wc)
        w11 = wr * wc
        idx00 = r0 * w + c0
        idx01 = r0 * w + c1
        idx10 = r1 * w + c0
        idx11 = r1 * w + c1
        rows.extend([i, i, i, i])
        cols.extend([idx00, idx01, idx10, idx11])
        vals.extend([w00, w01, w10, w11])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes, h * w), dtype=np.float64)


def _build_sparse_protocol_mapper(project_root: Path, top_k: int = 8) -> Dict[str, np.ndarray]:
    data_dir = project_root / "data" / "test2023"
    if not data_dir.exists():
        raise ValueError(f"Protocol calibration directory not found: {data_dir}")
    pattern = re.compile(r"^(\d+)_(\d+)\.npz$")
    du_list = []
    ys_list = []
    for npz_path in sorted(data_dir.glob("*.npz")):
        m = pattern.match(npz_path.name)
        if m is None:
            continue
        level = int(m.group(1))
        sample_id = int(m.group(2))
        if level == 0:
            data_mat = project_root / "KTC2023" / "TrainingData" / f"data{sample_id}.mat"
            ref_mat = project_root / "KTC2023" / "TrainingData" / "ref.mat"
        else:
            data_mat = project_root / "EvaluationData_full" / "evaluation_datasets" / f"level{level}" / f"data{sample_id}.mat"
            ref_mat = project_root / "EvaluationData_full" / "evaluation_datasets" / f"level{level}" / "ref.mat"
        if not data_mat.exists() or not ref_mat.exists():
            continue
        y_data = np.load(npz_path)
        ys = np.asarray(y_data["ys"], dtype=np.float64).reshape(-1)
        data = sio.loadmat(str(data_mat))
        ref = sio.loadmat(str(ref_mat))
        du = (np.asarray(data["Uel"], dtype=np.float64).reshape(-1) - np.asarray(ref["Uelref"], dtype=np.float64).reshape(-1))
        if ys.size != 208 or du.size != 2356:
            continue
        du_list.append(du)
        ys_list.append(ys)
    if len(du_list) < 12:
        raise ValueError(f"Not enough aligned samples for protocol calibration, got {len(du_list)}")
    du_mat = np.stack(du_list, axis=0)
    ys_mat = np.stack(ys_list, axis=0)
    n_samples = du_mat.shape[0]
    du_std = du_mat.std(axis=0, keepdims=True) + 1e-12
    ys_std = ys_mat.std(axis=0, keepdims=True) + 1e-12
    du_norm = (du_mat - du_mat.mean(axis=0, keepdims=True)) / du_std
    ys_norm = (ys_mat - ys_mat.mean(axis=0, keepdims=True)) / ys_std
    corr = (du_norm.T @ ys_norm) / max(n_samples - 1, 1)
    abs_corr = np.abs(corr)
    out_dim = ys_mat.shape[1]
    idx = np.zeros((out_dim, top_k), dtype=np.int64)
    weight = np.zeros((out_dim, top_k), dtype=np.float64)
    bias = np.zeros((out_dim,), dtype=np.float64)
    for j in range(out_dim):
        cols = np.argpartition(abs_corr[:, j], -top_k)[-top_k:]
        x = du_mat[:, cols]
        a = np.concatenate([x, np.ones((n_samples, 1), dtype=np.float64)], axis=1)
        coef = np.linalg.lstsq(a, ys_mat[:, j], rcond=None)[0]
        idx[j, :] = cols
        weight[j, :] = coef[:top_k]
        bias[j] = coef[top_k]
    return {"idx": idx, "weight": weight, "bias": bias}


class LinearizedKTCBackend(PhysicsBackend):
    """
    y_lin(sigma_img) = y0 + J_img @ (sigma_img - sigma0_img)
    where y is in normalized EIM domain [1,1,16,16].
    """
    def __init__(
        self,
        output_size: int = 128,
        device: str = "cuda",
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        voltage: float = 1.0
    ):
        self.output_size = output_size
        self.device = torch.device(device)
        if mean is None or std is None:
            raise ValueError("KTC backend requires dataset mean/std for EIM normalization.")
        self.mean_16x13 = mean.detach().cpu().numpy().reshape(16, 13).astype(np.float64)
        self.std_16x13 = std.detach().cpu().numpy().reshape(16, 13).astype(np.float64)
        self.voltage = float(voltage)
        self.project_root = Path(__file__).resolve().parents[3]
        self.mapper = _build_sparse_protocol_mapper(self.project_root, top_k=8)
        self.mesh, self.mesh2 = load_mesh("Mesh_sparse.mat")
        ref_mat_path = self.project_root / "EvaluationData_full" / "evaluation_datasets" / "level1" / "ref.mat"
        if not ref_mat_path.exists():
            raise ValueError(f"Reference file not found: {ref_mat_path}")
        ref_mat = sio.loadmat(str(ref_mat_path))
        self.inj = np.asarray(ref_mat["Injref"], dtype=np.float64)
        self.mpat = np.asarray(ref_mat["Mpat"], dtype=np.float64)
        self.uref_full_vec = np.asarray(ref_mat["Uelref"], dtype=np.float64).reshape(-1)
        if self.uref_full_vec.size != self.mpat.shape[1] * self.inj.shape[1]:
            raise ValueError(
                f"Unexpected Uelref size: {self.uref_full_vec.size}, expected {self.mpat.shape[1] * self.inj.shape[1]}"
            )
        self.vincl = np.ones((self.mpat.shape[1], self.inj.shape[1]), dtype=bool)
        self.solver = KTCFwd.EITFEM(self.mesh2, self.inj, self.mpat, self.vincl)
        self.z = (1e-6) * np.ones((32, 1), dtype=np.float64)
        self.n_nodes = self.mesh.g.shape[0]
        self.expand_index = _build_expand_index()
        self._w_img_to_nodes: Optional[sp.csr_matrix] = None
        self._sigma0_flat_t: Optional[torch.Tensor] = None
        self._y0_compact_t: Optional[torch.Tensor] = None
        self._j_img_t: Optional[torch.Tensor] = None

    def prepare(self, measurements, sigma_init) -> None:
        sigma0 = sigma_init.detach().to(self.device)
        if sigma0.shape[0] != 1 or sigma0.shape[1] != 1:
            raise ValueError(f"Only single-sample sigma_init [1,1,H,W] is supported, got {sigma0.shape}")
        if measurements.shape != (1, 1, 16, 16):
            raise ValueError(f"Measurement shape must be [1,1,16,16], got {measurements.shape}")
        h, w = int(sigma0.shape[-2]), int(sigma0.shape[-1])
        if self._w_img_to_nodes is None or self._w_img_to_nodes.shape[1] != h * w:
            self._w_img_to_nodes = _build_node_sampling_matrix(self.mesh.g, h, w)
        sigma0_img = sigma0[0, 0].detach().cpu().numpy().astype(np.float64)
        sigma0_flat = sigma0_img.reshape(-1)
        sigma0_nodes = self._w_img_to_nodes @ sigma0_flat
        sigma0_phys = np.clip(0.745 + sigma0_nodes, 1e-6, None).reshape(-1, 1)
        u0, j_full = self._solve_u_and_jac(sigma0_phys, need_jac=True)
        if j_full is None:
            raise ValueError("Internal error: Jacobian not available.")
        du0 = u0 - self.uref_full_vec
        y0_raw = self._map_du_to_compact(du0)
        j_raw_mesh = self._map_jac_to_compact(j_full)
        j_img = j_raw_mesh @ self._w_img_to_nodes
        mean_flat = self.mean_16x13.reshape(-1)
        std_flat = self.std_16x13.reshape(-1)
        scale = 1.0 / (self.voltage * std_flat)
        y0_norm = (y0_raw / self.voltage - mean_flat) / std_flat
        j_norm = j_img * scale[:, None]
        self._sigma0_flat_t = torch.from_numpy(sigma0_flat.astype(np.float32)).to(self.device)
        self._y0_compact_t = torch.from_numpy(y0_norm.astype(np.float32)).to(self.device)
        self._j_img_t = torch.from_numpy(j_norm.astype(np.float32)).to(self.device)

    def predict(self, sigma):
        if self._sigma0_flat_t is None or self._y0_compact_t is None or self._j_img_t is None:
            raise ValueError("Backend is not prepared. Call prepare() before predict().")
        sigma_flat = sigma[0, 0].reshape(-1)
        delta = sigma_flat - self._sigma0_flat_t
        y_compact = self._y0_compact_t + torch.matmul(self._j_img_t, delta)
        y_eim = torch.zeros((1, 1, 16, 16), device=sigma.device, dtype=sigma.dtype)
        for compact_idx, row, col in self.expand_index:
            y_eim[0, 0, row, col] = y_compact[compact_idx]
        return y_eim

    def _solve_u_and_jac(self, sigma_nodes: np.ndarray, need_jac: bool):
        u = np.asarray(self.solver.SolveForward(sigma_nodes, self.z), dtype=np.float64).reshape(-1)
        if not need_jac:
            return u, None
        j = np.asarray(self.solver.Jacobian(sigma_nodes, self.z), dtype=np.float64)
        if j.shape[0] != u.size or j.shape[1] != self.n_nodes:
            raise ValueError(f"Unexpected Jacobian shape {j.shape}, expected ({u.size}, {self.n_nodes})")
        return u, j

    def _map_du_to_compact(self, du_full: np.ndarray) -> np.ndarray:
        idx = self.mapper["idx"]
        weight = self.mapper["weight"]
        bias = self.mapper["bias"]
        x = du_full[idx]
        y = np.sum(weight * x, axis=1) + bias
        return y.astype(np.float64)

    def _map_jac_to_compact(self, j_full: np.ndarray) -> np.ndarray:
        idx = self.mapper["idx"]
        weight = self.mapper["weight"]
        out_dim = idx.shape[0]
        j_compact = np.zeros((out_dim, j_full.shape[1]), dtype=np.float64)
        for i in range(weight.shape[1]):
            j_compact += weight[:, i:i + 1] * j_full[idx[:, i], :]
        return j_compact
