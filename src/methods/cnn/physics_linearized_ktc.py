"""
Linearized KTC physics backend for test-time optimization.

This version uses:
- true KTC forward model
- true KTC Jacobian
- linearization around NN initialization sigma0
- measurement-domain alignment to normalized EIM used by dataloader
"""
from typing import Optional, List, Tuple
from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp
import torch

from .physics_backend import PhysicsBackend


ktc_path = Path(__file__).resolve().parents[2] / "ktc_methods"
if str(ktc_path) not in sys.path:
    sys.path.insert(0, str(ktc_path))

import KTCFwd  # type: ignore
import KTCAux  # type: ignore
from ...sim_dataset import load_mesh


def _valid_eim_columns(inj_idx: int, num_electrodes: int = 16) -> List[int]:
    skip = {(inj_idx - 1) % num_electrodes, inj_idx, (inj_idx + 1) % num_electrodes}
    return [j for j in range(num_electrodes) if j not in skip]


def _build_expand_index() -> List[Tuple[int, int, int]]:
    # (compact_idx, row, col)
    idx_map = []
    compact_idx = 0
    for row in range(16):
        for col in _valid_eim_columns(row, 16):
            idx_map.append((compact_idx, row, col))
            compact_idx += 1
    return idx_map


def _build_virtual_pair_inj32() -> np.ndarray:
    # 16 virtual patterns on 32 physical electrodes.
    inj = np.zeros((32, 16), dtype=np.float64)
    for p in range(16):
        a = (2 * p) % 32
        b = (2 * p + 1) % 32
        c = (2 * p + 2) % 32
        d = (2 * p + 3) % 32
        inj[a, p] = 0.5
        inj[b, p] = 0.5
        inj[c, p] = -0.5
        inj[d, p] = -0.5
    return inj


def _build_pair_transform_31_to_16() -> np.ndarray:
    """
    Convert 31 adjacent physical differences to 16 virtual-pair adjacent differences.
    Missing d31 (V31-V0) is reconstructed via loop constraint:
        d31 = -sum_{k=0..30} d_k
    """
    a = np.zeros((16, 31), dtype=np.float64)
    eye31 = np.eye(31, dtype=np.float64)
    d31_basis = -np.ones((31,), dtype=np.float64)

    def basis(k: int) -> np.ndarray:
        if k < 31:
            return eye31[k]
        return d31_basis

    for j in range(16):
        k0 = (2 * j) % 32
        k1 = (2 * j + 1) % 32
        k2 = (2 * j + 2) % 32
        # D_j = 0.5 * (d_k0 + 2*d_k1 + d_k2)
        a[j] = 0.5 * (basis(k0) + 2.0 * basis(k1) + basis(k2))
    return a


def _build_node_sampling_matrix(node_xy: np.ndarray, h: int, w: int) -> sp.csr_matrix:
    """
    Build sparse linear map W: image_flat -> node_values using bilinear interpolation.
    Domain is fixed to [-0.115, 0.115] x [-0.115, 0.115].
    """
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

        # bilinear weights
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

    mat = sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes, h * w), dtype=np.float64)
    return mat


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

        self.mesh, self.mesh2 = load_mesh("Mesh_sparse.mat")
        self.inj = _build_virtual_pair_inj32()
        _, self.mpat, _ = KTCAux.setMeasurementPattern(32)  # mpat is 32x31
        self.vincl = np.ones((31, self.inj.shape[1]), dtype=bool)
        self.solver = KTCFwd.EITFEM(self.mesh2, self.inj, self.mpat, self.vincl)
        self.z = (1e-6) * np.ones((32, 1), dtype=np.float64)

        self.n_nodes = self.mesh2.g.shape[0]
        self.a_pair = _build_pair_transform_31_to_16()  # 16x31
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
            self._w_img_to_nodes = _build_node_sampling_matrix(self.mesh2.g, h, w)

        sigma0_img = sigma0[0, 0].detach().cpu().numpy().astype(np.float64)
        sigma0_flat = sigma0_img.reshape(-1)
        sigma0_nodes = self._w_img_to_nodes @ sigma0_flat

        # Map network output to physical conductivity (baseline 1 + perturbation)
        sigma0_phys = np.clip(1.0 + sigma0_nodes, 1e-6, None).reshape(-1, 1)
        sigma_ref = np.ones((self.n_nodes, 1), dtype=np.float64)

        d_adj_ref, _ = self._solve_adj_and_jac(sigma_ref, need_jac=False)
        d_adj_0, j_adj_0 = self._solve_adj_and_jac(sigma0_phys, need_jac=True)

        y_compact_ref, _ = self._adj_to_compact_and_jac(d_adj_ref, None)
        y_compact_0, j_compact_mesh = self._adj_to_compact_and_jac(d_adj_0, j_adj_0)

        y_delta_0 = y_compact_0 - y_compact_ref
        if j_compact_mesh is None:
            raise ValueError("Internal error: Jacobian not available.")

        # chain: image -> nodes -> compact measurement
        j_img = j_compact_mesh @ self._w_img_to_nodes  # [208, H*W]

        mean_flat = self.mean_16x13.reshape(-1)
        std_flat = self.std_16x13.reshape(-1)
        scale = 1.0 / (self.voltage * std_flat)

        y0_norm = (y_delta_0 / self.voltage - mean_flat) / std_flat
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

    def _solve_adj_and_jac(self, sigma_nodes: np.ndarray, need_jac: bool):
        self.solver.SolveForward(sigma_nodes, self.z)
        d_adj = np.asarray(self.mpat.T * self.solver.C * self.solver.theta[self.solver.ng2:, :], dtype=np.float64)
        # d_adj shape: [31, 16]

        if not need_jac:
            return d_adj, None

        j_vec = self.solver.Jacobian(sigma_nodes, self.z)  # [31*16, n_nodes], inj-major
        j_adj = j_vec.reshape(16, 31, self.n_nodes)        # [inj, meas31, n_nodes]
        return d_adj, j_adj

    def _adj_to_compact_and_jac(self, d_adj: np.ndarray, j_adj: Optional[np.ndarray]):
        values = []
        j_rows = []

        for inj_idx in range(16):
            # d_adj[:, inj_idx]: 31 adjacent physical differences
            d_pair = self.a_pair @ d_adj[:, inj_idx]  # [16]
            if j_adj is not None:
                j_pair = self.a_pair @ j_adj[inj_idx, :, :]  # [16, n_nodes]
            else:
                j_pair = None

            valid_cols = _valid_eim_columns(inj_idx, 16)
            for col in valid_cols:
                values.append(d_pair[col])
                if j_pair is not None:
                    j_rows.append(j_pair[col])

        y_compact = np.asarray(values, dtype=np.float64)  # [208]
        if j_adj is None:
            return y_compact, None
        j_compact = np.stack(j_rows, axis=0).astype(np.float64)  # [208, n_nodes]
        return y_compact, j_compact

