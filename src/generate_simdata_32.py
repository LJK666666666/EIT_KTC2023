"""
生成 32 电极仿真训练数据（KTC 前向求解器）.

输出目录结构:
  {output_dir}/train/*.npz
  {output_dir}/valid/*.npz
  {output_dir}/meta/ref_pattern.npz
  {output_dir}/meta/config.json

每个样本包含:
  xs:     [128, 128] 导电率增量图 (sigma - 1.0)
  ys:     [M, 1]      32电极 deltaU 测量向量
  ys_ref: [M, 1]      参考测量 Uref（便于后续校验）
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from sim_dataset import SimulatedEITDataset


PROTOCOL_OFFICIAL = "official32"
PROTOCOL_CHALLENGE = "challenge76"


def to_sigma_delta_128(label_map_256: np.ndarray) -> np.ndarray:
    """将 256x256 标签图转换为 128x128 导电率增量图."""
    sigma = np.zeros_like(label_map_256, dtype=np.float32)
    sigma[label_map_256 == 0] = 1.0
    sigma[label_map_256 == 1] = 0.5
    sigma[label_map_256 == 2] = 1.5
    delta = sigma - 1.0
    delta_128 = np.array(
        Image.fromarray(delta).resize((128, 128), Image.Resampling.BILINEAR),
        dtype=np.float32
    )
    return delta_128


def save_meta(output_dir: Path, helper: SimulatedEITDataset, config: dict) -> None:
    """保存测量模式元数据."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        meta_dir / "ref_pattern.npz",
        Inj=helper.Inj.astype(np.float64),
        Mpat=helper.Mpat.astype(np.float64),
        vincl=helper.vincl.astype(np.bool_)
    )
    with open(meta_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def select_use_evaluation_pattern(protocol: str) -> bool:
    """
    根据协议选择测量模式:
      - official32: 对齐 Main_SimData.py，使用 setMeasurementPattern(32)
      - challenge76: 使用 ref.mat 的 Injref/Mpat（挑战数据协议）
    """
    if protocol == PROTOCOL_OFFICIAL:
        return False
    if protocol == PROTOCOL_CHALLENGE:
        return True
    raise ValueError(f"Unknown protocol: {protocol}")


def generate_split(
    helper: SimulatedEITDataset,
    split_dir: Path,
    num_samples: int,
    u_ref: np.ndarray,
    add_noise: bool,
    seed: int
) -> None:
    """生成单个 split（train 或 valid）."""
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in tqdm(range(num_samples), desc=f"Generating {split_dir.name}", ncols=100):
        sigma_label = helper._create_phantoms()
        sigma_nodes = helper._interpolate_to_mesh(sigma_label)
        u_meas = helper.solver.SolveForward(sigma_nodes, helper.z)

        if add_noise:
            noise = helper.solver.InvLn @ rng.standard_normal((u_ref.shape[0], 1))
            u_meas_noisy = u_meas + noise
        else:
            u_meas_noisy = u_meas

        delta_u = (u_meas_noisy - u_ref).astype(np.float32)
        xs = to_sigma_delta_128(sigma_label)

        np.savez(
            split_dir / f"{i}.npz",
            xs=xs,
            ys=delta_u,
            ys_ref=u_ref.astype(np.float32)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 32-electrode simulated EIT dataset")
    parser.add_argument("--output_dir", type=str, default="data32", help="输出目录")
    parser.add_argument("--num_train", type=int, default=5000, help="训练样本数")
    parser.add_argument("--num_valid", type=int, default=500, help="验证样本数")
    parser.add_argument("--mesh_name", type=str, default="Mesh_dense.mat", help="网格文件名")
    parser.add_argument("--noise_std1", type=float, default=0.1, help="噪声参数1（百分比）")
    parser.add_argument("--noise_std2", type=float, default=0.0, help="噪声参数2")
    parser.add_argument("--segments", type=int, default=3, choices=[2, 3], help="目标类别数")
    parser.add_argument(
        "--protocol",
        type=str,
        default=PROTOCOL_CHALLENGE,
        choices=[PROTOCOL_OFFICIAL, PROTOCOL_CHALLENGE],
        help="测量协议：challenge76 对齐挑战数据 ref.mat（默认），official32 对齐 Main_SimData"
    )
    parser.add_argument("--no_noise", action="store_true", help="不加噪声")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有输出目录中的同名文件")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    use_evaluation_pattern = select_use_evaluation_pattern(args.protocol)

    if output_dir.exists() and not args.overwrite:
        existing_npz = list(output_dir.glob("train/*.npz")) + list(output_dir.glob("valid/*.npz"))
        if len(existing_npz) > 0:
            raise ValueError(
                f"Output directory already has dataset files: {output_dir}. "
                "Use --overwrite to continue."
            )

    print("=" * 80)
    print("Generate 32-electrode simulated dataset")
    print("=" * 80)
    print(f"Output dir: {output_dir}")
    print(f"Train samples: {args.num_train}")
    print(f"Valid samples: {args.num_valid}")
    print(f"Noise: {'off' if args.no_noise else 'on'} (std1={args.noise_std1}, std2={args.noise_std2})")
    print(f"Protocol: {args.protocol}")
    print(f"Use evaluation pattern: {use_evaluation_pattern}")

    helper = SimulatedEITDataset(
        length=1,
        mesh_name=args.mesh_name,
        noise_std1=args.noise_std1,
        noise_std2=args.noise_std2,
        segments=args.segments,
        use_evaluation_pattern=use_evaluation_pattern
    )

    sigma_ref = np.ones((len(helper.mesh.g), 1), dtype=np.float64)
    u_ref = helper.solver.SolveForward(sigma_ref, helper.z)
    helper.solver.SetInvGamma(args.noise_std1, args.noise_std2, u_ref)

    num_measurements = int(u_ref.shape[0])
    print(f"Measurement dimension: {num_measurements}")
    print(f"Injection patterns: {helper.Inj.shape[1]}")
    print(f"Measurements per injection: {helper.Mpat.shape[1]}")

    start = time.time()
    config = {
        "protocol": args.protocol,
        "num_train": args.num_train,
        "num_valid": args.num_valid,
        "mesh_name": args.mesh_name,
        "noise_std1": args.noise_std1,
        "noise_std2": args.noise_std2,
        "segments": args.segments,
        "seed": args.seed,
        "add_noise": (not args.no_noise),
        "measurement_dim": num_measurements,
        "num_injection_patterns": int(helper.Inj.shape[1]),
        "num_measurements_per_injection": int(helper.Mpat.shape[1])
    }
    save_meta(output_dir, helper, config)
    generate_split(
        helper=helper,
        split_dir=train_dir,
        num_samples=args.num_train,
        u_ref=u_ref,
        add_noise=(not args.no_noise),
        seed=args.seed
    )
    generate_split(
        helper=helper,
        split_dir=valid_dir,
        num_samples=args.num_valid,
        u_ref=u_ref,
        add_noise=(not args.no_noise),
        seed=args.seed + 1
    )

    elapsed = time.time() - start
    total = args.num_train + args.num_valid
    avg = elapsed / max(total, 1)
    print("-" * 80)
    print(f"Done. Total samples: {total}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average per sample: {avg:.4f}s")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
