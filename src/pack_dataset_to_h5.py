"""
将 npz 数据集打包为 h5 或 pkl.

输入目录结构:
  input_dir/train/*.npz
  input_dir/valid/*.npz
  input_dir/meta/ref_pattern.npz (可选)
  input_dir/meta/config.json (可选)
"""

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def sort_npz_files(files):
    def key_fn(p: Path):
        try:
            return int(p.stem)
        except ValueError:
            return p.stem
    return sorted(files, key=key_fn)


def load_split(split_dir: Path):
    files = sort_npz_files(list(split_dir.glob("*.npz")))
    if len(files) == 0:
        return None
    first = np.load(files[0])
    xs_shape = first["xs"].shape
    ys_shape = first["ys"].shape
    ys_ref_shape = first["ys_ref"].shape
    xs = np.zeros((len(files), *xs_shape), dtype=np.float32)
    ys = np.zeros((len(files), *ys_shape), dtype=np.float32)
    ys_ref = np.zeros((len(files), *ys_ref_shape), dtype=np.float32)
    names = []
    for i, f in enumerate(tqdm(files, desc=f"Loading {split_dir.name}", ncols=100)):
        d = np.load(f)
        xs[i] = d["xs"].astype(np.float32)
        ys[i] = d["ys"].astype(np.float32)
        ys_ref[i] = d["ys_ref"].astype(np.float32)
        names.append(f.name)
    return {
        "xs": xs,
        "ys": ys,
        "ys_ref": ys_ref,
        "filenames": names
    }


def read_meta(input_dir: Path):
    meta_dir = input_dir / "meta"
    meta = {}
    pattern_path = meta_dir / "ref_pattern.npz"
    config_path = meta_dir / "config.json"
    if pattern_path.exists():
        d = np.load(pattern_path)
        meta["Inj"] = d["Inj"]
        meta["Mpat"] = d["Mpat"]
        meta["vincl"] = d["vincl"]
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            meta["config"] = json.load(f)
    return meta


def save_h5(output_path: Path, packed: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        for split in ["train", "valid"]:
            if packed.get(split) is None:
                continue
            g = f.create_group(split)
            g.create_dataset("xs", data=packed[split]["xs"], compression="gzip", compression_opts=4)
            g.create_dataset("ys", data=packed[split]["ys"], compression="gzip", compression_opts=4)
            g.create_dataset("ys_ref", data=packed[split]["ys_ref"], compression="gzip", compression_opts=4)
            names = np.array(packed[split]["filenames"], dtype=h5py.string_dtype(encoding="utf-8"))
            g.create_dataset("filenames", data=names)
        meta = packed.get("meta", {})
        if len(meta) > 0:
            mg = f.create_group("meta")
            if "Inj" in meta:
                mg.create_dataset("Inj", data=meta["Inj"])
            if "Mpat" in meta:
                mg.create_dataset("Mpat", data=meta["Mpat"])
            if "vincl" in meta:
                mg.create_dataset("vincl", data=meta["vincl"].astype(np.uint8))
            if "config" in meta:
                mg.attrs["config_json"] = json.dumps(meta["config"], ensure_ascii=False)


def save_pkl(output_path: Path, packed: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(packed, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description="Pack NPZ dataset to H5 or PKL")
    parser.add_argument("--input_dir", type=str, default="data32", help="输入数据目录")
    parser.add_argument("--output_path", type=str, default="", help="输出文件路径")
    parser.add_argument("--format", type=str, default="h5", choices=["h5", "pkl"], help="输出格式")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = input_dir / ("dataset.h5" if args.format == "h5" else "dataset.pkl")

    print("=" * 80)
    print("Pack dataset")
    print("=" * 80)
    print(f"Input dir: {input_dir}")
    print(f"Output: {output_path}")
    print(f"Format: {args.format}")

    train = load_split(input_dir / "train")
    valid = load_split(input_dir / "valid")
    meta = read_meta(input_dir)
    packed = {"train": train, "valid": valid, "meta": meta}

    if args.format == "h5":
        save_h5(output_path, packed)
    else:
        save_pkl(output_path, packed)

    print("-" * 80)
    if train is not None:
        print(f"Train: xs={train['xs'].shape}, ys={train['ys'].shape}")
    if valid is not None:
        print(f"Valid: xs={valid['xs'].shape}, ys={valid['ys'].shape}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
