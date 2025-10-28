import argparse
import os
import re
import sys
import csv
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio
import importlib
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for script usage
import matplotlib.pyplot as plt

# Ensure project root is on sys.path so we can import Codes_Python modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CODE_DIR = ROOT / 'Codes_Python'
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# Dynamically import KTCScoring to avoid static import path issues (Codes_Python for segmentation)
try:
    KTCScoring = importlib.import_module('Codes_Python.KTCScoring')
except Exception:
    try:
        KTCScoring = importlib.import_module('KTCScoring')
    except Exception as e:
        raise ImportError('Could not import KTCScoring. Please run from the project root so Codes_Python is on sys.path.') from e

# Load scoring module specifically from programs/KTC2023-CUQI9/KTCScoring.py
CUQI9_DIR = ROOT / 'programs' / 'KTC2023-CUQI9'
CUQI9_SCORING_PATH = CUQI9_DIR / 'KTCScoring.py'
if CUQI9_SCORING_PATH.exists():
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('cuqi9_scoring', str(CUQI9_SCORING_PATH))
        cuqi9_scoring = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(cuqi9_scoring)
    except Exception as e:
        cuqi9_scoring = None
else:
    cuqi9_scoring = None


def load_array_from_file(path: Path) -> Optional[np.ndarray]:
    """Load an array from a .mat or .npz file.

    Tries common keys in order of likelihood.
    Returns None if no suitable array is found.
    """
    if path.suffix.lower() == '.mat':
        try:
            mdic = sio.loadmat(path)
        except Exception:
            return None
        # Try common keys
        for key in ['reconstruction', 'deltareco_pixgrid', 'reco', 'image', 'img']:
            if key in mdic:
                arr = mdic[key]
                # MATLAB often stores arrays with extra dimensions
                arr = np.asarray(arr)
                # squeeze but keep 2D if possible
                arr = np.squeeze(arr)
                if arr.ndim == 2:
                    return arr
                # If 3D with last dim = 1
                if arr.ndim == 3 and arr.shape[-1] == 1:
                    return arr[..., 0]
        return None

    if path.suffix.lower() == '.npz':
        try:
            with np.load(path) as npz:
                for key in ['deltareco_pixgrid', 'reconstruction', 'reco', 'image', 'img']:
                    if key in npz:
                        arr = np.asarray(npz[key])
                        arr = np.squeeze(arr)
                        if arr.ndim == 2:
                            return arr
                        if arr.ndim == 3 and arr.shape[-1] == 1:
                            return arr[..., 0]
        except Exception:
            return None

    return None


def is_segmented(arr: np.ndarray) -> bool:
    arr_nanless = arr[np.isfinite(arr)]
    if arr_nanless.size == 0:
        return False
    uniq = np.unique(arr_nanless)
    return (uniq.size <= 10) and np.all(np.mod(uniq, 1) == 0)


def pick_cmap(arr: np.ndarray) -> str:
    """Choose colormap based on value distribution.
    - If array looks like segmented (few unique integers), use 'gray'.
    - Otherwise use 'viridis'.
    """
    if is_segmented(arr):
        return 'gray'
    return 'viridis'


def save_image(arr: np.ndarray, out_path: Path) -> None:
    # Ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = pick_cmap(arr)
    fig, ax = plt.subplots()
    im = ax.imshow(arr, cmap=cmap)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    # colorbar without title
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(str(out_path), bbox_inches='tight', dpi=200)
    plt.close(fig)


def save_comparison(gt: np.ndarray, arr: np.ndarray, out_path: Path, cmap_arr: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im_l = axes[0].imshow(gt, cmap='gray')
    axes[0].axis('image')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    plt.colorbar(im_l, ax=axes[0], fraction=0.046, pad=0.04)
    im_r = axes[1].imshow(arr, cmap=cmap_arr)
    axes[1].axis('image')
    axes[1].set_xticks([]); axes[1].set_yticks([])
    plt.colorbar(im_r, ax=axes[1], fraction=0.046, pad=0.04)
    fig.savefig(str(out_path), bbox_inches='tight', dpi=200)
    plt.close(fig)


def sanitize_name(name: str) -> str:
    # Keep base numeric names or alphanum with underscores
    base = re.sub(r"[^0-9A-Za-z_\-]", "_", name)
    return base


def parse_index_from_name(name: str) -> Optional[int]:
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def load_groundtruth(level: int, idx: int) -> Optional[np.ndarray]:
    gt_dir = ROOT / 'EvaluationData' / 'GroundTruths' / f'level_{level}'
    mat_path = gt_dir / f"{idx}_true.mat"
    png_path = gt_dir / f"{idx}_true.png"
    if mat_path.exists():
        try:
            mdic = sio.loadmat(str(mat_path))
            for key in ['truth', 'groundtruth', 'gt', 'image']:
                if key in mdic:
                    return np.squeeze(mdic[key])
        except Exception:
            pass
    if png_path.exists():
        try:
            img = plt.imread(str(png_path))
            if img.ndim == 3:
                img = img[..., 0]
            return np.squeeze(img)
        except Exception:
            pass
    return None


def convert_folder(input_root: Path, output_root: Path, level: int) -> Tuple[int, int]:
    """Convert all .mat/.npz files under input_root to images under output_root.
    Returns the number of images written.
    """
    count = 0
    scored = 0
    # CSV for scores
    csv_path = output_root / 'scoring_results.csv'
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'file', 'level', 'score'])
    for path in input_root.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ('.mat', '.npz'):
            continue
        arr = load_array_from_file(path)
        if arr is None or arr.ndim != 2:
            continue
        rel = path.relative_to(input_root)
        out_dir = output_root / rel.parent
        stem = sanitize_name(path.stem)
        idx = parse_index_from_name(stem)
        out_path = out_dir / f"{stem}.png"
        seg_path = out_dir / f"{stem}_seg.png"
        cmp_reco_path = out_dir / f"{stem}_compare_reco.png"
        cmp_seg_path = out_dir / f"{stem}_compare_seg.png"
        try:
            # Save raw (continuous or segmented) image
            save_image(arr, out_path)

            # Save segmented image: if already segmented, reuse; otherwise segment using KTCScoring
            if is_segmented(arr):
                seg_arr = arr
            else:
                # Use the same parameters as in the main reconstruction script
                seg_arr = KTCScoring.cv_NLOpt(arr, log_par=1.5, linear_par=1, exp_par=0)
            save_image(seg_arr, seg_path)
            count += 1

            # If groundtruth available, save comparison figures and compute score
            if idx is not None:
                gt = load_groundtruth(level, idx)
                if gt is not None:
                    save_comparison(gt, arr, cmp_reco_path, pick_cmap(arr))
                    save_comparison(gt, seg_arr, cmp_seg_path, 'gray')
                    # Compute score using CUQI9 scoring if available, else fallback to Codes_Python one (same signature)
                    try:
                        scorer = (cuqi9_scoring.scoringFunction if cuqi9_scoring is not None else KTCScoring.scoringFunction)
                        score_val = float(scorer(gt, seg_arr))
                        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([idx, rel.as_posix(), level, score_val])
                        scored += 1
                    except Exception:
                        pass
        except Exception:
            # Skip problematic files silently per spec (no alternatives)
            continue
    return count, scored


def parse_args():
    p = argparse.ArgumentParser(description='Convert EIT reconstruction results to images and scores (PNG + CSV).')
    p.add_argument('--input', '-i', type=str, default=str(Path('results') / 'level1'),
                   help='Input folder containing result files (default: results/level1)')
    p.add_argument('--output', '-o', type=str, default=str(Path('results') / 'level1'),
                   help='Output folder to store images (default: results/level1)')
    p.add_argument('--level', '-l', type=int, default=1, help='Level number to locate ground truths (default: 1)')
    return p.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    n, m = convert_folder(input_root, output_root, args.level)
    print(f"Saved {n} image(s), scored {m} case(s) to {output_root}")


if __name__ == '__main__':
    main()
