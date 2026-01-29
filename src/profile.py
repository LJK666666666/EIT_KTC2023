"""Model profiling for deep learning methods."""
import argparse
import json
import time
from pathlib import Path
import sys
import re

import torch
from torch.profiler import profile, ProfilerActivity
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import ConfigManager
from src.methods import create_method

DEEP_METHODS = ['cnn', 'diffusion', 'deepdbar', 'fno']


def parse_args():
    parser = argparse.ArgumentParser(description='EIT Model Profiling')
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=DEEP_METHODS,
        help='Deep learning method to profile'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: src/configs/{method}_config.yaml)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for profiling'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=2,
        help='Warmup runs before timing'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=5,
        help='Number of timed inference runs'
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        default=None,
        help='Base name (or absolute base path) for results directory (saved as {base}_{num})'
    )
    return parser.parse_args()


def create_result_dir(base_dir: Path) -> Path:
    base_dir = Path(base_dir)
    parent = base_dir.parent
    base_name = base_dir.name
    parent.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{2}})$")
    max_idx = -1
    for item in parent.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))

    result_dir = parent / f"{base_name}_{max_idx + 1:02d}"
    result_dir.mkdir(parents=True, exist_ok=False)
    return result_dir

def build_measurements(config, batch_size, device):
    use_eim = config.get('data', {}).get('use_eim', True)
    if not use_eim:
        raise ValueError('Profiling only supports EIM input (use_eim=true).')

    return torch.randn(batch_size, 1, 16, 16, device=device)


def build_flops_inputs(method_name, config, batch_size, device, measurements):
    if method_name == 'diffusion':
        img_size = config.get('model', {}).get('input_size', 128)
        x = torch.randn(batch_size, 1, img_size, img_size, device=device)
        t = torch.zeros(batch_size, dtype=torch.long, device=device)
        y = measurements
        return (x, t, y)

    if method_name == 'deepdbar':
        size = config.get('model', {}).get('output_size', 64)
        x = torch.randn(batch_size, 1, size, size, device=device)
        return (x,)

    input_size = config.get('model', {}).get('input_size', 16)
    x = torch.randn(batch_size, 1, input_size, input_size, device=device)
    return (x,)


def estimate_flops(model, forward_args, device):
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    with torch.no_grad():
        with profile(activities=activities, record_shapes=True, with_flops=True) as prof:
            model(*forward_args)

    total_flops = 0
    for evt in prof.key_averages():
        flops = getattr(evt, 'flops', None)
        if flops:
            total_flops += flops

    return total_flops


def measure_inference_time(method, measurements, warmup, runs, device):
    method.model.eval()
    with torch.no_grad():
        for _ in tqdm(range(warmup), desc='Warmup', leave=False):
            method.inference(measurements)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        times_ms = []
        for _ in tqdm(range(runs), desc='Timing', leave=False):
            start = time.perf_counter()
            method.inference(measurements)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000)

    avg_ms = sum(times_ms) / len(times_ms)
    return avg_ms, times_ms


def main():
    args = parse_args()

    if args.config is None:
        args.config = f'src/configs/{args.method}_config.yaml'

    config = ConfigManager.load_config(args.config)
    config['device'] = args.device
    config['method_name'] = args.method
    if 'training' in config:
        config['training']['device'] = args.device

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is not available.')

    method = create_method(args.method, config)
    if args.checkpoint:
        method.load_checkpoint(args.checkpoint)

    measurements = build_measurements(config, args.batch_size, device)
    flops_inputs = build_flops_inputs(args.method, config, args.batch_size, device, measurements)

    param_count = sum(p.numel() for p in method.model.parameters())
    param_size_bytes = sum(p.numel() * p.element_size() for p in method.model.parameters())
    param_size_mb = param_size_bytes / (1024 ** 2)

    flops = estimate_flops(method.model, flops_inputs, device)
    gflops = flops / 1e9

    avg_ms, times_ms = measure_inference_time(
        method,
        measurements,
        args.warmup,
        args.runs,
        device
    )

    if args.result_dir is None:
        base_dir = Path('results') / f'profile_{args.method}'
    else:
        req_base = Path(args.result_dir)
        base_dir = req_base if req_base.is_absolute() else Path('results') / args.result_dir

    result_dir = create_result_dir(base_dir)

    results = {
        'method': args.method,
        'device': str(device),
        'config': args.config,
        'checkpoint': args.checkpoint,
        'batch_size': args.batch_size,
        'warmup': args.warmup,
        'runs': args.runs,
        'params': param_count,
        'param_size_mb': param_size_mb,
        'flops': flops,
        'gflops': gflops,
        'avg_inference_ms': avg_ms,
        'times_ms': times_ms,
        'result_dir': str(result_dir)
    }

    result_path = result_dir / 'profile.json'
    result_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

    print(f"Method: {args.method}")
    print(f"Parameters: {param_count:,} ({param_size_mb:.2f} MB)")
    print(f"FLOPs (forward): {gflops:.3f} GFLOPs")
    print(f"Avg inference time: {avg_ms:.2f} ms")
    print(f"Results saved to: {result_dir}")


if __name__ == '__main__':
    main()
