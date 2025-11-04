"""
监控SimData生成进度
"""

import os
import time
from datetime import datetime, timedelta


def monitor_progress(target_samples=10000, check_interval=60):
    """
    监控数据生成进度

    Args:
        target_samples: 目标样本数量
        check_interval: 检查间隔（秒）
    """
    gt_path = "SimData/level_1/gt"
    measurements_path = "SimData/level_1/measurements"

    print("="*70)
    print("SimData Generation Progress Monitor")
    print("="*70)
    print(f"Target: {target_samples} samples")
    print(f"Check interval: {check_interval} seconds")
    print("-"*70)

    last_count = 0
    start_time = time.time()

    while True:
        # 统计已生成的文件数
        if os.path.exists(gt_path):
            gt_files = len([f for f in os.listdir(gt_path) if f.endswith('.npy')])
            meas_files = len([f for f in os.listdir(measurements_path) if f.endswith('.npy')])

            current_count = min(gt_files, meas_files)
            elapsed_time = time.time() - start_time

            # 计算进度
            progress = (current_count / target_samples) * 100

            # 计算速度
            if elapsed_time > 0:
                speed = current_count / elapsed_time  # samples per second
                if speed > 0:
                    remaining_samples = target_samples - current_count
                    eta_seconds = remaining_samples / speed
                    eta = timedelta(seconds=int(eta_seconds))
                else:
                    eta = "N/A"
            else:
                speed = 0
                eta = "N/A"

            # 显示进度
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Progress: {current_count}/{target_samples} ({progress:.2f}%) | "
                  f"Speed: {speed*60:.2f} samples/min | ETA: {eta}")

            # 检查是否完成
            if current_count >= target_samples:
                print("-"*70)
                print(f"✓ Generation completed!")
                print(f"Total time: {timedelta(seconds=int(elapsed_time))}")
                break

            last_count = current_count

        time.sleep(check_interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Monitor SimData generation progress')
    parser.add_argument('--target', type=int, default=10000,
                       help='Target number of samples (default: 10000)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds (default: 60)')

    args = parser.parse_args()

    monitor_progress(args.target, args.interval)
