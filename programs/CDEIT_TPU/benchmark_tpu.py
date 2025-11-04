#!/usr/bin/env python3
"""
TPU v5e-1 vs v5e-8 性能基准测试
"""
import time
import torch
import argparse

# 检测 TPU
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_TPU = True
    device = torch_xla.device()
except ImportError:
    HAS_TPU = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def benchmark_training(model, batch_size, num_iterations=100):
    """基准测试训练速度"""
    print(f"\n{'='*60}")
    print(f"训练基准测试 - Batch Size: {batch_size}")
    print(f"{'='*60}")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 预热
    print("预热中...")
    for _ in range(10):
        x = torch.randn(batch_size, 1, 128, 128, device=device)
        y = model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if HAS_TPU:
            xm.mark_step()

    # 正式测试
    print(f"运行 {num_iterations} 次迭代...")
    start_time = time.time()

    for i in range(num_iterations):
        x = torch.randn(batch_size, 1, 128, 128, device=device)
        y = model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if HAS_TPU:
            xm.mark_step()

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed
            print(f"  [{i+1}/{num_iterations}] 速度: {speed:.2f} it/s")

    total_time = time.time() - start_time
    avg_speed = num_iterations / total_time
    samples_per_sec = avg_speed * batch_size

    print(f"\n结果:")
    print(f"  总时间: {total_time:.2f}s")
    print(f"  平均速度: {avg_speed:.2f} iterations/s")
    print(f"  吞吐量: {samples_per_sec:.2f} samples/s")

    return samples_per_sec

def benchmark_inference(model, batch_size, num_iterations=100):
    """基准测试推理速度"""
    print(f"\n{'='*60}")
    print(f"推理基准测试 - Batch Size: {batch_size}")
    print(f"{'='*60}")

    model.eval()

    # 预热
    print("预热中...")
    with torch.no_grad():
        for _ in range(10):
            x = torch.randn(batch_size, 1, 128, 128, device=device)
            y = model(x)

            if HAS_TPU:
                xm.mark_step()

    # 正式测试
    print(f"运行 {num_iterations} 次迭代...")
    start_time = time.time()

    with torch.no_grad():
        for i in range(num_iterations):
            x = torch.randn(batch_size, 1, 128, 128, device=device)
            y = model(x)

            if HAS_TPU:
                xm.mark_step()

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                print(f"  [{i+1}/{num_iterations}] 速度: {speed:.2f} it/s")

    total_time = time.time() - start_time
    avg_speed = num_iterations / total_time
    samples_per_sec = avg_speed * batch_size

    print(f"\n结果:")
    print(f"  总时间: {total_time:.2f}s")
    print(f"  平均速度: {avg_speed:.2f} iterations/s")
    print(f"  吞吐量: {samples_per_sec:.2f} samples/s")

    return samples_per_sec

def create_simple_model():
    """创建一个简单的 CNN 模型"""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 1, 3, padding=1),
    ).to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='要测试的批大小列表')
    parser.add_argument('--iterations', type=int, default=50,
                        help='每个批大小的迭代次数')
    args = parser.parse_args()

    print(f"设备: {device}")
    if HAS_TPU:
        print("TPU 模式")
    elif torch.cuda.is_available():
        print(f"GPU 模式: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU 模式")

    model = create_simple_model()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试不同批大小
    train_results = {}
    inference_results = {}

    for batch_size in args.batch_sizes:
        try:
            # 训练测试
            train_throughput = benchmark_training(model, batch_size, args.iterations)
            train_results[batch_size] = train_throughput

            # 推理测试
            inference_throughput = benchmark_inference(model, batch_size, args.iterations)
            inference_results[batch_size] = inference_throughput

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n❌ Batch Size {batch_size} 内存溢出！")
                break
            else:
                raise

    # 打印总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    print("\n训练吞吐量:")
    for bs, throughput in train_results.items():
        print(f"  Batch Size {bs}: {throughput:.2f} samples/s")

    print("\n推理吞吐量:")
    for bs, throughput in inference_results.items():
        print(f"  Batch Size {bs}: {throughput:.2f} samples/s")

    # 计算最佳批大小
    if train_results:
        best_train_bs = max(train_results, key=train_results.get)
        print(f"\n最佳训练批大小: {best_train_bs} ({train_results[best_train_bs]:.2f} samples/s)")

    if inference_results:
        best_inference_bs = max(inference_results, key=inference_results.get)
        print(f"最佳推理批大小: {best_inference_bs} ({inference_results[best_inference_bs]:.2f} samples/s)")

if __name__ == "__main__":
    main()
