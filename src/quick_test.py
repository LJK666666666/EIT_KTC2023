"""
快速测试 - 1条数据，1个epoch

验证完整流程：数据生成 -> 训练 -> 评估
"""

import sys
import torch
import os
import numpy as np

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*70)
print("QUICK TEST: 1 Sample, 1 Epoch")
print("="*70)

# Step 1: 生成1条测试数据
print("\n[Step 1/3] Generating 1 sample...")
print("-"*70)

from sim_dataset import SimulatedEITDataset

# 清理旧数据
if os.path.exists('SimData/level_1'):
    import shutil
    shutil.rmtree('SimData/level_1')
    print("Cleared old data")

# 生成4条数据（2条训练，2条验证）
dataset_gen = SimulatedEITDataset(length=4, use_evaluation_pattern=True)
os.makedirs('SimData/level_1/gt', exist_ok=True)
os.makedirs('SimData/level_1/measurements', exist_ok=True)

for i in range(4):
    phantom, measurements = dataset_gen[i]
    np.save(f'SimData/level_1/gt/gt_{i:05d}.npy', phantom.numpy())
    np.save(f'SimData/level_1/measurements/u_{i:05d}.npy', measurements.numpy())

print(f"✓ Generated 4 samples")
print(f"  Phantom shape: {phantom.shape}")
print(f"  Measurements shape: {measurements.shape}")

# Step 2: 训练1个epoch
print("\n[Step 2/3] Training for 1 epoch...")
print("-"*70)

from train import train_model

config = {
    'exp_name': 'quick_test',
    'model_name': 'simple',
    'input_dim': 2356,
    'data_path': 'SimData/level_1',
    'num_samples': None,  # 使用所有可用样本
    'train_split': 0.5,  # 2条训练，2条验证
    'batch_size': 2,
    'epochs': 1,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'weight_ce': 1.0,
    'weight_dice': 1.0,
    'patience': 1,
    'early_stopping': 1,
    'save_interval': 1,
    'num_workers': 0
}

try:
    model, history, save_dir = train_model(config)
    print(f"\n✓ Training completed")
    print(f"  Save dir: {save_dir}")
    print(f"  Train loss: {history['train_loss'][0]:.4f}")
    print(f"  Val loss: {history['val_loss'][0]:.4f}")
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: 评估
print("\n[Step 3/3] Evaluating on test data...")
print("-"*70)

from evaluate import evaluate_all_levels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(save_dir, 'best_model.pth')
eval_save_dir = os.path.join(save_dir, 'evaluation')

try:
    results = evaluate_all_levels(
        model_path=model_path,
        device=device,
        save_dir=eval_save_dir
    )

    print(f"\n✓ Evaluation completed")

    # 显示结果
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    total_score = 0
    total_samples = 0

    for level_key in sorted(results.keys()):
        level_data = results[level_key]
        level = level_data['level']
        avg_score = level_data['average_score']
        n_samples = len(level_data['scores'])

        print(f"Level {level}: {avg_score:.4f} ({n_samples} samples)")

        total_score += avg_score * n_samples
        total_samples += n_samples

    overall_avg = total_score / total_samples if total_samples > 0 else 0
    print("-"*70)
    print(f"Overall: {overall_avg:.4f} ({total_samples} samples)")
    print("="*70)

except Exception as e:
    print(f"\n✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓✓✓ ALL TESTS PASSED ✓✓✓")
print("="*70)
print(f"\nResults saved to: {save_dir}")
