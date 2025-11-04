"""
快速测试训练和评估流程
"""

import sys
import torch

# 测试配置 - 快速训练用于验证流程
test_config = {
    'exp_name': 'test_eit_reconstruction',
    'model_name': 'simple',
    'input_dim': 2356,  # 使用评估数据的测量维度
    'data_path': 'SimData/level_1',
    'num_samples': 200,  # 使用200个样本
    'train_split': 0.8,
    'batch_size': 16,
    'epochs': 5,  # 只训练5个epoch用于测试
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'weight_ce': 1.0,
    'weight_dice': 1.0,
    'patience': 3,
    'early_stopping': 5,
    'save_interval': 2,
    'num_workers': 0
}

print("="*70)
print("Testing Training and Evaluation Pipeline")
print("="*70)

# 导入训练模块
from train import train_model

print("\n[1/2] Training model...")
print("-"*70)

try:
    model, history, save_dir = train_model(test_config)
    print(f"\n✓ Training completed successfully!")
    print(f"  Model saved to: {save_dir}")
except Exception as e:
    print(f"\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试评估
print("\n[2/2] Testing evaluation...")
print("-"*70)

try:
    from evaluate import evaluate_all_levels
    import os

    # 使用最佳模型
    model_path = os.path.join(save_dir, 'best_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    eval_save_dir = os.path.join(save_dir, 'evaluation')

    results = evaluate_all_levels(
        model_path=model_path,
        device=device,
        save_dir=eval_save_dir
    )

    print(f"\n✓ Evaluation completed successfully!")
    print(f"  Results saved to: {eval_save_dir}")

    # 显示结果摘要
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    total_score = 0
    total_samples = 0

    for level_key in sorted(results.keys()):
        level_data = results[level_key]
        level = level_data['level']
        avg_score = level_data['average_score']
        n_samples = len(level_data['scores'])

        print(f"Level {level}: Average Score = {avg_score:.4f} ({n_samples} samples)")

        total_score += avg_score * n_samples
        total_samples += n_samples

    overall_avg = total_score / total_samples if total_samples > 0 else 0
    print("-"*70)
    print(f"Overall Average Score: {overall_avg:.4f}")
    print("="*70)

except Exception as e:
    print(f"\n✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("✓ All tests passed successfully!")
print("="*70)
