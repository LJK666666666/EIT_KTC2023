import os
import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_scores():
    """
    汇总所有level的评分结果，计算总分和统计信息
    """
    results_dir = Path(__file__).parent / 'results'
    
    # 检查结果目录是否存在
    if not results_dir.exists():
        print(f"错误: 结果目录不存在: {results_dir}")
        return
    
    # 查找所有level目录
    level_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('level')]
    
    if not level_dirs:
        print("未找到任何level目录")
        return
    
    all_scores = []
    level_summaries = []
    
    print("正在读取各level评分数据...")
    
    for level_dir in sorted(level_dirs):
        level_name = level_dir.name
        csv_file = level_dir / 'scoring_results.csv'
        
        if not csv_file.exists():
            print(f"警告: {level_name} 目录中没有找到 scoring_results.csv")
            continue
        
        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)
            
            # 添加level列
            df['level'] = level_name
            
            # 计算该level的统计信息
            level_scores = df['score'].values
            level_avg = np.mean(level_scores)
            level_std = np.std(level_scores)
            level_min = np.min(level_scores)
            level_max = np.max(level_scores)
            
            level_summary = {
                'level': level_name,
                'file_count': len(df),
                'average_score': level_avg,
                'std_score': level_std,
                'min_score': level_min,
                'max_score': level_max,
                'total_score': np.sum(level_scores)
            }
            
            all_scores.append(df)
            level_summaries.append(level_summary)
            
            print(f"{level_name}: 文件数={len(df)}, 平均分={level_avg:.4f}, 标准差={level_std:.4f}")
            
        except Exception as e:
            print(f"读取 {csv_file} 时出错: {e}")
            continue
    
    if not all_scores:
        print("没有找到有效的评分数据")
        return
    
    # 合并所有数据
    combined_df = pd.concat(all_scores, ignore_index=True)
    
    # 创建汇总统计
    summary_df = pd.DataFrame(level_summaries)
    
    # 计算总体统计
    total_files = len(combined_df)
    overall_avg = np.mean(combined_df['score'])
    overall_std = np.std(combined_df['score'])
    overall_min = np.min(combined_df['score'])
    overall_max = np.max(combined_df['score'])
    overall_total = np.sum(combined_df['score'])
    
    # 添加总体统计行
    overall_summary = pd.DataFrame([{
        'level': 'Overall',
        'file_count': total_files,
        'average_score': overall_avg,
        'std_score': overall_std,
        'min_score': overall_min,
        'max_score': overall_max,
        'total_score': overall_total
    }])
    
    summary_df = pd.concat([summary_df, overall_summary], ignore_index=True)
    
    # 保存结果
    output_dir = results_dir
    
    # 保存详细数据
    detailed_csv_path = output_dir / 'all_scores_detailed.csv'
    combined_df.to_csv(detailed_csv_path, index=False, encoding='utf-8')
    
    # 保存汇总统计
    summary_csv_path = output_dir / 'scores_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    
    print(f"\n=== 评分汇总完成 ===")
    print(f"总文件数: {total_files}")
    print(f"总体平均分: {overall_avg:.4f}")
    print(f"总体标准差: {overall_std:.4f}")
    print(f"总分: {overall_total:.4f}")
    print(f"详细数据保存至: {detailed_csv_path}")
    print(f"汇总统计保存至: {summary_csv_path}")
    
    # 打印详细统计表格
    print("\n=== 各Level统计详情 ===")
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    return combined_df, summary_df

if __name__ == '__main__':
    aggregate_scores()