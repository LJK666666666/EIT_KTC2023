import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def generate_score_report():
    """
    生成详细的评分报告和可视化图表
    """
    results_dir = Path(__file__).parent / 'results'
    
    # 读取汇总数据
    summary_file = results_dir / 'scores_summary.csv'
    detailed_file = results_dir / 'all_scores_detailed.csv'
    
    if not summary_file.exists() or not detailed_file.exists():
        print("请先运行 aggregate_scores.py 生成评分数据")
        return
    
    # 读取数据
    summary_df = pd.read_csv(summary_file)
    detailed_df = pd.read_csv(detailed_file)
    
    # 创建报告目录
    report_dir = results_dir / 'reports'
    report_dir.mkdir(exist_ok=True)
    
    # 生成详细报告
    with open(report_dir / 'score_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== EIT重建评分分析报告 ===\n\n")
        
        # 总体统计
        overall_stats = summary_df[summary_df['level'] == 'Overall'].iloc[0]
        f.write("总体统计:\n")
        f.write(f"总测试文件数: {overall_stats['file_count']}\n")
        f.write(f"总体平均分: {overall_stats['average_score']:.4f}\n")
        f.write(f"总体标准差: {overall_stats['std_score']:.4f}\n")
        f.write(f"最高分: {overall_stats['max_score']:.4f}\n")
        f.write(f"最低分: {overall_stats['min_score']:.4f}\n")
        f.write(f"总分: {overall_stats['total_score']:.4f}\n\n")
        
        # 各Level统计
        f.write("各难度级别统计:\n")
        level_stats = summary_df[summary_df['level'] != 'Overall']
        for _, row in level_stats.iterrows():
            f.write(f"{row['level']}: 平均分={row['average_score']:.4f}, "
                   f"标准差={row['std_score']:.4f}, "
                   f"范围=[{row['min_score']:.4f}, {row['max_score']:.4f}]\n")
    
    # 生成可视化图表
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 各Level平均分柱状图
    level_data = summary_df[summary_df['level'] != 'Overall']
    axes[0, 0].bar(level_data['level'], level_data['average_score'], 
                   color='skyblue', alpha=0.7, edgecolor='navy')
    axes[0, 0].set_title('各难度级别平均分', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('难度级别')
    axes[0, 0].set_ylabel('平均分')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(level_data['average_score']):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. 各Level分数分布箱线图
    level_scores = []
    level_names = []
    for level in sorted(detailed_df['level'].unique()):
        if level != 'Overall':
            level_scores.append(detailed_df[detailed_df['level'] == level]['score'].values)
            level_names.append(level)
    
    axes[0, 1].boxplot(level_scores, labels=level_names)
    axes[0, 1].set_title('各难度级别分数分布', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('难度级别')
    axes[0, 1].set_ylabel('分数')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 分数随难度变化趋势
    axes[1, 0].plot(level_data['level'], level_data['average_score'], 
                   marker='o', linewidth=2, markersize=8, color='red')
    axes[1, 0].fill_between(level_data['level'], 
                           level_data['average_score'] - level_data['std_score'],
                           level_data['average_score'] + level_data['std_score'],
                           alpha=0.3, color='red')
    axes[1, 0].set_title('分数随难度变化趋势', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('难度级别')
    axes[1, 0].set_ylabel('平均分 ± 标准差')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 分数直方图
    axes[1, 1].hist(detailed_df['score'], bins=20, alpha=0.7, color='green', 
                   edgecolor='black')
    axes[1, 1].set_title('所有分数分布直方图', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('分数')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 添加统计信息到直方图
    axes[1, 1].axvline(overall_stats['average_score'], color='red', linestyle='--', 
                      label=f'平均分: {overall_stats["average_score"]:.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(report_dir / 'score_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成HTML报告
    html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>EIT重建评分分析报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .stats { margin: 20px 0; }
        .table { width: 100%; border-collapse: collapse; }
        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: center; }
        .table th { background-color: #f2f2f2; }
        .summary { background-color: #e8f4fd; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>EIT重建评分分析报告</h1>
        <p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>总体统计</h2>
        <p><strong>总测试文件数:</strong> {overall_stats['file_count']}</p>
        <p><strong>总体平均分:</strong> {overall_stats['average_score']:.4f}</p>
        <p><strong>总体标准差:</strong> {overall_stats['std_score']:.4f}</p>
        <p><strong>最高分:</strong> {overall_stats['max_score']:.4f}</p>
        <p><strong>最低分:</strong> {overall_stats['min_score']:.4f}</p>
        <p><strong>总分:</strong> {overall_stats['total_score']:.4f}</p>
    </div>
    
    <div class="stats">
        <h2>各难度级别详细统计</h2>
        {level_stats.to_html(classes='table', index=False, float_format='%.4f')}
    </div>
    
    <div class="stats">
        <h2>可视化分析</h2>
        <img src="score_analysis_plots.png" alt="评分分析图表" style="max-width: 100%; height: auto;">
    </div>
    
    <div class="stats">
        <h2>性能分析</h2>
        <p><strong>最佳表现级别:</strong> {level_stats.loc[level_stats['average_score'].idxmax(), 'level']} (平均分: {level_stats['average_score'].max():.4f})</p>
        <p><strong>最具挑战级别:</strong> {level_stats.loc[level_stats['average_score'].idxmin(), 'level']} (平均分: {level_stats['average_score'].min():.4f})</p>
        <p><strong>稳定性最佳级别:</strong> {level_stats.loc[level_stats['std_score'].idxmin(), 'level']} (标准差: {level_stats['std_score'].min():.4f})</p>
    </div>
</body>
</html>
"""
    
    with open(report_dir / 'score_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"=== 报告生成完成 ===")
    print(f"文本报告: {report_dir / 'score_analysis_report.txt'}")
    print(f"HTML报告: {report_dir / 'score_analysis_report.html'}")
    print(f"可视化图表: {report_dir / 'score_analysis_plots.png'}")
    
    return summary_df, detailed_df

if __name__ == '__main__':
    generate_score_report()