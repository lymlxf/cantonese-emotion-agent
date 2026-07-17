import pandas as pd
import glob
import os

# 1. 获取所有匹配的 parquet 文件路径
# 假设文件都在当前目录下，且命名格式一致
file_pattern = "data/train-*.parquet"
all_files = sorted(glob.glob(file_pattern))

if not all_files:
    print(f"❌ 未找到匹配 '{file_pattern}' 的文件，请检查路径或文件名。")
else:
    print(f"✅ 发现 {len(all_files)} 个文件，开始处理...")

    # 用于存储所有 label 的列表
    all_labels = []

    # 2. 遍历每个文件
    for i, file_path in enumerate(all_files):
        try:
            # 【关键优化】只读取 'label' 列！忽略巨大的 'audio_file' 列
            # 这能节省 99% 的内存和读取时间
            df_temp = pd.read_parquet(file_path, columns=['label'])
            
            # 将 label 列转换为列表并追加到总列表
            all_labels.extend(df_temp['label'].dropna().tolist())
            
            # 简单的进度打印
            if (i + 1) % 10 == 0 or (i + 1) == len(all_files):
                print(f"   已处理 {i + 1}/{len(all_files)} 个文件...")
                
        except Exception as e:
            print(f"⚠️ 读取文件 {file_path} 时出错: {e}")

    # 3. 将累积的列表转换为 Series 进行统计
    print("\n🔄 正在计算统计信息...")
    series_labels = pd.Series(all_labels)
    
    # 计算统计值
    stats = series_labels.value_counts().reset_index()
    stats.columns = ['label', 'count']
    
    total_count = stats['count'].sum()
    stats['percentage'] = (stats['count'] / total_count * 100).round(4)
    
    # 4. 展示结果
    print("\n" + "="*60)
    print(f"📊 全局标签分布统计 (总样本数: {total_count:,})")
    print("="*60)
    print(stats.to_string(index=False))
    