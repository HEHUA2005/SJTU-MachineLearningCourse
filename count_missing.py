import pandas as pd

years = range(2015, 2020)
excel_files = [f"Dataset/{year}.xlsx" for year in years]

all_data_frames = []
loaded_files = []
not_found_files = []

print("尝试加载以下文件:")
for file in excel_files:
    print(f"- {file}")
    try:
        df = pd.read_excel(file)
        all_data_frames.append(df)
        loaded_files.append(file)
    except FileNotFoundError:
        print(f"警告: 文件 {file} 未找到，将被跳过。")
        not_found_files.append(file)
    except Exception as e:
        print(f"加载文件 {file} 时出错: {e}")
        not_found_files.append(file)


# 合并所有数据到一个DataFrame
combined_df = pd.concat(all_data_frames, ignore_index=True)
print(f"\n成功加载并合并了 {len(loaded_files)} 个文件: {', '.join(loaded_files)}")

print(f"\n合并后的数据形状: {combined_df.shape}")

# 1. 检查每列的缺失值数量
missing_values_count = combined_df.isnull().sum()

# 2. 计算每列的缺失值百分比
total_rows = len(combined_df)
missing_values_percentage = (missing_values_count / total_rows) * 100

# 3. 创建一个包含缺失值数量和百分比的DataFrame，并排序
missing_info_df = pd.DataFrame(
    {
        "列名": combined_df.columns,
        "缺失数量": missing_values_count,
        "缺失百分比": missing_values_percentage,
    }
)

# 只显示包含缺失值的列，并按缺失百分比降序排列
missing_info_df_filtered = missing_info_df[missing_info_df["缺失数量"] > 0].sort_values(
    by="缺失百分比", ascending=False
)

print("\n--- 存在缺失值的列的统计信息 (按缺失百分比降序) ---")
if missing_info_df_filtered.empty:
    print("太棒了！数据集中没有发现任何缺失值。")
else:
    print(missing_info_df_filtered)

# 4. 额外检查：数据类型
print("\n--- 各列数据类型 ---")
print(combined_df.dtypes)

# 5. 目标变量检查
# 假设目标变量列名与您描述的一致
regression_target = "Number of nights in CITY"
classification_target = "Purpose of visit to CITY"

if regression_target in combined_df.columns:
    print(f"\n--- 回归目标 '{regression_target}' 的缺失值检查 ---")
    print(f"缺失数量: {combined_df[regression_target].isnull().sum()}")
    if combined_df[regression_target].isnull().sum() > 0:
        print(
            f"警告: 回归目标 '{regression_target}' 中存在缺失值，这通常需要处理（例如删除这些行）。"
        )
else:
    print(f"警告: 回归目标列 '{regression_target}' 在数据中未找到。")

if classification_target in combined_df.columns:
    print(f"\n--- 分类目标 '{classification_target}' 的缺失值检查 ---")
    print(f"缺失数量: {combined_df[classification_target].isnull().sum()}")
    if combined_df[classification_target].isnull().sum() > 0:
        print(
            f"警告: 分类目标 '{classification_target}' 中存在缺失值，这通常需要处理（例如删除这些行）。"
        )
else:
    print(f"警告: 分类目标列 '{classification_target}' 在数据中未找到。")

# 6. 对于有大量可能值且看起来像分类的数值列，检查其唯一值数量
print("\n--- 检查数值型特征的唯一值数量 (可能帮助识别伪数值型分类特征) ---")
numerical_cols = combined_df.select_dtypes(include=["number"]).columns
for col in numerical_cols:
    unique_count = combined_df[col].nunique()
    if unique_count < 20:  # 阈值可以调整，小于20个唯一值的数值列可能是分类的
        print(
            f"列 '{col}' (数值型) 有 {unique_count} 个唯一值: {combined_df[col].unique()[:5]}..."
        )  # 显示前5个唯一值
    elif (
        unique_count > total_rows * 0.9
    ):  # 如果唯一值数量接近总行数，可能是ID或非常稀疏的特征
        print(
            f"列 '{col}' (数值型) 有 {unique_count} 个唯一值，接近总行数，可能是ID或稀疏特征。"
        )

# 7. 'Survey date' 列的预处理与检查
date_column = "Survey date"
if date_column in combined_df.columns:
    print(f"\n--- '{date_column}' 列处理与检查 ---")
    # 尝试转换为日期时间格式
    original_date_dtype = combined_df[date_column].dtype
    combined_df[date_column] = pd.to_datetime(combined_df[date_column], errors="coerce")
    # 检查转换后产生的NaT (Not a Time) 值，即转换失败的值
    nat_count = combined_df[date_column].isnull().sum()
    print(f"'{date_column}' 原始数据类型: {original_date_dtype}")
    print(
        f"尝试将 '{date_column}' 转换为日期时间格式后，产生 {nat_count} 个无法解析的日期 (NaT)。"
    )
    if nat_count > 0:
        print(
            f"警告: '{date_column}' 中有无法解析为日期的条目，这将导致在后续按日期筛选时丢失数据或出错。"
        )
    else:
        print(f"'{date_column}' 已成功转换为日期时间格式。")
        # 按年份筛选 (根据作业要求 2015-2019)
        combined_df_filtered_by_date = combined_df[
            (combined_df[date_column].dt.year >= 2015)
            & (combined_df[date_column].dt.year <= 2019)
        ]
        print(f"原始数据行数: {total_rows}")
        print(
            f"按年份 (2015-2019) 筛选后的数据行数: {len(combined_df_filtered_by_date)}"
        )
        if len(combined_df_filtered_by_date) < total_rows:
            print(
                f"注意: 已根据年份筛选数据，移除了 {total_rows - len(combined_df_filtered_by_date)} 行。"
            )
else:
    print(f"警告: 日期列 '{date_column}' 在数据中未找到，无法进行基于日期的筛选。")
