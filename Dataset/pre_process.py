import pandas as pd
import os

years = range(2015, 2020)
excel_files = [f"{year}.xlsx" for year in years]
all_data_frames = []

print("尝试读取以下文件：")
for file_path in excel_files:
    print(file_path)
    if not os.path.exists(file_path):
        print(f"警告：文件未找到 - {file_path}")

for file in excel_files:
    try:
        if os.path.exists(file):
            df_temp = pd.read_excel(file)
            all_data_frames.append(df_temp)
            print(f"成功读取 {file}，形状：{df_temp.shape}")
            if "Survey date" in df_temp.columns:
                print(
                    f"  文件 {file} 中 'Survey date' 列的数据类型：{df_temp['Survey date'].dtype}"
                )
                print(
                    f"  文件 {file} 中 'Survey date' 列的前5个值：\n{df_temp['Survey date'].head()}"
                )
            else:
                print(f"  警告：在文件 {file} 中未找到 'Survey date' 列")
        else:
            print(f"跳过不存在的文件：{file}")
    except Exception as e:
        print(f"读取文件 {file} 时出错：{e}")

if not all_data_frames:
    print("没有加载任何DataFrame。正在退出。")
    exit()

combined_df = pd.concat(all_data_frames, ignore_index=True)
df = combined_df.copy()  # 在副本上操作

print(f"\n连接后的形状：{df.shape}")
if "Survey date" in df.columns:
    print(f"连接后 'Survey date' 的数据类型：{df['Survey date'].dtype}")
    print(
        f"连接后 'Survey date' 中唯一类型的计数：\n{df['Survey date'].apply(type).value_counts()}"
    )
    print(f"连接后（转换前）'Survey date' 的示例值：\n{df['Survey date'].head(10)}")

    # 转换可能是Excel序列日期的数值。
    # 创建一个临时的数值型调查日期序列，对于非数值部分强制转换为错误 (NaN)。
    numeric_survey_dates = pd.to_numeric(df["Survey date"], errors="coerce")

    # 为典型的Excel序列日期数值定义一个掩码。
    # Windows版Excel：基准日期 '1899-12-30'。
    # 例如，2015-01-01 是 42005，2020-01-01 是 43831。
    # 如果您的日期超出此常见范围，请调整此范围（例如，25000 到 70000）。
    excel_serial_mask = (
        (numeric_survey_dates > 25000)
        & (numeric_survey_dates < 70000)
        & numeric_survey_dates.notna()
    )

    # 使用正确的Excel日期基准转换这些数值。
    # .loc 确保我们修改的是原始DataFrame 'df'
    df.loc[excel_serial_mask, "Survey date"] = pd.to_datetime(
        numeric_survey_dates[excel_serial_mask], unit="D", origin="1899-12-30"
    )

    # 步骤2：将任何剩余的值（例如，日期字符串如 "1/30/2015" 或已经是datetime对象的值）转换为datetime。
    # 这也将正确处理已从步骤1转换的日期，确保整个列的类型为datetime。
    # errors='coerce' 会将任何剩余无法解析的值转换为 NaT (Not a Time)。
    df["Survey date"] = pd.to_datetime(df["Survey date"], errors="coerce")
    # --- 日期转换结束 ---

    print(f"\n转换后 'Survey date' 的数据类型：{df['Survey date'].dtype}")
    print(
        f"转换后 'Survey date' 中 NaT (Not a Time) 的数量：{df['Survey date'].isnull().sum()}"
    )
    print(f"转换后 'Survey date' 的示例值：\n{df['Survey date'].head(10)}")
else:
    print("错误：在合并后的DataFrame中未找到 'Survey date' 列。")


print(f"\n原始数据框形状（日期转换后）：{df.shape}")
print(f"原始数据中有 {df.isnull().any().sum()} 列包含至少一个缺失值。")

# 删除缺失值超过10%的列
missing_percentage = df.isnull().sum() / len(df)
cols_to_drop = missing_percentage[missing_percentage > 0.1].index
df_after_cols_dropped = df.copy()
if len(cols_to_drop) > 0:
    df_after_cols_dropped.drop(columns=cols_to_drop, inplace=True)
    print(f"\n因缺失值过多 (>10%) 而删除的列 ({len(cols_to_drop)} 列):")
    for col in cols_to_drop:
        print(f"- {col} (缺失率: {missing_percentage[col]:.2%})")
    print(f"\n删除列后的数据框形状: {df_after_cols_dropped.shape}")
else:
    print("\n没有列因缺失值超过10%而被删除。")

# 删除包含任何缺失值的行 (在剩余列中)
rows_before_dropna = len(df_after_cols_dropped)
df_processed = df_after_cols_dropped.dropna()
rows_after_dropna = len(df_processed)

if rows_before_dropna - rows_after_dropna > 0:
    print(
        f"\n删除了 {rows_before_dropna - rows_after_dropna} 行，这些行在剩余列中包含缺失值。"
    )
else:
    print("\n没有行因包含缺失值而被删除（在筛选列之后）。")

print(f"\n最终处理后的数据框形状: {df_processed.shape}")

print("\n处理后数据的前几行:")
print(df_processed.head())

if df_processed.empty:
    print(
        "\n注意：经过预处理后，数据框为空。这可能是因为原始数据量较小，且缺失值较多，"
        "或者日期转换导致许多行被识别为包含NaT并被后续步骤删除。"
    )
# 进行一些One hot处理
# df_processed = pd.get_dummies(
#     df_processed,
#     columns=[
#         "Nationality",
#         "Country of residence",
#         "Gender",
#         "Immigration airport",
#         "Purpose of visit to CITY",
#         "Travel type",
#         "Most desired place",
#          "Most satisfied place"
#     ],
# )
df_processed.to_csv("processed_data.csv", index=False)
print("\n处理后的数据已保存到 processed_data.csv")
