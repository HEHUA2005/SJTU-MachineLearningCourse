import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

IMAGE_DIR = "task2/images_logistic"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    print(f"目录 '{IMAGE_DIR}' 已创建。")

try:
    df = pd.read_csv("Dataset/processed_data.csv")
    print("数据集加载成功。")
except FileNotFoundError:
    print("错误: 'Dataset/processed_data.csv' 文件未找到。请检查文件路径是否正确。")
    exit()

df["Survey date"] = pd.to_datetime(df["Survey date"])
df = df.sort_values("Survey date").reset_index(drop=True)

TARGET = "Purpose of visit to CITY"
FEATURES = [col for col in df.columns if col not in [TARGET, "Survey date"]]

X = df[FEATURES]
y = df[TARGET]


le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = le.classes_

print(f"特征数量: {len(FEATURES)}")
print(f"目标变量类别: {target_names}")
print("-" * 30)

print("生成并保存数据探索图...")

plt.figure(figsize=(10, 6))
sns.countplot(y=y, order=y.value_counts().index, palette="viridis")
plt.title("Distribution of Target Variable (Purpose of visit to CITY)")
plt.xlabel("Count")
plt.ylabel("Purpose of Visit")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "01_target_distribution.png"))
plt.close()
print(f"图1 - 目标变量分布图已保存至 '{IMAGE_DIR}/01_target_distribution.png'")


numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ],
    remainder="passthrough",
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(random_state=42, max_iter=2000, class_weight="balanced"),
        ),
    ]
)

tscv = TimeSeriesSplit(n_splits=5)

accuracy_scores, macro_f1_scores, weighted_f1_scores = [], [], []
fold_labels = []

print("\n开始使用逻辑回归进行时间序列交叉验证...")

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    accuracy_scores.append(acc)
    macro_f1_scores.append(macro_f1)
    weighted_f1_scores.append(weighted_f1)
    fold_labels.append(f"Fold {fold + 1}")

    print(
        f"{fold_labels[-1]}/{tscv.n_splits}: Accuracy={acc:.4f}, Macro F1={macro_f1:.4f}, Weighted F1={weighted_f1:.4f}"
    )

    if fold == tscv.n_splits - 1:
        last_fold_model = model_pipeline
        last_fold_y_test = y_test
        last_fold_y_pred = y_pred

print("-" * 30)
print("交叉验证完成。")
print("\n逻辑回归模型的平均性能:")
print(f"  平均准确率 (Average Accuracy):       {np.mean(accuracy_scores):.4f}")
print(f"  平均宏F1分数 (Average Macro F1):      {np.mean(macro_f1_scores):.4f}")
print(f"  平均加权F1分数 (Average Weighted F1): {np.mean(weighted_f1_scores):.4f}")


print("\n生成并保存模型性能与解释性图...")

metrics_df = pd.DataFrame(
    {
        "Fold": fold_labels,
        "Accuracy": accuracy_scores,
        "Macro F1": macro_f1_scores,
        "Weighted F1": weighted_f1_scores,
    }
)
metrics_df.set_index("Fold").plot(kind="bar", figsize=(12, 7), colormap="coolwarm")
plt.title("Model Performance Across Folds")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "02_performance_across_folds.png"))
plt.close()
print(f"图2 - 各折性能对比图已保存至 '{IMAGE_DIR}/02_performance_across_folds.png'")


cm = confusion_matrix(
    last_fold_y_test, last_fold_y_pred, labels=le.transform(target_names)
)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
)
plt.title("Confusion Matrix for the Last Fold")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig(os.path.join(IMAGE_DIR, "03_confusion_matrix_last_fold.png"))
plt.close()
print(f"图3 - 混淆矩阵图已保存至 '{IMAGE_DIR}/03_confusion_matrix_last_fold.png'")


try:
    ohe_feature_names = (
        last_fold_model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )
    final_feature_names = np.concatenate([numeric_features, ohe_feature_names])

    if len(target_names) == 2:
        coefficients = last_fold_model.named_steps["classifier"].coef_[0]
    else:
        coefficients = np.mean(
            np.abs(last_fold_model.named_steps["classifier"].coef_), axis=0
        )

    importance_df = pd.DataFrame(
        {"Feature": final_feature_names, "Importance": coefficients}
    )
    importance_df["Abs_Importance"] = importance_df["Importance"].abs()
    importance_df = importance_df.sort_values(by="Abs_Importance", ascending=False)

    plt.figure(figsize=(12, 10))
    sns.barplot(
        x="Importance", y="Feature", data=importance_df.head(30), palette="mako"
    )
    plt.title("Top 30 Feature Importances (Coefficients)")
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "04_feature_importance.png"))
    plt.close()
    print(f"图4 - 特征重要性图已保存至 '{IMAGE_DIR}/04_feature_importance.png'")

except Exception as e:
    print(f"\n无法生成特征重要性图: {e}")

print("\n所有可视化图表已成功保存。")
