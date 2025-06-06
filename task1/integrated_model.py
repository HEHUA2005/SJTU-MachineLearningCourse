import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import warnings

warnings.filterwarnings("ignore")

IMAGE_DIR = "task1/images"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    print(f"目录 '{IMAGE_DIR}' 已创建。")


def load_and_preprocess_data(filepath="Dataset/processed_data.csv"):
    try:
        df = pd.read_csv(filepath)
        print("数据加载成功。")
    except FileNotFoundError:
        print(f"错误：找不到数据集文件 '{filepath}'。")
        return None, None

    df["Survey date"] = pd.to_datetime(df["Survey date"])
    df = df.sort_values("Survey date").reset_index(drop=True)
    TARGET = "Number of nights in CITY"
    if df[TARGET].isnull().any():
        df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    y = df[TARGET]
    X = df.drop(columns=[TARGET, "Survey date"])

    for col in X.select_dtypes(include=np.number).columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(-999)
    for col in X.select_dtypes(include=["object", "category"]).columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna("MISSING")

    columns_to_treat_as_categorical = [
        "Nationality",
        "Country of residence",
        "Gender",
        "Immigration airport",
        "Purpose of visit to CITY",
        "Travel type",
        "Most desired place",
        "Most satisfied place",
    ]
    for col in columns_to_treat_as_categorical:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X, y


def get_models(X):
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(include=["category"]).columns.tolist()

    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    lgbm_preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
    )

    best_lgbm_params = {
        "learning_rate": 0.028,
        "n_estimators": 500,
        "num_leaves": 46,
        "min_child_samples": 11,
        "subsample": 0.92,
        "colsample_bytree": 0.76,
        "reg_alpha": 0.009,
        "reg_lambda": 0.0007,
        "random_state": 42,
        "verbosity": -1,
    }
    lgbm_pipeline = Pipeline(
        steps=[
            ("preprocessor", lgbm_preprocessor),
            ("regressor", lgb.LGBMRegressor(**best_lgbm_params)),
        ]
    )

    ridge_pipeline = Pipeline(
        steps=[
            ("preprocessor", linear_preprocessor),
            ("regressor", Ridge(alpha=150.0, random_state=42)),
        ]
    )

    estimators = [("lgbm", lgbm_pipeline), ("ridge", ridge_pipeline)]
    stacking_regressor = StackingRegressor(
        estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=5
    )

    models = {
        "Single LGBM": lgbm_pipeline,
        "Single Ridge": ridge_pipeline,
        "Stacking Ensemble": stacking_regressor,
    }
    return models


def run_evaluation(models, X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    for name, model in models.items():
        print(f"--- 正在评估模型: {name} ---")
        mse_scores, mae_scores, r2_scores = [], [], []
        y_test_last, y_pred_last = None, None
        for fold, (train_index, test_index) in enumerate(tscv.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse_scores.append(mean_squared_error(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
            if fold == tscv.n_splits - 1:
                y_test_last, y_pred_last = y_test, y_pred
        results.append(
            {
                "Model": name,
                "Avg MSE": np.mean(mse_scores),
                "Avg MAE": np.mean(mae_scores),
                "Avg R2": np.mean(r2_scores),
                "y_test_last": y_test_last,
                "y_pred_last": y_pred_last,
            }
        )
        print(f"--- {name} 评估完成 ---\n")
    return pd.DataFrame(results)


def visualize_results(results_df):
    print("生成并保存改进后的可视化对比图...")

    # 性能指标对比图 (每个指标一个子图)
    metrics_to_plot = ["Avg MSE", "Avg MAE", "Avg R2"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(22, 7))

    fig.suptitle("Comparison of Model Performance Metrics", fontsize=20, y=1.03)

    for i, metric in enumerate(metrics_to_plot):
        sns.barplot(
            x="Model",
            y=metric,
            data=results_df,
            ax=axes[i],
            palette="viridis",
            width=0.6,
        )

        axes[i].set_title(f"Comparison of {metric}", fontsize=16)
        axes[i].set_ylabel("Average Score", fontsize=12)
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=10, labelsize=12)
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(IMAGE_DIR, "01_model_performance_comparison_subplots.png"))
    plt.close()
    print(
        f"图1 - 模型性能对比图（子图版）已保存至 '{IMAGE_DIR}/01_model_performance_comparison_subplots.png'"
    )

    num_models = len(results_df)
    fig, axes = plt.subplots(1, num_models, figsize=(8 * num_models, 7))
    if num_models == 1:
        axes = [axes]

    for ax, (idx, row) in zip(axes, results_df.iterrows()):
        y_test = row["y_test_last"]
        y_pred = row["y_pred_last"]

        sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.5)
        limits = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
        ax.plot(
            limits,
            limits,
            color="red",
            linestyle="--",
            lw=2,
            label="Perfect Prediction",
        )

        ax.set_title(
            f"'{row['Model']}'\nPredictions vs True Values (Last Fold)", fontsize=14
        )
        ax.set_xlabel("True Values", fontsize=12)
        ax.set_ylabel("Predicted Values", fontsize=12)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "02_prediction_scatter_plots.png"))
    plt.close()
    print(f"图2 - 预测散点图已保存至 '{IMAGE_DIR}/02_prediction_scatter_plots.png'")


if __name__ == "__main__":
    X_main, y_main = load_and_preprocess_data()
    if X_main is not None:
        models = get_models(X_main)
        results = run_evaluation(models, X_main, y_main)

        print("\n--- 所有模型平均性能汇总 ---")
        print(results[["Model", "Avg MSE", "Avg MAE", "Avg R2"]].round(4))

        visualize_results(results)
        print("\n所有分析和可视化已完成。")
