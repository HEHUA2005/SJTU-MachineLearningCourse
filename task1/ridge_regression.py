import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import warnings

warnings.filterwarnings("ignore")


def load_and_preprocess_data(filepath="Dataset/processed_data.csv"):
    try:
        df = pd.read_csv(filepath)
        print("数据加载成功。")
    except FileNotFoundError:
        print(f"错误：找不到数据集文件 '{filepath}'。请确保文件路径正确。")
        return None, None

    if "Survey date" not in df.columns:
        print("错误：数据集中缺少 'Survey date' 列。")
        return None, None

    df["Survey date"] = pd.to_datetime(df["Survey date"])
    df = df.sort_values("Survey date").reset_index(drop=True)

    TARGET = "Number of nights in CITY"
    if df[TARGET].isnull().any():
        df[TARGET] = df[TARGET].fillna(df[TARGET].median())

    y = df[TARGET]
    X = df.drop(columns=[TARGET, "Survey date"])

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [
        "Nationality",
        "Country of residence",
        "Gender",
        "Immigration airport",
        "Purpose of visit to CITY",
        "Travel type",
        "Most desired place",
        "Most satisfied place",
    ]
    categorical_features = [col for col in categorical_features if col in X.columns]
    numeric_features = [
        col for col in numeric_features if col not in categorical_features
    ]
    X[numeric_features] = X[numeric_features].fillna(-999)
    X[categorical_features] = X[categorical_features].fillna("MISSING")

    return X, y, numeric_features, categorical_features


def evaluate_model(model_pipeline, X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores, mae_scores, r2_scores = [], [], []

    print("\n开始进行5折时间序列交叉验证...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X, y)):
        print(f"--- 第 {fold + 1} 折 ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    print("\n--- 模型平均性能评估 ---")
    print(f"平均均方误差 (MSE): {np.mean(mse_scores):.4f}")
    print(f"平均绝对误差 (MAE): {np.mean(mae_scores):.4f}")
    print(f"平均决定系数 (R2): {np.mean(r2_scores):.4f}")


def run_evaluation_with_best_params(X, y, numeric_features, categorical_features):
    print("\n--- 选择一：使用预设的最佳超参数进行评估 ---")

    best_params_ridge = {"regressor__alpha": 996.3674093019035}

    model_pipeline = build_pipeline(numeric_features, categorical_features)
    model_pipeline.set_params(**best_params_ridge)

    print(f"使用预设参数: {best_params_ridge}")
    evaluate_model(model_pipeline, X, y)


def run_optuna_tuning(X, y, numeric_features, categorical_features):
    print("\n--- 选择二：使用Optuna重新寻找最优超参数 ---")

    def objective(trial):
        alpha = trial.suggest_float("regressor__alpha", 1e-1, 1000.0, log=True)

        model_pipeline = build_pipeline(numeric_features, categorical_features)
        model_pipeline.set_params(regressor__alpha=alpha)

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        for train_index, test_index in tscv.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model_pipeline.fit(X_train, y_train)
            preds = model_pipeline.predict(X_test)
            scores.append(mean_squared_error(y_test, preds))

        return np.mean(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\n--- Optuna 调参完成 ---")
    print("最佳MSE值: ", study.best_value)
    print("最佳超参数: ", study.best_params)


def build_pipeline(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", Ridge(random_state=42))]
    )


if __name__ == "__main__":
    X, y, numeric_features, categorical_features = load_and_preprocess_data()

    if X is not None:
        print("\n请选择要执行的操作:")
        print("1: 使用预设的最佳超参数进行性能评估")
        print("2: 使用Optuna重新寻找最优超参数")

        choice = input("请输入您的选择 (1 或 2): ")

        if choice == "1":
            run_evaluation_with_best_params(
                X, y, numeric_features, categorical_features
            )
        elif choice == "2":
            run_optuna_tuning(X, y, numeric_features, categorical_features)
        else:
            print("输入无效，请输入 1 或 2。")
