import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import optuna  # Import Optuna
import warnings

warnings.filterwarnings("ignore")


def main():
    try:
        df = pd.read_csv("Dataset/processed_data.csv")
        print("数据加载成功 (Data loaded successfully).")
    except FileNotFoundError:
        print(
            "错误：找不到数据集文件 'Dataset/processed_data.csv'。请确保文件路径正确。"
        )
        print(
            "Error: Dataset file 'Dataset/processed_data.csv' not found. Please ensure the path is correct."
        )
        return

    if "Survey date" not in df.columns:
        print("错误：数据集中缺少 'Survey date' 列，无法进行时间序列分析。")
        print(
            "Error: 'Survey date' column is missing, cannot perform time series analysis."
        )
        return

    df["Survey date"] = pd.to_datetime(df["Survey date"])
    df = df.sort_values("Survey date").reset_index(drop=True)

    TARGET = "Number of nights in CITY"
    if TARGET not in df.columns:
        print(f"错误：数据集中缺少目标变量 '{TARGET}' 列。")
        print(f"Error: Target variable '{TARGET}' is missing from the dataset.")
        return

    if df[TARGET].isnull().any():
        print(f"警告：目标变量 '{TARGET}' 中存在缺失值。将使用中位数填充。")
        print(
            f"Warning: Missing values found in target '{TARGET}'. Filling with median."
        )
        df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        if df[TARGET].isnull().any():
            print(
                f"警告：转换后目标变量 '{TARGET}' 中仍存在无法解析的数值，使用中位数填充。"
            )
            print(
                f"Warning: After conversion, non-numeric values still exist in '{TARGET}'. Filling with median."
            )
            df[TARGET] = df[TARGET].fillna(df[TARGET].median())

    y = df[TARGET]
    X = df.drop(columns=[TARGET, "Survey date"])

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
    for col in X.columns:
        if col in columns_to_treat_as_categorical:
            X[col] = X[col].astype("category")
        elif X[col].dtype == "object":
            try:
                X[col] = pd.to_numeric(X[col])
            except ValueError:
                X[col] = X[col].astype("category")

    for col in X.columns:
        if X[col].dtype != "category" and X[col].isnull().any():
            X[col] = X[col].fillna(-999)  # Fill numeric NaNs

    print("\n数据预处理完成 (Data preprocessing complete).")

    # --- User Choice ---
    print("\n请选择一个操作模式 (Please select an operating mode):")
    print(
        "1: 使用预定义的最佳超参数进行性能评估 (Evaluate with pre-defined best hyperparameters)"
    )
    print(
        "2: 重新运行Optuna寻找新的最优超参数 (Re-run Optuna to find new optimal hyperparameters)"
    )

    choice = input("请输入您的选择 (1/2): Enter your choice (1/2): ")

    best_params = {}
    if choice == "1":
        print("\n--- 选择 1: 使用预定义的超参数 ---")

        best_params = {
            "learning_rate": 0.028234600635161717,
            "n_estimators": 500,
            "num_leaves": 46,
            "min_child_samples": 11,
            "subsample": 0.920018493747736,
            "colsample_bytree": 0.7676041673251832,
            "reg_alpha": 0.009254735141160568,
            "reg_lambda": 0.0007520390963301944,
            "random_state": 42,  # for reproducibility
            "verbosity": -1,
        }
        print("已加载预定义的超参数 (Loaded pre-defined hyperparameters).")

    elif choice == "2":
        print("\n--- 选择 2: 重新寻找最优超参数 ---")

        def objective(trial):
            params = {
                "objective": "regression_l1",
                "metric": "mae",
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 100),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                "random_state": 42,
                "verbosity": -1,
            }

            model = lgb.LGBMRegressor(**params)

            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            for train_index, test_index in tscv.split(X, y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_test, y_test)],
                    callbacks=[lgb.early_stopping(10, verbose=False)],
                )
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                scores.append(mae)

            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        # Using 50 trials for a quick search. Increase for a more thorough search.
        study.optimize(objective, n_trials=50, show_progress_bar=True)

        print("\nOptuna 调参完成 (Optuna tuning finished).")
        print(f"最佳 MAE 值 (Best MAE value): {study.best_value:.4f}")
        print("找到的最佳超参数 (Best hyperparameters found):")
        print(study.best_params)

        best_params = study.best_params
        best_params["random_state"] = 42
        best_params["verbosity"] = -1

    else:
        print("无效的选择。请输入 1 或 2。(Invalid choice. Please enter 1 or 2.)")
        return

    print("\n--- 开始使用最终超参数进行滚动预测评估 ---")
    print("--- Starting rolling forecast evaluation with final hyperparameters ---")

    final_model = lgb.LGBMRegressor(**best_params)

    tscv = TimeSeriesSplit(n_splits=5)
    mse_scores, mae_scores, r2_scores = [], [], []

    for fold, (train_index, test_index) in enumerate(tscv.split(X, y)):
        print(f"\n--- 第 {fold + 1} 折 (Fold {fold + 1}) ---")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(
            f"训练集大小 (Train size): {len(X_train)}, 测试集大小 (Test size): {len(X_test)}"
        )

        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_scores.append(mse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"平均绝对误差 (MAE): {mae:.4f}")
        print(f"决定系数 (R2 Score): {r2:.4f}")

    print("\n--- 模型平均性能评估 (Average Model Performance) ---")
    print(f"平均均方误差 (Average MSE): {np.mean(mse_scores):.4f}")
    print(f"平均平均绝对误差 (Average MAE): {np.mean(mae_scores):.4f}")
    print(f"平均决定系数 (Average R2 Score): {np.mean(r2_scores):.4f}")


# run as 'PYTHONPATH=. python light_gbm_regression.py'
if __name__ == "__main__":
    main()
