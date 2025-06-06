import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

IMAGE_DIR = "task2/images_nonlinear"
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    print(f"Directory '{IMAGE_DIR}' created.")

try:
    df = pd.read_csv("Dataset/processed_data.csv")
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'Dataset/processed_data.csv' not found.")
    exit()

df["Survey date"] = pd.to_datetime(df["Survey date"])
df = df.sort_values("Survey date").reset_index(drop=True)

TARGET = "Purpose of visit to CITY"
FEATURES = [col for col in df.columns if col not in [TARGET, "Survey date"]]

X = df[FEATURES]
y = df[TARGET]

# Fill missing values
for col in X.select_dtypes(include=np.number).columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=["object", "category"]).columns:
    if X[col].isnull().any():
        X[col] = X[col].fillna("MISSING")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = le.classes_

print(f"Number of features: {len(FEATURES)}")
print(f"Target variable classes: {target_names}")
print("-" * 30)
print("Generating and saving data exploration plot...")
plt.figure(figsize=(10, 6))
sns.countplot(y=y, order=y.value_counts().index, palette="viridis")
plt.title("Distribution of Target Variable (Purpose of visit to CITY)")
plt.xlabel("Count")
plt.ylabel("Purpose of Visit")
plt.tight_layout()
plt.savefig(os.path.join(IMAGE_DIR, "01_target_distribution.png"))
plt.close()
print(
    f"Figure 1 - Target distribution saved to '{IMAGE_DIR}/01_target_distribution.png'"
)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Baseline preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough",
)

# PCA preprocessor
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler()), ("pca", PCA(n_components=10))]
)
preprocessor_pca = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

tscv = TimeSeriesSplit(n_splits=5)


def evaluate_model(model_pipeline, X, y_encoded, tscv):
    accuracy_scores, macro_f1_scores, weighted_f1_scores = [], [], []
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
        print(
            f"Fold {fold + 1}: Accuracy={acc:.4f}, Macro F1={macro_f1:.4f}, Weighted F1={weighted_f1:.4f}"
        )
    return accuracy_scores, macro_f1_scores, weighted_f1_scores


# Baseline Model
print("\nEvaluating Baseline Model...")
model_pipeline_baseline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LGBMClassifier(random_state=42)),
    ]
)
accuracy_baseline, macro_f1_baseline, weighted_f1_baseline = evaluate_model(
    model_pipeline_baseline, X, y_encoded, tscv
)

# PCA Model
print("\nEvaluating PCA Model...")
model_pipeline_pca = Pipeline(
    steps=[
        ("preprocessor", preprocessor_pca),
        ("classifier", LGBMClassifier(random_state=42)),
    ]
)
accuracy_pca, macro_f1_pca, weighted_f1_pca = evaluate_model(
    model_pipeline_pca, X, y_encoded, tscv
)

# Feature Selection Model
print("\nEvaluating Feature Selection Model...")
model_for_selection = LGBMClassifier(random_state=42)
model_for_selection.fit(preprocessor.fit_transform(X), y_encoded)
selector = SelectFromModel(model_for_selection, prefit=True, max_features=50)
model_pipeline_fs = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("classifier", LGBMClassifier(random_state=42)),
    ]
)
accuracy_fs, macro_f1_fs, weighted_f1_fs = evaluate_model(
    model_pipeline_fs, X, y_encoded, tscv
)

print("\nGenerating and saving performance comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(3)
width = 0.2
ax.bar(
    x - width,
    [
        np.mean(accuracy_baseline),
        np.mean(macro_f1_baseline),
        np.mean(weighted_f1_baseline),
    ],
    width,
    label="Baseline",
)
ax.bar(
    x,
    [np.mean(accuracy_pca), np.mean(macro_f1_pca), np.mean(weighted_f1_pca)],
    width,
    label="PCA",
)
ax.bar(
    x + width,
    [np.mean(accuracy_fs), np.mean(macro_f1_fs), np.mean(weighted_f1_fs)],
    width,
    label="Feature Selection",
)
ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Macro F1", "Weighted F1"])
ax.set_title("Comparison of Feature Learning Techniques")
ax.legend()
plt.savefig(os.path.join(IMAGE_DIR, "05_feature_learning_comparison.png"))
plt.close()

print("\nGenerating and saving PCA explained variance plot...")
pca = PCA().fit(StandardScaler().fit_transform(X[numeric_features]))
explained_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(explained_variance)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance")
plt.savefig(os.path.join(IMAGE_DIR, "06_pca_explained_variance.png"))
plt.close()

print("\nAll visualization plots have been successfully saved.")
