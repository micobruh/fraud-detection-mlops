import nbformat as nbf


nb = nbf.v4.new_notebook()
cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        """# IEEE Fraud Detection EDA

This notebook explores the interim IEEE fraud-detection data stored as parquet files in `data/interim/ieee-fraud-detection`.

The goal is to understand the structure of the transaction and identity tables, identify data quality and predictive patterns, and build a lightweight baseline for predicting `isFraud`.

We focus on:

- target imbalance and dataset coverage
- missingness and schema consistency
- fraud patterns in key transaction, categorical, and identity features
- time-based transaction behavior from `TransactionDT`
- a simple, chronological validation baseline for fraud prediction
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.figsize"] = (12, 6)

DATA_DIR = Path("../data/interim/ieee-fraud-detection")
RANDOM_STATE = 42


def top_missing(df, name, n=20):
    missing = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(n)
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    missing["dataset"] = name
    return missing


def fraud_rate_table(df, feature, top_n=10):
    temp = df[[feature, "isFraud"]].copy()
    counts = temp[feature].value_counts(dropna=False)
    keep = counts.head(top_n).index
    temp[feature] = temp[feature].where(temp[feature].isin(keep), "Other")
    summary = (
        temp.groupby(feature, dropna=False)
        .agg(transactions=("isFraud", "size"), fraud_rate=("isFraud", "mean"))
        .sort_values("fraud_rate", ascending=False)
    )
    return summary


def plot_fraud_rate(summary, title, xlabel):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ordered = summary.sort_values("fraud_rate", ascending=False)
    sns.barplot(
        data=ordered.reset_index(),
        x=ordered.index.name or "index",
        y="fraud_rate",
        color="#d95f02",
        ax=ax1,
    )
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Fraud rate")
    ax1.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """train_transaction = pd.read_parquet(DATA_DIR / "train_transaction.parquet")
train_identity = pd.read_parquet(DATA_DIR / "train_identity.parquet")
test_transaction = pd.read_parquet(DATA_DIR / "test_transaction.parquet")
test_identity = pd.read_parquet(DATA_DIR / "test_identity.parquet")

overview = pd.DataFrame(
    {
        "rows": [
            len(train_transaction),
            len(train_identity),
            len(test_transaction),
            len(test_identity),
        ],
        "columns": [
            train_transaction.shape[1],
            train_identity.shape[1],
            test_transaction.shape[1],
            test_identity.shape[1],
        ],
    },
    index=[
        "train_transaction",
        "train_identity",
        "test_transaction",
        "test_identity",
    ],
)

display(overview)

schema_check = pd.DataFrame(
    {
        "train_identity_columns": train_identity.columns.astype(str),
        "test_identity_columns": test_identity.columns.astype(str),
    }
)
display(schema_check.head(12))
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """target_rate = train_transaction["isFraud"].mean()
target_counts = train_transaction["isFraud"].value_counts().sort_index()
identity_coverage = train_identity["TransactionID"].nunique() / train_transaction["TransactionID"].nunique()

print(f"Fraud rate in training transactions: {target_rate:.2%}")
print(f"Transactions with identity information: {identity_coverage:.2%}")

summary_stats = pd.DataFrame(
    {
        "metric": [
            "train fraud rate",
            "non-fraud transactions",
            "fraud transactions",
            "identity coverage on train",
            "train_transaction memory (MB)",
            "train_identity memory (MB)",
        ],
        "value": [
            target_rate,
            int(target_counts.loc[0]),
            int(target_counts.loc[1]),
            identity_coverage,
            train_transaction.memory_usage(deep=True).sum() / 1024**2,
            train_identity.memory_usage(deep=True).sum() / 1024**2,
        ],
    }
)
display(summary_stats)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=train_transaction, x="isFraud", palette=["#4daf4a", "#e41a1c"], ax=axes[0])
axes[0].set_title("Target Class Distribution")
axes[0].set_xlabel("isFraud")
axes[0].set_ylabel("Transactions")

target_counts.plot.pie(
    autopct="%1.1f%%",
    startangle=90,
    colors=["#4daf4a", "#e41a1c"],
    ax=axes[1],
)
axes[1].set_ylabel("")
axes[1].set_title("Target Share")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Missingness and Data Quality

High missingness is a defining characteristic of this dataset, especially among the engineered `V`, `D`, and identity columns. Missingness itself may carry signal for fraud, so we inspect it directly rather than dropping columns early.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """missing_summary = pd.concat(
    [
        top_missing(train_transaction, "train_transaction", n=25),
        top_missing(train_identity, "train_identity", n=25),
    ],
    ignore_index=True,
)
display(missing_summary.head(20))

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

tx_missing = top_missing(train_transaction, "train_transaction", n=15).sort_values("missing_rate")
axes[0].barh(tx_missing["feature"], tx_missing["missing_rate"], color="#377eb8")
axes[0].set_title("Top Missing Features: Transaction Table")
axes[0].set_xlabel("Missing rate")

id_missing = top_missing(train_identity, "train_identity", n=15).sort_values("missing_rate")
axes[1].barh(id_missing["feature"], id_missing["missing_rate"], color="#984ea3")
axes[1].set_title("Top Missing Features: Identity Table")
axes[1].set_xlabel("Missing rate")

plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Transaction Amount and Time Patterns

`TransactionDT` is a relative time delta, so we treat it as an ordering variable rather than a real timestamp. For EDA, we derive approximate hours and days since the hidden reference point.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """time_df = train_transaction[["TransactionDT", "TransactionAmt", "isFraud"]].copy()
time_df["TransactionHour"] = (time_df["TransactionDT"] // 3600) % 24
time_df["TransactionDay"] = time_df["TransactionDT"] // (3600 * 24)
time_df["LogTransactionAmt"] = np.log1p(time_df["TransactionAmt"])

display(
    time_df[["TransactionAmt", "LogTransactionAmt", "TransactionDT", "TransactionHour", "TransactionDay"]]
    .describe()
    .T
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.kdeplot(
    data=time_df.sample(min(len(time_df), 120000), random_state=RANDOM_STATE),
    x="LogTransactionAmt",
    hue="isFraud",
    common_norm=False,
    fill=True,
    ax=axes[0],
)
axes[0].set_title("Log Transaction Amount by Target")

hourly = (
    time_df.groupby("TransactionHour")
    .agg(transactions=("isFraud", "size"), fraud_rate=("isFraud", "mean"))
    .reset_index()
)
sns.lineplot(data=hourly, x="TransactionHour", y="fraud_rate", marker="o", color="#d95f02", ax=axes[1])
axes[1].set_title("Fraud Rate by Hour of Day")
axes[1].set_ylabel("Fraud rate")

plt.tight_layout()
plt.show()

daily = (
    time_df.groupby("TransactionDay")
    .agg(transactions=("isFraud", "size"), fraud_rate=("isFraud", "mean"))
    .reset_index()
)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
sns.lineplot(data=daily, x="TransactionDay", y="transactions", color="#377eb8", ax=axes[0])
axes[0].set_title("Transaction Volume Over Time")
axes[0].set_ylabel("Transactions")

sns.lineplot(data=daily, x="TransactionDay", y="fraud_rate", color="#e41a1c", ax=axes[1])
axes[1].set_title("Fraud Rate Over Time")
axes[1].set_ylabel("Fraud rate")
axes[1].set_xlabel("Approximate day index")

plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Categorical Fraud Patterns

The dataset contains many masked categorical columns. We start with business-readable fields such as product, card network, email domain, and selected match variables.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """categorical_features = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "M4"]

for feature in categorical_features:
    summary = fraud_rate_table(train_transaction, feature, top_n=10)
    print(f"\\nFraud profile for {feature}")
    display(summary)
    plot_fraud_rate(summary, f"Fraud Rate by {feature}", feature)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Identity and Device Signals

Identity information is only available for a subset of transactions, but these fields often capture browser, OS, and device traits that are useful for fraud detection.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """identity_analysis = train_transaction[
    ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt"]
].merge(train_identity, on="TransactionID", how="left")

identity_analysis["has_identity"] = identity_analysis["DeviceType"].notna()

identity_presence = (
    identity_analysis.groupby("has_identity")["isFraud"]
    .agg(["size", "mean"])
    .rename(columns={"size": "transactions", "mean": "fraud_rate"})
)
display(identity_presence)

for feature in ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]:
    summary = fraud_rate_table(identity_analysis, feature, top_n=12)
    print(f"\\nFraud profile for {feature}")
    display(summary)
    plot_fraud_rate(summary, f"Fraud Rate by {feature}", feature)
"""
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        """## Predictive Baseline

To keep the notebook practical, we fit a lightweight gradient-boosting baseline on a curated subset of transaction and identity variables. We use a chronological split based on `TransactionDT`, which is more realistic than a random split for fraud problems with evolving behavior.
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """feature_columns = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    "C1",
    "C2",
    "C5",
    "C11",
    "C12",
    "C14",
    "D1",
    "D2",
    "D3",
    "D4",
    "D10",
    "D11",
    "D15",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
]

identity_columns = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_15",
    "id_16",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_36",
    "id_37",
    "id_38",
]

model_df = train_transaction[feature_columns].merge(
    train_identity[identity_columns], on="TransactionID", how="left"
)

model_df["TransactionHour"] = (model_df["TransactionDT"] // 3600) % 24
model_df["TransactionDay"] = model_df["TransactionDT"] // (3600 * 24)
model_df["LogTransactionAmt"] = np.log1p(model_df["TransactionAmt"])

model_df = model_df.sort_values("TransactionDT").reset_index(drop=True)
split_idx = int(len(model_df) * 0.8)
train_df = model_df.iloc[:split_idx].copy()
valid_df = model_df.iloc[split_idx:].copy()

target = "isFraud"
drop_columns = ["TransactionID", "isFraud"]
feature_df = model_df.drop(columns=drop_columns)

categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = [col for col in feature_df.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                            encoded_missing_value=-1,
                        ),
                    ),
                ]
            ),
            categorical_cols,
        ),
    ]
)

model = HistGradientBoostingClassifier(
    learning_rate=0.08,
    max_depth=6,
    max_iter=250,
    min_samples_leaf=80,
    random_state=RANDOM_STATE,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

X_train = train_df.drop(columns=drop_columns)
y_train = train_df[target]
X_valid = valid_df.drop(columns=drop_columns)
y_valid = valid_df[target]

pipeline.fit(X_train, y_train)
valid_pred = pipeline.predict_proba(X_valid)[:, 1]

metrics = pd.DataFrame(
    {
        "metric": ["ROC AUC", "Average precision", "Validation fraud rate"],
        "value": [
            roc_auc_score(y_valid, valid_pred),
            average_precision_score(y_valid, valid_pred),
            y_valid.mean(),
        ],
    }
)
display(metrics)

top_k = max(1, int(len(y_valid) * 0.05))
top_precision = y_valid.iloc[np.argsort(-valid_pred)[:top_k]].mean()
lift = top_precision / y_valid.mean()
print(f"Precision within top 5% highest-risk validation transactions: {top_precision:.2%}")
print(f"Lift over base fraud rate: {lift:.2f}x")
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """sample_size = min(5000, len(X_valid))
sample_idx = np.linspace(0, len(X_valid) - 1, sample_size, dtype=int)
X_valid_sample = X_valid.iloc[sample_idx]
y_valid_sample = y_valid.iloc[sample_idx]

perm = permutation_importance(
    pipeline,
    X_valid_sample,
    y_valid_sample,
    n_repeats=5,
    random_state=RANDOM_STATE,
    scoring="average_precision",
)

importance = pd.DataFrame(
    {
        "feature": X_valid.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }
).sort_values("importance_mean", ascending=False)

display(importance.head(15))

top_importance = importance.head(12).sort_values("importance_mean")
plt.figure(figsize=(12, 7))
plt.barh(top_importance["feature"], top_importance["importance_mean"], color="#377eb8")
plt.xlabel("Permutation importance (average precision drop)")
plt.title("Most Important Features in the Baseline Model")
plt.tight_layout()
plt.show()
"""
    )
)

cells.append(
    nbf.v4.new_code_cell(
        """key_findings = pd.DataFrame(
    {
        "finding": [
            "The training set is heavily imbalanced, so precision-recall metrics are more informative than accuracy.",
            "Only a minority of transactions include identity data, but those fields still contain meaningful fraud signal.",
            "Missingness is extensive across engineered features, suggesting that null patterns themselves may be predictive.",
            "Fraud patterns vary over the relative transaction timeline, so time-aware validation is appropriate.",
            "A compact baseline already separates fraud substantially better than the raw base rate, indicating the prediction task is learnable.",
        ]
    }
)
display(key_findings)
"""
    )
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.x",
    },
}

with open("notebooks/EDA.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
