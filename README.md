# Motivation

While some Kaggle solutions use stratified cross-validation, I adopted a time-aware validation strategy to better reflect real-world fraud detection scenarios, where transaction distributions shift over time. This avoids temporal leakage and provides more realistic performance estimates.

# Procedures

1. IEEE-CIS Raw Data (Local/Cloud)

2. Preprocessing & ELT cleaning, encoding train/val/test split

3. Model training (e.g. LightGBM/XGBoost) + Threshold tuning

4. MLflow Tracking params, metrics, artifects, registry

5. FastAPI Inference /predict /health

6. Prediction Logs + Monitoring drift/score shift

7. Scheduled Retraining via Prefect

# Champion Model

The final champion model is an XGBoost classifier using the `base` feature set. It was selected because it achieved the highest validation ROC-AUC among the evaluated candidates, then confirmed strong performance on the held-out streaming test window.

Final model configuration:

```text
Registered model: FraudDetectionXGBoostChampion
Model URI: models:/FraudDetectionXGBoostChampion@champion
Model: xgboost
Feature set: base
Training scope: train_plus_validation
Evaluation mode: streaming
Streaming batch size: 1
Classification threshold: 0.2138
Threshold selection: max validation F1
```

The operating threshold is selected after model selection. The model candidate is chosen by validation ROC-AUC, then the selected candidate's classification threshold is tuned on the streaming validation window to maximize F1. This moved the operating threshold from the default `0.5` to `0.2138`, improving validation recall while keeping precision acceptable for a fraud alerting workflow.

Validation metrics at selected threshold:

```text
F1: 0.5748
Precision: 0.6499
Recall: 0.5152
```

Final held-out test metrics:

```text
ROC-AUC: 0.9287
Average precision: 0.5960
F1: 0.5224
Precision: 0.8311
Recall: 0.3809
Accuracy: 0.9757
```

The test metrics above reflect the final held-out streaming test evaluation. The test set is used only for final evaluation, not for model or threshold selection.

The production-style pipeline trains models offline on historical data, validates candidate models on a later streaming validation window, retrains the selected candidate on train plus validation data, and evaluates once on the final streaming test window. After test evaluation, the selected model is promoted to the MLflow model registry with the `champion` alias and documented in `artifacts/champion_model.json`.

# Local pipeline runs with Docker

Build the pipeline image from the repository root:

```bash
docker build -t fraud-detection-mlops:local .
```

The Dockerfile default command is `python main.py test`, so running the image without a stage override starts streaming test. Use the mounted directories below to provide the local interim dataset and persist MLflow runs and generated artifacts after the container exits.

To run training:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/mlruns:/app/mlruns" \
  -v "$PWD/artifacts:/app/artifacts" \
  fraud-detection-mlops:local \
  python main.py training
```

To run streaming validation:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/mlruns:/app/mlruns" \
  -v "$PWD/artifacts:/app/artifacts" \
  fraud-detection-mlops:local \
  python main.py validation
```

To run streaming test evaluation:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/mlruns:/app/mlruns" \
  -v "$PWD/artifacts:/app/artifacts" \
  fraud-detection-mlops:local
```

This is equivalent to:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/mlruns:/app/mlruns" \
  -v "$PWD/artifacts:/app/artifacts" \
  fraud-detection-mlops:local \
  python main.py test
```

Validation expects `artifacts/model_comparison_incremental.csv` and `artifacts/selected_v_columns.json` to exist. Test evaluation expects `artifacts/model_validation_incremental.csv`, `artifacts/selected_model.json`, and `artifacts/selected_v_columns.json` to exist.

To view MLflow while a pipeline is running, use another terminal:

```bash
mlflow ui --backend-store-uri mlruns
```

Then open `http://127.0.0.1:5000`. Refresh the page while training runs to see new runs, metrics, params, and artifacts appear.

If MLflow fails with `Address already in use`, port `5000` is already occupied. Check the existing process:

```bash
lsof -i :5000
```

If an existing MLflow UI is already running, keep using `http://127.0.0.1:5000`. Otherwise, stop the old process:

```bash
kill <PID>
```

or start MLflow on another port:

```bash
mlflow ui --backend-store-uri mlruns --port 5001
```

Then open `http://127.0.0.1:5001`.

If MLflow is not installed in the local Python environment, run the UI from Docker instead:

```bash
docker run --rm \
  -p 5000:5000 \
  -v "$PWD/mlruns:/app/mlruns" \
  fraud-detection-mlops:local \
  mlflow ui --backend-store-uri /app/mlruns --host 0.0.0.0 --port 5000
```

To run the test suite inside the same image:

```bash
docker run --rm fraud-detection-mlops:local pytest
```

Note: mounting `artifacts:/app/artifacts` allows the pipeline stages to read and update files in the local `artifacts/` directory.

# Installing raw data from kaggle IEEE-CIS competition

1. Install KaggleHub CLI: ```pip install kagglehub```

2. Create your own Kaggle account, and then login in your computer: ```kagglehub.login()```

3. Join IEEE-CIS competetion from your Kaggle account, before you can call fdb.datasets with ieeecis

4. Download the authentication token from "My Account" on Kaggle, and save token at ~/.kaggle/kaggle.json on Linux, OSX and at C:\Users<Windows-username>.kaggle\kaggle.json on Windows.

5. Use the token by: ```export KAGGLE_API_TOKEN=KGAT_94797bc62067462d83bedbf21ed612bc``` (token name ```kaggle_api```)

6. After the setup, download using ```kaggle competitions download ieee-fraud-detection --p ./data/raw```

7. Unzip the files via ```unzip ./data/raw/ieee-fraud-detection.zip -d ./data/raw/ieee-fraud-detection/```

# Dataset Descrption

## Transaction Table:

TransactionDT: timedelta from a given reference datetime (not an actual timestamp)

TransactionAMT: transaction payment amount in USD

ProductCD: product code, the product for each transaction

card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.

addr: address

dist: distance

P_ and (R__) emaildomain: purchaser and recipient email domain

C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc. The actual meaning is masked.

D1-D15: timedelta, such as days between previous transaction, etc.

M1-M9: match, such as names on card and address, etc.

Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

## Identity Table

Identity information: Network connection information (IP, ISP, Proxy, etc) 

Digital signature (UA/browser/os/version, etc) 

Categorical Features: DeviceType, DeviceInfo, id_12 - id_38
