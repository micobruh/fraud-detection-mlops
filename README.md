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

# Local training with Docker

Build the training image from the repository root:

```bash
docker build -t fraud-detection-training:local .
```

To view MLflow while training is running, use two terminals.

Terminal 1, start the training flow:

```bash
docker run --rm \
  -v "$PWD/data:/app/data" \
  -v "$PWD/mlruns:/app/mlruns" \
  -v "$PWD/artifacts:/app/artifacts" \
  fraud-detection-training:local
```

The Dockerfile default command is `python main.py`, so the command above starts the model training flow. The mounted directories provide the real local dataset and persist MLflow runs and generated artifacts after the container exits.

Terminal 2, start the MLflow UI:

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
  fraud-detection-training:local \
  mlflow ui --backend-store-uri /app/mlruns --host 0.0.0.0 --port 5000
```

To run the test suite inside the same image:

```bash
docker run --rm fraud-detection-training:local pytest
```

Note: mounting `artifacts:/app/artifacts` allows training to update files in the local `artifacts/` directory.

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
