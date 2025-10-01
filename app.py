# Wine Quality Prediction with ElasticNet + MLflow (DAGsHub-safe)

import warnings
import sys
import logging
import tempfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn as ms
from mlflow.models.signature import infer_signature
import dagshub

# ------------------------------
# Logging setup
# ------------------------------
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# ------------------------------
# Metric evaluation
# ------------------------------
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# ------------------------------
# Initialize DAGsHub MLflow tracking
# ------------------------------
dagshub.init(  # type: ignore
    repo_owner='rudratyagi777',
    repo_name='fast_api_tutorials',
    mlflow=True
)

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download CSV. Error: %s", e)
        sys.exit(1)

    # Split data
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    train_x = train.drop("quality", axis=1)
    test_x = test.drop("quality", axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # Hyperparameters
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Start MLflow run
    with mlflow.start_run():
        # Train model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Predictions & metrics
        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")

        # Log parameters & metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Model signature & input example
        signature = infer_signature(train_x, lr.predict(train_x))

        # ------------------------------
        # DAGsHub-safe model logging
        # ------------------------------
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = f"{tmp_dir}/elasticnet_model"
            ms.save_model(lr, path=model_path, signature=signature, input_example=train_x.iloc[:5])
            mlflow.log_artifacts(model_path, artifact_path="ElasticnetWineModel")

    print("✅ Model, metrics, and parameters logged successfully to DAGsHub MLflow!")
