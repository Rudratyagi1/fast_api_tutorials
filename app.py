# Wine Quality Prediction with ElasticNet + MLflow

import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow
import mlflow.sklearn as ms
from mlflow.models.signature import infer_signature

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
# Main
# ------------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read dataset
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
    train_y = train["quality"]  # 1D Series for ElasticNet
    test_y = test["quality"]

    # Hyperparameters from command-line arguments
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # Start MLflow run
    with mlflow.start_run():
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
        predictions_train = lr.predict(train_x)
        signature = infer_signature(train_x, predictions_train)

        # Remote tracking optional
        # remote_server_uri = "https://dagshub.com/krishnaik06/mlflowexperiments.mlflow"
        # mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Log model
        if tracking_url_type_store != "file":
            # Register model on remote server
            ms.log_model(
                lr,
                name="ElasticnetWineModel",
                registered_model_name="ElasticnetWineModel",
                signature=signature,
                input_example=train_x.iloc[:5]
            )
        else:
            # Local logging
            ms.log_model(
                lr,
                name="ElasticnetWineModel",
                signature=signature,
                input_example=train_x.iloc[:5]
            )
