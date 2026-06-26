import os
import yaml
import joblib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
import tempfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Базовая директория проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Определяем путь к БД MLflow
if os.getenv("CI"):
    # В GitHub Actions используем временную папку
    db_path = os.path.join(tempfile.gettempdir(), "mlflow.db")
else:
    # Локально используем папку проекта
    db_path = os.path.join(BASE_DIR, "mlflow.db")

sqlite_uri = f"sqlite:///{db_path}"

# Читаем URI сервера из переменных окружения, либо используем локальный SQLite
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", sqlite_uri)
mlflow.set_tracking_uri(tracking_uri)

# Задаем имя эксперимента
mlflow.set_experiment("Iris_Classification")


def train_model():
    with open(os.path.join(BASE_DIR, "params.yaml"), "r") as f:
        params = yaml.safe_load(f)["train"]

    df = pd.read_csv(os.path.join(BASE_DIR, "data", "iris.csv"))
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.sklearn.autolog()

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        mlflow.log_metric("accuracy", acc)

        metrics_path = os.path.join(BASE_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"accuracy": acc}, f)

        model_path = os.path.join(BASE_DIR, "models", "model.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"Модель обучена. Accuracy: {acc}")


if __name__ == "__main__":
    train_model()
