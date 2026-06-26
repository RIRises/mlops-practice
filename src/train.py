import os
import yaml
import joblib
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Формируем абсолютные пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = "/home/cachy/mlops-practice/mlflow.db"
sqlite_uri = f"sqlite:///{db_path}"

# Читаем URI сервера из переменных окружения, либо используем локальный SQLite
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", sqlite_uri)
mlflow.set_tracking_uri(tracking_uri)

# Задаем имя эксперимента
mlflow.set_experiment("Iris_Classification")


def train_model():
    with open(os.path.join(BASE_DIR, "..", "params.yaml"), "r") as f:
        params = yaml.safe_load(f)["train"]

    df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "iris.csv"))
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Открываем сессию MLflow
    with mlflow.start_run():
        # ЯВНО логируем параметры
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])

        # Включаем автологирование для scikit-learn (дополнительно)
        mlflow.sklearn.autolog()

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # ЯВНО логируем метрику
        mlflow.log_metric("accuracy", acc)

        # Сохраняем метрику для DVC
        metrics_path = os.path.join(BASE_DIR, "..", "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({"accuracy": acc}, f)

        # Сохраняем модель для DVC и MLflow
        model_path = os.path.join(BASE_DIR, "..", "models", "model.pkl")
        joblib.dump(model, model_path)

        # Логируем модель как артефакт в MLflow
        mlflow.log_artifact(model_path)

        print(f"Модель обучена. Accuracy: {acc}")


if __name__ == "__main__":
    train_model()
