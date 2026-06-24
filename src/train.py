import os
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def train_model():
    # Читаем параметры из конфига
    with open(os.path.join(BASE_DIR, "..", "params.yaml"), "r") as f:
        params = yaml.safe_load(f)["train"]

    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Инициализируем модель с параметрами из yaml
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print("Accuracy:", acc)

    # Сохраняем артефакт (модель)
    joblib.dump(model, os.path.join(BASE_DIR, "..", "models", "model.pkl"))


if __name__ == "__main__":
    train_model()
