import mlflow.pyfunc
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = "/home/cachy/mlops-practice/mlflow.db"
sqlite_uri = f"sqlite:///{db_path}"

mlflow.set_tracking_uri(sqlite_uri)

model_name = "Iris_RF_Model"
alias = "champion"
model_uri = f"models:/{model_name}@{alias}"

print(f"Загружаем модель: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

dummy_data = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ],
)

prediction = model.predict(dummy_data)
print(f"✅ Предсказание боевой модели: Класс {prediction[0]}")
