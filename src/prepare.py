from sklearn.datasets import load_iris
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Загружаем датасет и сохраняем его в CSV
iris = load_iris(as_frame=True)
df = iris.frame
df.to_csv(os.path.join(DATA_DIR, "iris.csv"), index=False)
print("Данные успешно сохранены в data/iris.csv")
