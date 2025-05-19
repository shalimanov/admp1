# src/test_random.py
import os
import glob
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from .data_prep import load_and_prepare

# Кількість випадкових прикладів
N_SAMPLES = 100

# Шляхи
ARTIFACTS_DIR = os.path.join(os.getcwd(), 'artifacts')
MODELS_DIR = os.path.join(ARTIFACTS_DIR, 'models')
RANDOM_CM_DIR = os.path.join(ARTIFACTS_DIR, 'confusion_matrices', 'random')
os.makedirs(RANDOM_CM_DIR, exist_ok=True)

def main():
    # Завантажуємо дані
    X_train, X_test, y_train, y_test, feat_names = load_and_prepare()
    # Вбираємо N випадкових індексів
    idx = np.random.choice(len(X_test), size=min(N_SAMPLES, len(X_test)), replace=False)
    X_sample = X_test.iloc[idx]
    y_sample = y_test.iloc[idx]

    # Для кожного збереженого файлу-моделі
    for model_path in glob.glob(os.path.join(MODELS_DIR, '*.pkl')):
        name = os.path.splitext(os.path.basename(model_path))[0]
        model = joblib.load(model_path)

        # Передбачення та матриця
        y_pred = model.predict(X_sample)
        cm = confusion_matrix(y_sample, y_pred)

        # Зберігаємо як CSV
        df_cm = pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in np.unique(y_sample)],
            columns=[f"pred_{c}" for c in np.unique(y_sample)]
        )
        df_cm.to_csv(os.path.join(RANDOM_CM_DIR, f"{name}_random_cm.csv"))
        print(f"Saved random CM for {name} → {name}_random_cm.csv")

if __name__ == "__main__":
    main()
