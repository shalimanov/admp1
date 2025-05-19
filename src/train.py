# src/train.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from .data_prep import load_and_prepare
from .one_rule import OneRuleClassifier

# директорії для артефактів
ARTIFACTS_DIR = os.path.join(os.getcwd(), 'artifacts')
MODELS_DIR = os.path.join(ARTIFACTS_DIR, 'models')
METRICS_DIR = os.path.join(ARTIFACTS_DIR, 'metrics')
CM_DIR = os.path.join(ARTIFACTS_DIR, 'confusion_matrices')

for d in (MODELS_DIR, METRICS_DIR, CM_DIR):
    os.makedirs(d, exist_ok=True)

def evaluate_and_save(name, model, X_test, y_test):
    """Оцінивши модель, зберігає результати й саму модель в артефакти."""
    # передбачення
    y_pred = model.predict(X_test)

    # розрахунок метрик
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    acc = accuracy_score(y_test, y_pred)

    # вивід у консоль
    print(f"\n=== {name} ===")
    print(cm)
    print(report)
    print(f"Accuracy: {acc:.3f}")

    # збереження моделі
    model_path = os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl")
    joblib.dump(model, model_path)

    # збереження метрик у текстовий файл
    with open(os.path.join(METRICS_DIR, f"{name.replace(' ', '_')}.txt"), 'w') as f:
        f.write(f"=== {name} ===\n")
        f.write("Confusion matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification report:\n")
        f.write(report)
        f.write(f"\nAccuracy: {acc:.3f}\n")

    # збереження матриці невідповідностей як CSV
    pd.DataFrame(cm,
                 index=[f"true_{c}" for c in np.unique(y_test)],
                 columns=[f"pred_{c}" for c in np.unique(y_test)]
                ).to_csv(os.path.join(CM_DIR, f"{name.replace(' ', '_')}_cm.csv"))

def main():
    X_train, X_test, y_train, y_test, feat_names = load_and_prepare()

    # 1-Rule
    one_r = OneRuleClassifier(n_bins=5)
    one_r.fit(X_train, y_train)
    evaluate_and_save("OneRule", one_r, X_test, y_test)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    evaluate_and_save("NaiveBayes", nb, X_test, y_test)

    # Decision Trees з перебором глибин
    best_dt = None
    best_acc = -1
    for depth in [3, 5, 10, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        acc = accuracy_score(y_test, dt.predict(X_test))
        if acc > best_acc:
            best_acc = acc
            best_dt = dt
            best_depth = depth
    evaluate_and_save(f"DecisionTree_depth={best_depth}", best_dt, X_test, y_test)

    # kNN з перебором k
    best_knn = None
    best_acc = -1
    for k in [1, 5, 9, 15]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        acc = accuracy_score(y_test, knn.predict(X_test))
        if acc > best_acc:
            best_acc = acc
            best_knn = knn
            best_k = k
    evaluate_and_save(f"kNN_k={best_k}", best_knn, X_test, y_test)

if __name__ == "__main__":
    main()
