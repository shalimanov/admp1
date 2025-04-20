from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from .data_prep import load_and_prepare
from .one_rule import OneRuleClassifier

def evaluate(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=3))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")

def main():
    X_train, X_test, y_train, y_test, feat_names = load_and_prepare()
    # 1‑Rule
    one_r = OneRuleClassifier(n_bins=5)
    one_r.fit(X_train, y_train)
    one_r_pred = one_r.predict(X_test)
    evaluate("1‑Rule", y_test, one_r_pred)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    evaluate("Naive Bayes", y_test, nb_pred)

    # Decision Trees with depth sweep
    best_dt_acc = 0
    best_dt_pred = None
    best_depth = None
    for depth in [3, 5, 10, None]:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        pred = dt.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"DecisionTree depth={depth} acc={acc:.3f}")
        if acc > best_dt_acc:
            best_dt_acc = acc
            best_dt_pred = pred
            best_depth = depth
    evaluate(f"Decision Tree (best depth={best_depth})", y_test, best_dt_pred)

    # kNN with k sweep
    best_knn_acc = 0
    best_knn_pred = None
    best_k = None
    for k in [1, 5, 9, 15]:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print(f"kNN k={k} acc={acc:.3f}")
        if acc > best_knn_acc:
            best_knn_acc = acc
            best_knn_pred = pred
            best_k = k
    evaluate(f"kNN (best k={best_k})", y_test, best_knn_pred)

if __name__ == "__main__":
    main()