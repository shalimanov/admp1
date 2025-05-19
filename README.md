# Звіт: Лабораторної роботи 1: Класифікація

## 1. Мета та завдання

Метою цього практичного проєкту було:

- Реалізувати на Python чотири алгоритми класифікації:
  1. One-Rule  
  2. Gaussian Naive Bayes  
  3. Decision Tree  
  4. kNN  
- Підібрати придатний датасет з Kaggle, обрано Rain in Australia.  
- Провести цикл попередньої обробки даних:
  - Видалення некорисних ознак (`Date`, `RISK_MM`, `Evaporation`, `Sunshine`)  
  - Імпутація пропусків  
  - Кодування категоріальних змінних (one-hot, бінарне)  
  - Стандартизація числових ознак  
  - Додавання часових ознак та їх циклічне кодування  
- Навчити моделі, виконати тюнінг гіперпараметрів (`max_depth` для Decision Tree, `k` для kNN).  
- Оцінити кожну модель за метриками Accuracy, Precision, Recall, F1 та порівняти результати.

## 2. Дані

- **Джерело**: Kaggle Rain in Australia `weatherAUS.csv`  
- **Розмірність**: 142 193 записів, 23 вихідні ознаки + цільова `RainTomorrow`  
- **Ціль**: бінарна змінна — чи йде дощ наступного дня (`Yes`/`No`)

## 3. Попередня обробка даних

1. **Видалення стовпців**  
   - `RISK_MM`, `Evaporation`, `Sunshine`  
2. **Імпутація пропусків**  
   - числові → медіана  
   - категоріальні → мода  
3. **Кодування**  
   - `RainToday`, `RainTomorrow` → 0/1  
   - one-hot для `Location`, `WindGustDir`, `WindDir9am`, `WindDir3pm`  
4. **Нормалізація**  
   - `StandardScaler` для всіх числових  
5. **Часові ознаки**  
   ```python
   df['Date'] = pd.to_datetime(df['Date'])
   df['month']         = df['Date'].dt.month
   df['day_of_week']   = df['Date'].dt.dayofweek
   df['month_sin']     = np.sin(2*np.pi*df['month']/12)
   df['month_cos']     = np.cos(2*np.pi*df['month']/12)
   df['dow_sin']       = np.sin(2*np.pi*df['day_of_week']/7)
   df['dow_cos']       = np.cos(2*np.pi*df['day_of_week']/7)
   df.drop(columns=['Date'], inplace=True)
   ```

## 4. Реалізація моделей та тюнінг

### One-Rule
- квантильне бінування (5 бінів), вибір ознаки з мінімальною помилкою.

### Gaussian Naive Bayes
- `sklearn.naive_bayes.GaussianNB` без змін.

### Decision Tree
- `DecisionTreeClassifier(max_depth=…)`:
  ```
  depth=3  → acc=0.830  
  depth=5  → acc=0.837  
  depth=10 → acc=0.841 ← обрано  
  depth=None → acc=0.791  
  ```

### kNN
- `KNeighborsClassifier(n_neighbors=…)`:
  ```
  k=1   → acc=0.802  
  k=5   → acc=0.836  
  k=9   → acc=0.840  
  k=15  → acc=0.842 ← обрано  
  ```

## 5. Результати

### 5.1. Повний тестовий набір

| Модель                  | Accuracy | Precision (1) | Recall (1) | F1₁   |
|-------------------------|:--------:|:-------------:|:----------:|:-----:|
| One-Rule                | 0.810    |   0.591       |  0.497     | 0.539 |
| Naive Bayes             | 0.647    |   0.356       |  0.710     | 0.474 |
| Decision Tree (depth=10)| 0.842    |   0.713       |  0.489     | 0.581 |
| kNN (k=15)              | 0.842    |   0.739       |  0.456     | 0.564 |

#### 5.1.1 Матриці невідповідностей (повний тест)

**One-Rule**  
```
         pred_0  pred_1
true_0   19870    2194
true_1    3210    3165
```

**Naive Bayes**  
```
         pred_0  pred_1
true_0   13869    8195
true_1    1848    4527
```

**Decision Tree (depth=10)**  
```
         pred_0  pred_1
true_0   20810    1254
true_1    3255    3120
```

**kNN (k=15)**  
```
         pred_0  pred_1
true_0   21036    1028
true_1    3470    2905
```

---

### 5.2. Випадкова вибірка (n = 100)

| Модель                  | Accuracy | Precision (1) | Recall (1) | F1₁   |
|-------------------------|:--------:|:-------------:|:----------:|:-----:|
| One-Rule                | 0.740    |      —        |  0.000     | 0.000 |
| Naive Bayes             | 0.670    |   0.422       |  0.731     | 0.535 |
| Decision Tree (depth=10)| 0.780    |   0.643       |  0.346     | 0.450 |
| kNN (k=15)              | 0.800    |   0.750       |  0.346     | 0.474 |

#### 5.2.1 Матриці невідповідностей (випадкова вибірка)

**One-Rule**  
```
         pred_0  pred_1
true_0      74       0
true_1      26       0
```

**Naive Bayes**  
```
         pred_0  pred_1
true_0      48      26
true_1       7      19
```

**Decision Tree (depth=10)**  
```
         pred_0  pred_1
true_0      69       5
true_1      17       9
```

**kNN (k=15)**  
```
         pred_0  pred_1
true_0      71       3
true_1      17       9
```

## 6. Висновки

- **Decision Tree (depth=10)** отримує найвищу збалансовану ефективність на повному тесті (Accuracy=0.842, F1=0.581).  
- **kNN (k=15)** має трохи вищу Precision, але нижчий Recall (F1=0.564).  
- **Naive Bayes** демонструє найкращий Recall (0.710) на повному наборі, проте з низьким Precision.  
- **One-Rule** — найпростіша модель з найгіршим Recall для класу “дощ”.  
- На випадковій вибірці метрики дещо коливаються, але загальний тренд зберігається.
