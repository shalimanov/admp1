import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DROP_COLS = ['Sunshine', 'Evaporation']  # too many NaNs

BINARY_MAP = {'Yes': 1, 'No': 0}

def add_cyclical_date_features(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Додає до DataFrame колонку місяця, дня тижня та їх циклічні кодування.
    """
    df = df.copy()
    # Перетворюємо рядок у datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Базові ознаки
    df['month'] = df[date_col].dt.month
    df['day_of_week'] = df[date_col].dt.dayofweek

    # Циклічне кодування (sin/cos)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # Видаляємо оригінальний стовпець дати
    df.drop(columns=[date_col], inplace=True)

    return df

def load_and_prepare(csv_path='data/weatherAUS.csv', test_size=0.2, random_state=42):
    """Load CSV, clean NA, encode/cat->dummy, scale numeric.
    Returns tuple (X_train, X_test, y_train, y_test, feature_names)
    """
    df = pd.read_csv(csv_path)

    df = add_cyclical_date_features(df, 'Date')

    if 'RISK_MM' in df.columns:
        df.drop('RISK_MM', axis=1, inplace=True)

    # 1. drop heavy-missing cols
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # 2. fill missing numeric with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in [float, int]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. binary encode RainToday / RainTomorrow
    df['RainToday'] = df['RainToday'].map(BINARY_MAP)
    df['RainTomorrow'] = df['RainTomorrow'].map(BINARY_MAP)

    # 4. one‑hot encode multi‑category columns
    cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 5. separate target
    y = df['RainTomorrow']
    X = df.drop(columns=['RainTomorrow'])

    # 6. scale numeric features
    num_cols = X.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test, X.columns.tolist()