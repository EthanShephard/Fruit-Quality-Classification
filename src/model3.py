import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# ===============================
# CONFIG
# ===============================
DATA_PATH = "price_data.csv"
MODEL_PATH = "price_model.pkl"

# ===============================
# LOAD DATA
# ===============================
def load_data(path):
    df = pd.read_csv(path)

    required_columns = ["count", "weight_kg", "market_rate", "price"]
    for col in required_columns:
        if col not in df.columns:
            raise Exception(f"Missing column: {col}")

    return df


# ===============================
# TRAIN MODEL
# ===============================
def train_model(df):
    X = df[["count", "weight_kg", "market_rate"]]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train, y_train)

    # ===============================
    # EVALUATION
    # ===============================
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print("\n===== MODEL PERFORMANCE =====")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    return model


# ===============================
# SAVE MODEL
# ===============================
def save_model(model, path=MODEL_PATH):
    joblib.dump(model, path)
    print(f"Model saved at {path}")


# ===============================
# LOAD MODEL
# ===============================
def load_model(path=MODEL_PATH):
    return joblib.load(path)


# ===============================
# PREDICT FUNCTION (USE IN APP)
# ===============================
def predict_price(model, count, weight_kg, market_rate=120):
    features = np.array([[count, weight_kg, market_rate]])
    price = model.predict(features)[0]
    return round(float(price), 2)


# ===============================
# MAIN (TRAIN + SAVE)
# ===============================
if __name__ == "__main__":
    df = load_data(DATA_PATH)

    print("Dataset loaded:")
    print(df.head())

    model = train_model(df)

    save_model(model)