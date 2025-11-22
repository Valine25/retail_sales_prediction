# forecast_xgb_fast.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import time

# ------------------------
# CONFIG
# ------------------------
DATA_PATH = "retail_daily_sales.csv"   # Make sure file is in same folder
RESULTS_DIR = "forecast_xgb_results_fast"
FUTURE_DAYS = 30
os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(42)

# ------------------------
# HELPER: SMAPE
# ------------------------
def smape(a, f):
    a = np.array(a, dtype=float)
    f = np.array(f, dtype=float)
    return 100.0 / len(a) * np.sum(2.0 * np.abs(f - a) / (np.abs(a) + np.abs(f) + 1e-8))

# ------------------------
# LOAD & PREPARE DATA
# ------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

# Adjust column names if needed
if "Category" in df.columns and "Sales" in df.columns:
    df = df.rename(columns={"Category": "Product Category", "Sales": "Total Amount"})
elif "Product Category" in df.columns and "Total Amount" in df.columns:
    pass
else:
    raise ValueError("CSV must have (Category, Sales) or (Product Category, Total Amount).")

df = df.sort_values(["Product Category", "Date"]).reset_index(drop=True)
categories = df["Product Category"].unique().tolist()

print("Categories:", categories)

# ------------------------
# TRAIN PER CATEGORY
# ------------------------
summary = []
future_all = []

for cat in categories:
    print(f"\nProcessing category: {cat}")
    t0 = time.time()

    # Prepare category-specific data
    cat_df = df[df["Product Category"] == cat].set_index("Date")[["Total Amount"]].copy()

    # Ensure continuous daily index
    full_idx = pd.date_range(start=cat_df.index.min(), end=cat_df.index.max(), freq="D")
    cat_df = cat_df.reindex(full_idx).fillna(0)
    cat_df.index.name = "Date"

    # Time features
    cat_df["dayofweek"] = cat_df.index.dayofweek
    cat_df["month"] = cat_df.index.month
    cat_df["week"] = cat_df.index.isocalendar().week.astype(int)
    cat_df["is_weekend"] = (cat_df["dayofweek"] >= 5).astype(int)

    # Lag features
    cat_df["lag_1"] = cat_df["Total Amount"].shift(1)
    cat_df["lag_7"] = cat_df["Total Amount"].shift(7)
    cat_df["lag_30"] = cat_df["Total Amount"].shift(30)

    # Rolling averages
    cat_df["roll_7"] = cat_df["Total Amount"].shift(1).rolling(7).mean()
    cat_df["roll_30"] = cat_df["Total Amount"].shift(1).rolling(30).mean()

    # Drop rows with missing lag/rolling features
    cat_df = cat_df.dropna()

    # Train-test split
    train = cat_df.iloc[:-30]
    test = cat_df.iloc[-30:]

    features = ["dayofweek", "month", "week", "is_weekend",
                "lag_1", "lag_7", "lag_30", "roll_7", "roll_30"]

    X_train, y_train = train[features], train["Total Amount"]
    X_test, y_test = test[features], test["Total Amount"]

    # XGBoost Model (NO early stopping)
    model = XGBRegressor(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=4
    )

    model.fit(X_train, y_train)  # << FIXED: early stopping removed

    # Predictions on test
    preds = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    s = smape(y_test, preds)

    summary.append({"category": cat, "MAE": mae, "RMSE": rmse, "R2": r2, "SMAPE": s})

    # Save model
    joblib.dump(model, os.path.join(RESULTS_DIR, f"xgb_model_{cat}.joblib"))

    # Save test predictions
    res_test = test.reset_index().rename(columns={"index": "Date"})
    res_test["Predicted"] = preds
    res_test.to_csv(os.path.join(RESULTS_DIR, f"test_predictions_{cat}.csv"), index=False)

    # Plot & save PNG
    plt.figure(figsize=(8, 3))
    plt.plot(res_test["Date"], res_test["Total Amount"], label="Actual", marker="o")
    plt.plot(res_test["Date"], res_test["Predicted"], label="Predicted", marker="x")
    plt.title(f"{cat} - Actual vs Predicted (Test)")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"plot_test_{cat}.png"))
    plt.close()

    # ------------------------
    # FUTURE FORECAST (NEXT 30 DAYS)
    # ------------------------
    last_known = cat_df.copy()
    future_idx = pd.date_range(start=cat_df.index.max() + pd.Timedelta(days=1), periods=FUTURE_DAYS, freq="D")
    future_rows = []

    for date in future_idx:
        lag_1 = last_known["Total Amount"].iloc[-1]
        lag_7 = last_known["Total Amount"].shift(7).iloc[-1] if len(last_known) >= 7 else lag_1
        lag_30 = last_known["Total Amount"].shift(30).iloc[-1] if len(last_known) >= 30 else lag_1
        roll_7 = last_known["Total Amount"].shift(1).rolling(7).mean().iloc[-1]
        roll_30 = last_known["Total Amount"].shift(1).rolling(30).mean().iloc[-1]
        dow = date.dayofweek
        month = date.month
        week = date.isocalendar()[1]
        is_weekend = 1 if dow >= 5 else 0

        x = np.array([dow, month, week, is_weekend,
                      lag_1, lag_7, lag_30, roll_7, roll_30]).reshape(1, -1)
        pred = model.predict(x)[0]

        future_rows.append([date, cat, pred])

        new_row = pd.DataFrame({"Total Amount": [pred]}, index=[date])
        last_known = pd.concat([last_known, new_row])

    future_df_cat = pd.DataFrame(future_rows, columns=["Date", "Product Category", "Predicted_Sales"])
    future_df_cat.to_csv(os.path.join(RESULTS_DIR, f"future_forecast_{cat}.csv"), index=False)
    future_all.append(future_df_cat)

    print(f"Finished {cat} — MAE={mae:.1f}, RMSE={rmse:.1f}, R2={r2:.3f}, SMAPE={s:.2f}%")

# ------------------------
# SAVE SUMMARY
# ------------------------
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(RESULTS_DIR, "summary_metrics.csv"), index=False)

future_all_df = pd.concat(future_all).reset_index(drop=True)
future_all_df.to_csv(os.path.join(RESULTS_DIR, "future_forecast_all_categories.csv"), index=False)

print("\nAll results saved to:", RESULTS_DIR)
print(summary_df)
