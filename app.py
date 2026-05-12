from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
import pandas as pd
import os

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "HawkEye ML API is running"
    })

def calculate_wmape(actual, forecast):
    actual = pd.Series(actual).astype(float)
    forecast = pd.Series(forecast).astype(float)

    if actual.sum() == 0:
        return None

    return round((abs(actual - forecast).sum() / actual.sum()) * 100, 2)

def backtest_naive(actuals):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    for i in range(1, len(actuals)):
        forecasts.append(actuals[i - 1])
        actual_test.append(actuals[i])

    return calculate_wmape(actual_test, forecasts)

def backtest_moving_average(actuals, window=3):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    for i in range(window, len(actuals)):
        forecasts.append(sum(actuals[i-window:i]) / window)
        actual_test.append(actuals[i])

    return calculate_wmape(actual_test, forecasts)

def backtest_linear_regression(sku_df):
    forecasts = []
    actual_test = []

    for i in range(2, len(sku_df)):
        train = sku_df.iloc[:i]
        test = sku_df.iloc[i]

        X_train = train[["month_number"]]
        y_train = train["actual_units"]

        model = LinearRegression()
        model.fit(X_train, y_train)

        prediction = model.predict([[test["month_number"]]])[0]

        forecasts.append(prediction)
        actual_test.append(test["actual_units"])

    return calculate_wmape(actual_test, forecasts)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "records" not in data:
        return jsonify({
            "status": "error",
            "message": "No records provided"
        }), 400

    df = pd.DataFrame(data["records"])

    model_results = []
    best_models = []

    skus = df["sku"].unique()

    for sku in skus:

        sku_df = df[df["sku"] == sku].copy()
        sku_df["month_number"] = sku_df["month_number"].astype(float)
        sku_df["actual_units"] = sku_df["actual_units"].astype(float)

        sku_df = sku_df.sort_values("month_number")

        next_month = int(sku_df["month_number"].max()) + 1
        actuals = sku_df["actual_units"].tolist()

        # Linear Regression future forecast
        lr_model = LinearRegression()
        lr_model.fit(sku_df[["month_number"]], sku_df["actual_units"])
        lr_prediction = lr_model.predict([[next_month]])[0]
        lr_wmape = backtest_linear_regression(sku_df)

        # Naive future forecast
        naive_prediction = actuals[-1]
        naive_wmape = backtest_naive(actuals)

        # 3-Month Moving Average future forecast
        ma_prediction = sum(actuals[-3:]) / 3
        ma_wmape = backtest_moving_average(actuals, 3)

        sku_results = [
            {
                "sku": sku,
                "model": "Linear Regression",
                "prediction": round(float(lr_prediction), 2),
                "wmape": lr_wmape,
                "records_used": len(sku_df),
                "slope": round(float(lr_model.coef_[0]), 2)
            },
            {
                "sku": sku,
                "model": "Naive Forecast",
                "prediction": round(float(naive_prediction), 2),
                "wmape": naive_wmape,
                "records_used": len(sku_df),
                "slope": None
            },
            {
                "sku": sku,
                "model": "3-Month Moving Average",
                "prediction": round(float(ma_prediction), 2),
                "wmape": ma_wmape,
                "records_used": len(sku_df),
                "slope": None
            }
        ]

        model_results.extend(sku_results)

        best_model = min(
            sku_results,
            key=lambda x: x["wmape"] if x["wmape"] is not None else 999999
        )

        best_models.append({
            "sku": sku,
            "best_model": best_model["model"],
            "prediction": best_model["prediction"],
            "wmape": best_model["wmape"]
        })

    return jsonify({
        "status": "success",
        "message": "Backtested model comparison completed",
        "model_results": model_results,
        "best_models": best_models
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
