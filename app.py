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

    skus = df["sku"].unique()

    for sku in skus:

        sku_df = df[df["sku"] == sku].copy()

        sku_df["month_number"] = sku_df["month_number"].astype(float)
        sku_df["actual_units"] = sku_df["actual_units"].astype(float)
        sku_df["forecast_units"] = sku_df["forecast_units"].astype(float)

        next_month = int(sku_df["month_number"].max()) + 1

        # Historical benchmark WMAPE from submitted forecast
        benchmark_wmape = calculate_wmape(
            sku_df["actual_units"],
            sku_df["forecast_units"]
        )

        # Model 1: Linear Regression
        X = sku_df[["month_number"]]
        y = sku_df["actual_units"]

        lr_model = LinearRegression()
        lr_model.fit(X, y)

        lr_prediction = lr_model.predict([[next_month]])[0]

        model_results.append({
            "sku": sku,
            "model": "Linear Regression",
            "prediction": round(float(lr_prediction), 2),
            "wmape": benchmark_wmape,
            "records_used": len(sku_df),
            "slope": round(float(lr_model.coef_[0]), 2)
        })

        # Model 2: Naive Forecast
        naive_prediction = sku_df["actual_units"].iloc[-1]

        model_results.append({
            "sku": sku,
            "model": "Naive Forecast",
            "prediction": round(float(naive_prediction), 2),
            "wmape": benchmark_wmape,
            "records_used": len(sku_df),
            "slope": None
        })

        # Model 3: 3-Month Moving Average
        moving_average_prediction = sku_df["actual_units"].tail(3).mean()

        model_results.append({
            "sku": sku,
            "model": "3-Month Moving Average",
            "prediction": round(float(moving_average_prediction), 2),
            "wmape": benchmark_wmape,
            "records_used": len(sku_df),
            "slope": None
        })

    best_models = []

    for sku in skus:
        sku_models = [m for m in model_results if m["sku"] == sku]

        # For now all models share historical benchmark WMAPE.
        # Later we will calculate model-specific backtest WMAPE.
        best_model = min(sku_models, key=lambda x: x["wmape"])

        best_models.append({
            "sku": sku,
            "best_model": best_model["model"],
            "prediction": best_model["prediction"],
            "wmape": best_model["wmape"]
        })

    return jsonify({
        "status": "success",
        "message": "Model comparison completed",
        "model_results": model_results,
        "best_models": best_models
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
