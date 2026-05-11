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

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "records" not in data:
        return jsonify({
            "status": "error",
            "message": "No records provided"
        }), 400

    df = pd.DataFrame(data["records"])

    forecasts = []

    skus = df["sku"].unique()

    for sku in skus:

        sku_df = df[df["sku"] == sku].copy()

        X = sku_df[["month_number"]].astype(float)
        y = sku_df["actual_units"].astype(float)

        model = LinearRegression()
        model.fit(X, y)

        next_month = int(sku_df["month_number"].astype(int).max()) + 1
        prediction = model.predict([[next_month]])[0]

        actual_units = sku_df["actual_units"].astype(float)
        forecast_units = sku_df["forecast_units"].astype(float)

        absolute_error = abs(actual_units - forecast_units)

        wmape = (absolute_error.sum() / actual_units.sum()) * 100

        forecasts.append({
            "sku": sku,
            "prediction": round(float(prediction), 2),
            "records_used": len(sku_df),
            "slope": round(float(model.coef_[0]), 2),
            "wmape": round(float(wmape), 2)
        })

    return jsonify({
        "status": "success",
        "model_type": "Linear Regression",
        "forecast_count": len(forecasts),
        "forecasts": forecasts
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)