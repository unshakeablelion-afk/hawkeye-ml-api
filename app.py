from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
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

    if "month_number" not in df.columns or "actual_units" not in df.columns:
        return jsonify({
            "status": "error",
            "message": "Required fields: month_number, actual_units"
        }), 400

    X = df[["month_number"]].astype(float)
    y = df["actual_units"].astype(float)

    model = LinearRegression()
    model.fit(X, y)

    next_month = int(df["month_number"].max()) + 1
    prediction = model.predict([[next_month]])[0]

    return jsonify({
        "status": "success",
        "model_type": "Linear Regression",
        "next_month_number": next_month,
        "prediction": round(float(prediction), 2),
        "records_used": len(df),
        "slope": round(float(model.coef_[0]), 2),
        "intercept": round(float(model.intercept_), 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
