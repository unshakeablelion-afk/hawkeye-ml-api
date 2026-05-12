from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

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

def calculate_bias(actual, forecast):
    actual = pd.Series(actual).astype(float)
    forecast = pd.Series(forecast).astype(float)

    if actual.sum() == 0:
        return None

    bias = ((forecast - actual).sum() / actual.sum()) * 100

    return round(bias, 2)

def generate_narrative(sku, best_model, wmape, bias, prediction):

    trend_comment = "stable demand pattern"

    if prediction is not None and prediction > 250:
        trend_comment = "strong demand growth"

    if wmape is None:
        accuracy_comment = "forecast accuracy is not yet available"
    elif wmape < 10:
        accuracy_comment = "forecast accuracy is strong"
    elif wmape < 20:
        accuracy_comment = "forecast accuracy is moderate"
    else:
        accuracy_comment = "forecast accuracy needs improvement"

    if bias is None:
        bias_comment = "bias is not yet available"
    elif abs(bias) <= 5:
        bias_comment = "bias is well controlled"
    elif bias > 5:
        bias_comment = "forecast is trending toward overforecasting"
    else:
        bias_comment = "forecast is trending toward underforecasting"

    narrative = (
        f"{sku} shows {trend_comment}. "
        f"{best_model} is currently the best performing model "
        f"with {wmape}% WMAPE and {bias}% bias. "
        f"{accuracy_comment.capitalize()} and {bias_comment}."
    )

    return narrative

def backtest_naive(actuals):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    for i in range(1, len(actuals)):
        forecasts.append(actuals[i - 1])
        actual_test.append(actuals[i])

    wmape = calculate_wmape(actual_test, forecasts)
    bias = calculate_bias(actual_test, forecasts)

    return wmape, bias

def backtest_moving_average(actuals, window=3):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    for i in range(window, len(actuals)):
        forecasts.append(sum(actuals[i-window:i]) / window)
        actual_test.append(actuals[i])

    wmape = calculate_wmape(actual_test, forecasts)
    bias = calculate_bias(actual_test, forecasts)

    return wmape, bias

def backtest_exponential_smoothing(actuals):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    start_index = max(2, len(actuals) - 6)

    for i in range(start_index, len(actuals)):
        train = actuals[:i]

        try:
            model = ExponentialSmoothing(
                train,
                trend=None,
                seasonal=None
            )

            fitted = model.fit()
            prediction = fitted.forecast(1)[0]

            forecasts.append(prediction)
            actual_test.append(actuals[i])

        except Exception:
            continue

    if len(actual_test) == 0:
        return None, None

    wmape = calculate_wmape(actual_test, forecasts)
    bias = calculate_bias(actual_test, forecasts)

    return wmape, bias

def backtest_holt_trend(actuals):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    start_index = max(3, len(actuals) - 6)

    for i in range(start_index, len(actuals)):
        train = actuals[:i]

        try:
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal=None
            )

            fitted = model.fit()
            prediction = fitted.forecast(1)[0]

            forecasts.append(prediction)
            actual_test.append(actuals[i])

        except Exception:
            continue

    if len(actual_test) == 0:
        return None, None

    wmape = calculate_wmape(actual_test, forecasts)
    bias = calculate_bias(actual_test, forecasts)

    return wmape, bias

def backtest_holt_winters(actuals):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    start_index = max(22, len(actuals) - 2)

    for i in range(start_index, len(actuals)):
        train = actuals[:i]

        try:
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=12
            )

            fitted = model.fit()
            prediction = fitted.forecast(1)[0]

            forecasts.append(prediction)
            actual_test.append(actuals[i])

        except Exception:
            continue

    if len(actual_test) == 0:
        return None, None

    wmape = calculate_wmape(actual_test, forecasts)
    bias = calculate_bias(actual_test, forecasts)

    return wmape, bias

def backtest_linear_regression(sku_df):
    forecasts = []
    actual_test = []

    for i in range(2, len(sku_df)):
        train = sku_df.iloc[:i]
        test = sku_df.iloc[i]

        model = LinearRegression()

        model.fit(
            train[["month_number"]],
            train["actual_units"]
        )

        prediction = model.predict([[test["month_number"]]])[0]

        forecasts.append(prediction)
        actual_test.append(test["actual_units"])

    wmape = calculate_wmape(actual_test, forecasts)
    bias = calculate_bias(actual_test, forecasts)

    return wmape, bias

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
    narratives = []

    skus = df["sku"].unique()

    for sku in skus:

        sku_df = df[df["sku"] == sku].copy()

        sku_df["month_number"] = sku_df["month_number"].astype(float)
        sku_df["actual_units"] = sku_df["actual_units"].astype(float)

        sku_df = sku_df.sort_values("month_number")

        next_month = int(sku_df["month_number"].max()) + 1
        actuals = sku_df["actual_units"].tolist()

        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(
            sku_df[["month_number"]],
            sku_df["actual_units"]
        )

        lr_prediction = lr_model.predict([[next_month]])[0]
        lr_wmape, lr_bias = backtest_linear_regression(sku_df)

        # Naive
        naive_prediction = actuals[-1]
        naive_wmape, naive_bias = backtest_naive(actuals)

        # Moving Average
        ma_prediction = sum(actuals[-3:]) / 3
        ma_wmape, ma_bias = backtest_moving_average(actuals, 3)

        # Exponential Smoothing
        try:
            es_model = ExponentialSmoothing(
                actuals,
                trend=None,
                seasonal=None
            )

            es_fitted = es_model.fit()
            es_prediction = es_fitted.forecast(1)[0]

            es_wmape, es_bias = backtest_exponential_smoothing(actuals)

        except Exception:
            es_prediction = None
            es_wmape = None
            es_bias = None

        # Holt Trend
        try:
            holt_model = ExponentialSmoothing(
                actuals,
                trend="add",
                seasonal=None
            )

            holt_fitted = holt_model.fit()
            holt_prediction = holt_fitted.forecast(1)[0]

            holt_wmape, holt_bias = backtest_holt_trend(actuals)

        except Exception:
            holt_prediction = None
            holt_wmape = None
            holt_bias = None

        # Holt-Winters Seasonal
        try:
            if len(actuals) >= 18:
                hw_model = ExponentialSmoothing(
                    actuals,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=12
                )

                hw_fitted = hw_model.fit()
                hw_prediction = hw_fitted.forecast(1)[0]

                hw_wmape, hw_bias = backtest_holt_winters(actuals)

            else:
                hw_prediction = None
                hw_wmape = None
                hw_bias = None

        except Exception:
            hw_prediction = None
            hw_wmape = None
            hw_bias = None

        sku_results = [

            {
                "sku": sku,
                "model": "Linear Regression",
                "prediction": round(float(lr_prediction), 2),
                "wmape": lr_wmape,
                "bias": lr_bias,
                "records_used": len(sku_df),
                "slope": round(float(lr_model.coef_[0]), 2)
            },

            {
                "sku": sku,
                "model": "Naive Forecast",
                "prediction": round(float(naive_prediction), 2),
                "wmape": naive_wmape,
                "bias": naive_bias,
                "records_used": len(sku_df),
                "slope": None
            },

            {
                "sku": sku,
                "model": "3-Month Moving Average",
                "prediction": round(float(ma_prediction), 2),
                "wmape": ma_wmape,
                "bias": ma_bias,
                "records_used": len(sku_df),
                "slope": None
            },

            {
                "sku": sku,
                "model": "Exponential Smoothing",
                "prediction": None if es_prediction is None else round(float(es_prediction), 2),
                "wmape": es_wmape,
                "bias": es_bias,
                "records_used": len(sku_df),
                "slope": None
            },

            {
                "sku": sku,
                "model": "Holt Trend",
                "prediction": None if holt_prediction is None else round(float(holt_prediction), 2),
                "wmape": holt_wmape,
                "bias": holt_bias,
                "records_used": len(sku_df),
                "slope": None
            },

            {
                "sku": sku,
                "model": "Holt-Winters Seasonal",
                "prediction": None if hw_prediction is None else round(float(hw_prediction), 2),
                "wmape": hw_wmape,
                "bias": hw_bias,
                "records_used": len(sku_df),
                "slope": None
            }

        ]

        ranked_results = sorted(
            sku_results,
            key=lambda x: x["wmape"] if x["wmape"] is not None else 999999
        )

        for index, result in enumerate(ranked_results, start=1):
            result["rank"] = index

        model_results.extend(ranked_results)

        best_model = ranked_results[0]

        best_models.append({
            "sku": sku,
            "best_model": best_model["model"],
            "prediction": best_model["prediction"],
            "wmape": best_model["wmape"],
            "bias": best_model["bias"],
            "rank": best_model["rank"]
        })

        narrative = generate_narrative(
            sku,
            best_model["model"],
            best_model["wmape"],
            best_model["bias"],
            best_model["prediction"]
        )

        narratives.append({
            "sku": sku,
            "narrative": narrative
        })

    return jsonify({
        "status": "success",
        "message": "Forecast model comparison completed",
        "model_results": model_results,
        "best_models": best_models,
        "narratives": narratives
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
