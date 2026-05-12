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

def detect_seasonality(actuals, seasonal_periods=12):
    actuals = list(actuals)

    if len(actuals) < seasonal_periods * 2:
        return False, "Insufficient history"

    year_one = actuals[:seasonal_periods]
    year_two = actuals[seasonal_periods:seasonal_periods * 2]

    correlation = pd.Series(year_one).corr(pd.Series(year_two))

    if correlation is not None and correlation >= 0.70:
        return True, f"Seasonality detected with year-over-year pattern correlation of {round(float(correlation), 2)}"

    return False, f"No strong seasonality detected; correlation was {round(float(correlation), 2)}"

def get_trend_factor(actuals, seasonal_periods=12):
    actuals = list(actuals)

    if len(actuals) < seasonal_periods * 2:
        return 1.0

    previous_year = sum(actuals[-24:-12])
    latest_year = sum(actuals[-12:])

    if previous_year == 0:
        return 1.0

    return latest_year / previous_year

def generate_future_months(last_month_label, horizon=12):
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    try:
        month_text, year_text = str(last_month_label).split("-")
        last_month_index = month_names.index(month_text)
        year = int(year_text)
    except Exception:
        last_month_index = 11
        year = 2025

    future_months = []

    for i in range(1, horizon + 1):
        future_index = last_month_index + i
        month_index = future_index % 12
        future_year = year + (future_index // 12)

        future_months.append(f"{month_names[month_index]}-{future_year}")

    return future_months

def generate_forecast_horizon(model_name, actuals, sku_df, horizon=12, seasonal_periods=12):
    actuals = list(actuals)
    forecasts = []

    if len(actuals) == 0:
        return []

    try:
        if model_name == "Naive Forecast":
            forecasts = [actuals[-1]] * horizon

        elif model_name == "Seasonal Naive Forecast":
            if len(actuals) >= seasonal_periods:
                seasonal_base = actuals[-seasonal_periods:]
                forecasts = [seasonal_base[i % seasonal_periods] for i in range(horizon)]
            else:
                forecasts = [actuals[-1]] * horizon

        elif model_name == "Trend-Adjusted Seasonal Naive":
            if len(actuals) >= seasonal_periods * 2:
                trend_factor = get_trend_factor(actuals, seasonal_periods)
                seasonal_base = actuals[-seasonal_periods:]
                forecasts = [
                    seasonal_base[i % seasonal_periods] * trend_factor
                    for i in range(horizon)
                ]
            else:
                forecasts = [actuals[-1]] * horizon

        elif model_name == "Linear Regression":
            lr_model = LinearRegression()
            lr_model.fit(sku_df[["month_number"]], sku_df["actual_units"])

            max_month = int(sku_df["month_number"].max())
            future_month_numbers = [[max_month + i] for i in range(1, horizon + 1)]

            forecasts = lr_model.predict(future_month_numbers).tolist()

        elif model_name == "3-Month Moving Average":
            rolling_values = list(actuals)

            for _ in range(horizon):
                prediction = sum(rolling_values[-3:]) / 3
                forecasts.append(prediction)
                rolling_values.append(prediction)

        elif model_name == "Exponential Smoothing":
            model = ExponentialSmoothing(
                actuals,
                trend=None,
                seasonal=None
            )

            fitted = model.fit()
            forecasts = fitted.forecast(horizon).tolist()

        elif model_name == "Holt Trend":
            model = ExponentialSmoothing(
                actuals,
                trend="add",
                seasonal=None
            )

            fitted = model.fit()
            forecasts = fitted.forecast(horizon).tolist()

        elif model_name == "Holt-Winters Seasonal":
            if len(actuals) >= 24:
                model = ExponentialSmoothing(
                    actuals,
                    trend=None,
                    seasonal="add",
                    seasonal_periods=12
                )

                fitted = model.fit()
                forecasts = fitted.forecast(horizon).tolist()
            else:
                forecasts = [actuals[-1]] * horizon

        else:
            forecasts = [actuals[-1]] * horizon

    except Exception:
        forecasts = [actuals[-1]] * horizon

    return [round(float(value), 2) for value in forecasts]

def generate_narrative(sku, best_model, wmape, bias, prediction, demand_pattern):

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

    return (
        f"{sku} is classified as {demand_pattern}. "
        f"{sku} shows {trend_comment}. "
        f"{best_model} is currently the best performing model "
        f"with {wmape}% WMAPE and {bias}% bias. "
        f"{accuracy_comment.capitalize()} and {bias_comment}."
    )

def backtest_naive(actuals):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    for i in range(1, len(actuals)):
        forecasts.append(actuals[i - 1])
        actual_test.append(actuals[i])

    return calculate_wmape(actual_test, forecasts), calculate_bias(actual_test, forecasts)

def backtest_seasonal_naive(actuals, seasonal_periods=12):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    if len(actuals) <= seasonal_periods:
        return None, None

    for i in range(seasonal_periods, len(actuals)):
        forecasts.append(actuals[i - seasonal_periods])
        actual_test.append(actuals[i])

    return calculate_wmape(actual_test, forecasts), calculate_bias(actual_test, forecasts)

def backtest_trend_adjusted_seasonal_naive(actuals, seasonal_periods=12):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    if len(actuals) < seasonal_periods * 2:
        return None, None

    year_one = actuals[:seasonal_periods]
    year_two = actuals[seasonal_periods:seasonal_periods * 2]

    year_one_total = sum(year_one)
    year_two_total = sum(year_two)

    if year_one_total == 0:
        trend_factor = 1.0
    else:
        trend_factor = year_two_total / year_one_total

    for i in range(seasonal_periods, seasonal_periods * 2):
        historical_same_month = actuals[i - seasonal_periods]
        prediction = historical_same_month * trend_factor

        forecasts.append(prediction)
        actual_test.append(actuals[i])

    return calculate_wmape(actual_test, forecasts), calculate_bias(actual_test, forecasts)

def backtest_moving_average(actuals, window=3):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    for i in range(window, len(actuals)):
        forecasts.append(sum(actuals[i-window:i]) / window)
        actual_test.append(actuals[i])

    return calculate_wmape(actual_test, forecasts), calculate_bias(actual_test, forecasts)

def backtest_exponential_smoothing(actuals):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    start_index = max(2, len(actuals) - 6)

    for i in range(start_index, len(actuals)):
        train = actuals[:i]

        try:
            model = ExponentialSmoothing(train, trend=None, seasonal=None)
            fitted = model.fit()
            prediction = fitted.forecast(1)[0]

            forecasts.append(prediction)
            actual_test.append(actuals[i])

        except Exception:
            continue

    if len(actual_test) == 0:
        return None, None

    return calculate_wmape(actual_test, forecasts), calculate_bias(actual_test, forecasts)

def backtest_holt_trend(actuals):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    start_index = max(3, len(actuals) - 6)

    for i in range(start_index, len(actuals)):
        train = actuals[:i]

        try:
            model = ExponentialSmoothing(train, trend="add", seasonal=None)
            fitted = model.fit()
            prediction = fitted.forecast(1)[0]

            forecasts.append(prediction)
            actual_test.append(actuals[i])

        except Exception:
            continue

    if len(actual_test) == 0:
        return None, None

    return calculate_wmape(actual_test, forecasts), calculate_bias(actual_test, forecasts)

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
                trend=None,
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

    return calculate_wmape(actual_test, forecasts), calculate_bias(actual_test, forecasts)

def backtest_linear_regression(sku_df):
    forecasts = []
    actual_test = []

    for i in range(2, len(sku_df)):
        train = sku_df.iloc[:i]
        test = sku_df.iloc[i]

        model = LinearRegression()
        model.fit(train[["month_number"]], train["actual_units"])

        prediction = model.predict([[test["month_number"]]])[0]

        forecasts.append(prediction)
        actual_test.append(test["actual_units"])

    return calculate_wmape(actual_test, forecasts), calculate_bias(actual_test, forecasts)

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
    demand_patterns = []
    forecast_horizons = []

    skus = df["sku"].unique()

    for sku in skus:

        sku_df = df[df["sku"] == sku].copy()

        sku_df["month_number"] = sku_df["month_number"].astype(float)
        sku_df["actual_units"] = sku_df["actual_units"].astype(float)

        sku_df = sku_df.sort_values("month_number")

        next_month = int(sku_df["month_number"].max()) + 1
        actuals = sku_df["actual_units"].tolist()

        last_month_label = sku_df.iloc[-1]["month"]
        future_months = generate_future_months(last_month_label, 12)

        seasonality_detected, seasonality_reason = detect_seasonality(actuals)
        demand_pattern = "Seasonal" if seasonality_detected else "Non-seasonal / trend-stable"

        demand_patterns.append({
            "sku": sku,
            "demand_pattern": demand_pattern,
            "seasonality_detected": seasonality_detected,
            "reason": seasonality_reason
        })

        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(sku_df[["month_number"]], sku_df["actual_units"])

        lr_prediction = lr_model.predict([[next_month]])[0]
        lr_wmape, lr_bias = backtest_linear_regression(sku_df)

        # Naive
        naive_prediction = actuals[-1]
        naive_wmape, naive_bias = backtest_naive(actuals)

        # Seasonal Naive
        if len(actuals) >= 12:
            seasonal_naive_prediction = actuals[-12]
            seasonal_naive_wmape, seasonal_naive_bias = backtest_seasonal_naive(actuals, 12)
        else:
            seasonal_naive_prediction = None
            seasonal_naive_wmape = None
            seasonal_naive_bias = None

        # Trend-Adjusted Seasonal Naive
        if len(actuals) >= 24:
            trend_factor = get_trend_factor(actuals, 12)
            trend_adjusted_seasonal_naive_prediction = actuals[-12] * trend_factor

            tasn_wmape, tasn_bias = backtest_trend_adjusted_seasonal_naive(actuals, 12)
        else:
            trend_factor = None
            trend_adjusted_seasonal_naive_prediction = None
            tasn_wmape = None
            tasn_bias = None

        # Moving Average
        ma_prediction = sum(actuals[-3:]) / 3
        ma_wmape, ma_bias = backtest_moving_average(actuals, 3)

        # Exponential Smoothing
        try:
            es_model = ExponentialSmoothing(actuals, trend=None, seasonal=None)
            es_fitted = es_model.fit()
            es_prediction = es_fitted.forecast(1)[0]

            es_wmape, es_bias = backtest_exponential_smoothing(actuals)

        except Exception:
            es_prediction = None
            es_wmape = None
            es_bias = None

        # Holt Trend
        try:
            holt_model = ExponentialSmoothing(actuals, trend="add", seasonal=None)
            holt_fitted = holt_model.fit()
            holt_prediction = holt_fitted.forecast(1)[0]

            holt_wmape, holt_bias = backtest_holt_trend(actuals)

        except Exception:
            holt_prediction = None
            holt_wmape = None
            holt_bias = None

        # Holt-Winters Seasonal
        try:
            if len(actuals) >= 24:
                hw_model = ExponentialSmoothing(
                    actuals,
                    trend=None,
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
                "slope": round(float(lr_model.coef_[0]), 2),
                "demand_pattern": demand_pattern
            },
            {
                "sku": sku,
                "model": "Naive Forecast",
                "prediction": round(float(naive_prediction), 2),
                "wmape": naive_wmape,
                "bias": naive_bias,
                "records_used": len(sku_df),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": sku,
                "model": "Seasonal Naive Forecast",
                "prediction": None if seasonal_naive_prediction is None else round(float(seasonal_naive_prediction), 2),
                "wmape": seasonal_naive_wmape,
                "bias": seasonal_naive_bias,
                "records_used": len(sku_df),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": sku,
                "model": "Trend-Adjusted Seasonal Naive",
                "prediction": None if trend_adjusted_seasonal_naive_prediction is None else round(float(trend_adjusted_seasonal_naive_prediction), 2),
                "wmape": tasn_wmape,
                "bias": tasn_bias,
                "records_used": len(sku_df),
                "slope": None,
                "demand_pattern": demand_pattern,
                "trend_factor": None if trend_factor is None else round(float(trend_factor), 3)
            },
            {
                "sku": sku,
                "model": "3-Month Moving Average",
                "prediction": round(float(ma_prediction), 2),
                "wmape": ma_wmape,
                "bias": ma_bias,
                "records_used": len(sku_df),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": sku,
                "model": "Exponential Smoothing",
                "prediction": None if es_prediction is None else round(float(es_prediction), 2),
                "wmape": es_wmape,
                "bias": es_bias,
                "records_used": len(sku_df),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": sku,
                "model": "Holt Trend",
                "prediction": None if holt_prediction is None else round(float(holt_prediction), 2),
                "wmape": holt_wmape,
                "bias": holt_bias,
                "records_used": len(sku_df),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": sku,
                "model": "Holt-Winters Seasonal",
                "prediction": None if hw_prediction is None else round(float(hw_prediction), 2),
                "wmape": hw_wmape,
                "bias": hw_bias,
                "records_used": len(sku_df),
                "slope": None,
                "demand_pattern": demand_pattern
            }
        ]

        if seasonality_detected:
            ranked_candidates = sku_results
        else:
            ranked_candidates = [
                result for result in sku_results
                if result["model"] not in [
                    "Seasonal Naive Forecast",
                    "Trend-Adjusted Seasonal Naive",
                    "Holt-Winters Seasonal"
                ]
            ]

        ranked_results = sorted(
            ranked_candidates,
            key=lambda x: x["wmape"] if x["wmape"] is not None else 999999
        )

        unranked_results = [
            result for result in sku_results
            if result not in ranked_results
        ]

        for index, result in enumerate(ranked_results, start=1):
            result["rank"] = index

        for result in unranked_results:
            result["rank"] = "-"

        model_results.extend(ranked_results + unranked_results)

        best_model = ranked_results[0]

        best_models.append({
            "sku": sku,
            "best_model": best_model["model"],
            "prediction": best_model["prediction"],
            "wmape": best_model["wmape"],
            "bias": best_model["bias"],
            "rank": best_model["rank"],
            "demand_pattern": demand_pattern
        })

        horizon_values = generate_forecast_horizon(
            best_model["model"],
            actuals,
            sku_df,
            horizon=12,
            seasonal_periods=12
        )

        horizon_rows = []

        for index, value in enumerate(horizon_values):
            horizon_rows.append({
                "month": future_months[index],
                "forecast": value
            })

        forecast_horizons.append({
            "sku": sku,
            "model": best_model["model"],
            "forecast": horizon_rows
        })

        narrative = generate_narrative(
            sku,
            best_model["model"],
            best_model["wmape"],
            best_model["bias"],
            best_model["prediction"],
            demand_pattern
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
        "narratives": narratives,
        "demand_patterns": demand_patterns,
        "forecast_horizons": forecast_horizons
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
