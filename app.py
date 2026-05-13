from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
import pandas as pd
import math
import os
import warnings
import mysql.connector

warnings.filterwarnings("ignore")

app = Flask(__name__)


def clean_for_json(value):
    if isinstance(value, dict):
        return {key: clean_for_json(val) for key, val in value.items()}

    if isinstance(value, list):
        return [clean_for_json(item) for item in value]

    if isinstance(value, tuple):
        return [clean_for_json(item) for item in value]

    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    return value


@app.errorhandler(Exception)
def handle_exception(error):
    return jsonify(clean_for_json({
        "status": "error",
        "message": str(error),
        "error_type": type(error).__name__
    })), 500


RF_FEATURES = [
    "month_number",
    "month_of_year",
    "quarter",
    "lag_1",
    "lag_3_avg",
    "lag_6",
    "lag_6_avg",
    "lag_12",
    "year_over_year_growth",
    "recent_3_month_growth",
    "peak_month_flag",
    "post_peak_flag"
]
DB_CONFIG = {
    "host": os.environ.get("DB_HOST"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD"),
    "database": os.environ.get("DB_NAME")
}


def get_db_connection():
    return mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"]
    )
    
@app.route("/")
def home():
    return jsonify({
        "status": "success",
        "message": "HawkEye ML API is running"
    })

@app.route("/test-db")
def test_db():

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("SELECT 1")
        result = cursor.fetchone()

        cursor.close()
        connection.close()

        return jsonify({
            "status": "success",
            "database_connected": True,
            "result": result[0]
        })

    except Exception as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


@app.route("/test-tables")
def test_tables():

    try:
        connection = get_db_connection()
        cursor = connection.cursor()

        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        cursor.close()
        connection.close()

        table_names = [table[0] for table in tables]

        return jsonify({
            "status": "success",
            "tables": table_names
        })

    except Exception as error:
        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


def save_forecast_run(run_name, sku_count):

    connection = get_db_connection()
    cursor = connection.cursor()

    sql = """
    INSERT INTO forecast_runs
    (run_name, sku_count)
    VALUES (%s, %s)
    """

    cursor.execute(sql, (
        run_name,
        sku_count
    ))

    connection.commit()

    run_id = cursor.lastrowid

    cursor.close()
    connection.close()

    return run_id

def save_model_result(run_id, sku, model_name, prediction, wmape, bias, rank_value):

    connection = get_db_connection()
    cursor = connection.cursor()

    sql = """
    INSERT INTO forecast_model_results
    (run_id, sku, model_name, prediction, wmape, bias, rank_value)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    cursor.execute(sql, (
        run_id,
        sku,
        model_name,
        prediction,
        wmape,
        bias,
        str(rank_value)
    ))

    connection.commit()

    cursor.close()
    connection.close()

def calculate_wmape(actual, forecast):
    actual = pd.Series(actual).astype(float)
    forecast = pd.Series(forecast).astype(float)

    if actual.sum() == 0:
        return None

    result = (abs(actual - forecast).sum() / actual.sum()) * 100
    return round(float(result), 2)


def calculate_bias(actual, forecast):
    actual = pd.Series(actual).astype(float)
    forecast = pd.Series(forecast).astype(float)

    if actual.sum() == 0:
        return None

    bias = ((forecast - actual).sum() / actual.sum()) * 100
    return round(float(bias), 2)


def get_error_factor(wmape):
    if wmape is None:
        return 0.15

    return max(float(wmape) / 100, 0.08)


def detect_seasonality(actuals, seasonal_periods=12):
    actuals = list(actuals)

    if len(actuals) < seasonal_periods * 2:
        return False, "Insufficient history"

    year_one = actuals[:seasonal_periods]
    year_two = actuals[seasonal_periods:seasonal_periods * 2]

    correlation = pd.Series(year_one).corr(pd.Series(year_two))

    if correlation is None or pd.isna(correlation):
        return False, "No strong seasonality detected; correlation could not be calculated"

    correlation_value = float(correlation)

    if correlation_value >= 0.70:
        return True, f"Seasonality detected with year-over-year pattern correlation of {round(correlation_value, 2)}"

    return False, f"No strong seasonality detected; correlation was {round(correlation_value, 2)}"


def get_trend_factor(actuals, seasonal_periods=12):
    actuals = list(actuals)

    if len(actuals) < seasonal_periods * 2:
        return 1.0

    previous_year = sum(actuals[-24:-12])
    latest_year = sum(actuals[-12:])

    if previous_year == 0:
        return 1.0

    return float(latest_year / previous_year)


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


def get_peak_months(actuals):
    actuals = list(actuals)

    if len(actuals) < 12:
        return []

    rows = []

    for i, value in enumerate(actuals):
        month_number = i + 1
        month_of_year = ((month_number - 1) % 12) + 1

        rows.append({
            "month_of_year": month_of_year,
            "actual_units": float(value)
        })

    df = pd.DataFrame(rows)

    monthly_avg = (
        df.groupby("month_of_year")["actual_units"]
        .mean()
        .sort_values(ascending=False)
    )

    return list(monthly_avg.head(2).index)


def safe_growth(current_value, prior_value):
    if prior_value is None or prior_value == 0:
        return 0

    return float((current_value - prior_value) / prior_value)


def build_ml_features_from_actuals(actuals):
    rows = []
    actuals = list(actuals)
    peak_months = get_peak_months(actuals)

    for i in range(len(actuals)):
        month_number = i + 1
        month_of_year = ((month_number - 1) % 12) + 1
        quarter = ((month_of_year - 1) // 3) + 1

        previous_month_of_year = 12 if month_of_year == 1 else month_of_year - 1

        lag_1 = actuals[i - 1] if i >= 1 else None
        lag_3_avg = sum(actuals[i - 3:i]) / 3 if i >= 3 else None
        lag_6 = actuals[i - 6] if i >= 6 else None
        lag_6_avg = sum(actuals[i - 6:i]) / 6 if i >= 6 else None
        lag_12 = actuals[i - 12] if i >= 12 else None

        year_over_year_growth = None
        if i >= 12 and lag_12 is not None:
            year_over_year_growth = safe_growth(actuals[i], lag_12)

        recent_3_month_growth = None
        if i >= 6:
            recent_3_avg = sum(actuals[i - 3:i]) / 3
            prior_3_avg = sum(actuals[i - 6:i - 3]) / 3
            recent_3_month_growth = safe_growth(recent_3_avg, prior_3_avg)

        peak_month_flag = 1 if month_of_year in peak_months else 0
        post_peak_flag = 1 if previous_month_of_year in peak_months else 0

        rows.append({
            "month_number": month_number,
            "month_of_year": month_of_year,
            "quarter": quarter,
            "lag_1": lag_1,
            "lag_3_avg": lag_3_avg,
            "lag_6": lag_6,
            "lag_6_avg": lag_6_avg,
            "lag_12": lag_12,
            "year_over_year_growth": year_over_year_growth,
            "recent_3_month_growth": recent_3_month_growth,
            "peak_month_flag": peak_month_flag,
            "post_peak_flag": post_peak_flag,
            "actual_units": actuals[i]
        })

    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.dropna()

    return feature_df


def build_next_ml_features(actuals):
    actuals = list(actuals)
    peak_months = get_peak_months(actuals)

    next_month_number = len(actuals) + 1
    next_month_of_year = ((next_month_number - 1) % 12) + 1
    next_quarter = ((next_month_of_year - 1) // 3) + 1
    previous_month_of_year = 12 if next_month_of_year == 1 else next_month_of_year - 1

    lag_1 = actuals[-1]
    lag_3_avg = sum(actuals[-3:]) / 3
    lag_6 = actuals[-6]
    lag_6_avg = sum(actuals[-6:]) / 6
    lag_12 = actuals[-12]

    recent_3_avg = sum(actuals[-3:]) / 3
    prior_3_avg = sum(actuals[-6:-3]) / 3

    year_over_year_growth = safe_growth(lag_1, actuals[-13]) if len(actuals) >= 13 else 0
    recent_3_month_growth = safe_growth(recent_3_avg, prior_3_avg)

    peak_month_flag = 1 if next_month_of_year in peak_months else 0
    post_peak_flag = 1 if previous_month_of_year in peak_months else 0

    return pd.DataFrame([{
        "month_number": next_month_number,
        "month_of_year": next_month_of_year,
        "quarter": next_quarter,
        "lag_1": lag_1,
        "lag_3_avg": lag_3_avg,
        "lag_6": lag_6,
        "lag_6_avg": lag_6_avg,
        "lag_12": lag_12,
        "year_over_year_growth": year_over_year_growth,
        "recent_3_month_growth": recent_3_month_growth,
        "peak_month_flag": peak_month_flag,
        "post_peak_flag": post_peak_flag
    }])


def get_random_forest_feature_importance(actuals):
    feature_df = build_ml_features_from_actuals(actuals)

    if len(feature_df) < 6:
        return []

    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        min_samples_leaf=1
    )

    model.fit(feature_df[RF_FEATURES], feature_df["actual_units"])

    importance_rows = []

    for feature, importance in zip(RF_FEATURES, model.feature_importances_):
        importance_rows.append({
            "feature": feature,
            "importance": round(float(importance) * 100, 2)
        })

    return sorted(importance_rows, key=lambda x: x["importance"], reverse=True)


def generate_forecast_explanation(sku, best_model, demand_pattern, feature_rows):
    top_features = feature_rows[:4] if feature_rows else []

    driver_comments = []

    feature_comment_map = {
        "lag_12": "same month last year is strongly influencing the forecast, which points to seasonal repetition",
        "lag_6": "mid-year demand behavior is influencing the forecast",
        "lag_6_avg": "the recent six-month average is shaping the forecast level",
        "lag_3_avg": "recent three-month demand momentum is influencing the forecast",
        "lag_1": "the most recent month is heavily influencing the forecast",
        "month_of_year": "calendar month seasonality is influencing the forecast",
        "quarter": "quarterly seasonality is influencing the forecast",
        "year_over_year_growth": "year-over-year growth is influencing the forecast direction",
        "recent_3_month_growth": "recent growth or slowdown is affecting the forecast",
        "peak_month_flag": "the model recognizes this SKU has recurring peak-month behavior",
        "post_peak_flag": "the model is accounting for demand behavior after a peak period",
        "month_number": "the model is using the overall time trend"
    }

    for row in top_features:
        feature = row["feature"]
        importance = row["importance"]

        driver_comments.append({
            "feature": feature,
            "importance": importance,
            "interpretation": feature_comment_map.get(
                feature,
                "this feature is influencing the forecast"
            )
        })

    if len(top_features) == 0:
        summary = (
            f"{sku} does not have enough usable machine learning feature history "
            f"to generate a strong driver explanation."
        )
    else:
        main_driver = top_features[0]["feature"]

        summary = (
            f"{sku} is classified as {demand_pattern}. "
            f"The selected model is {best_model}. "
            f"The strongest machine-learning driver is {main_driver}, "
            f"meaning the forecast is being shaped mainly by "
            f"{feature_comment_map.get(main_driver, 'the strongest available signal')}."
        )

    return {
        "sku": sku,
        "model": best_model,
        "demand_pattern": demand_pattern,
        "summary": summary,
        "drivers": driver_comments
    }


def predict_random_forest_next(actuals):
    feature_df = build_ml_features_from_actuals(actuals)

    if len(feature_df) < 6 or len(actuals) < 13:
        return None

    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        min_samples_leaf=1
    )

    model.fit(feature_df[RF_FEATURES], feature_df["actual_units"])

    next_features = build_next_ml_features(actuals)

    prediction = model.predict(next_features[RF_FEATURES])[0]

    return float(prediction)


def predict_xgboost_next(actuals):
    feature_df = build_ml_features_from_actuals(actuals)

    if len(feature_df) < 6 or len(actuals) < 13:
        return None

    model = XGBRegressor(
        n_estimators=75,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(feature_df[RF_FEATURES], feature_df["actual_units"])

    next_features = build_next_ml_features(actuals)

    prediction = model.predict(next_features[RF_FEATURES])[0]

    return float(prediction)


def backtest_random_forest(actuals):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    if len(actuals) < 18:
        return None, None

    start_index = max(15, len(actuals) - 4)

    for i in range(start_index, len(actuals)):
        train_actuals = actuals[:i]

        try:
            prediction = predict_random_forest_next(train_actuals)

            if prediction is None:
                continue

            forecasts.append(prediction)
            actual_test.append(actuals[i])

        except Exception:
            continue

    if len(actual_test) == 0:
        return None, None

    return calculate_wmape(actual_test, forecasts), calculate_bias(actual_test, forecasts)


def backtest_xgboost(actuals):
    actuals = list(actuals)
    forecasts = []
    actual_test = []

    if len(actuals) < 18:
        return None, None

    start_index = max(15, len(actuals) - 4)

    for i in range(start_index, len(actuals)):
        train_actuals = actuals[:i]

        try:
            prediction = predict_xgboost_next(train_actuals)

            if prediction is None:
                continue

            forecasts.append(prediction)
            actual_test.append(actuals[i])

        except Exception:
            continue

    if len(actual_test) == 0:
        return None, None

    return calculate_wmape(actual_test, forecasts), calculate_bias(actual_test, forecasts)


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
                forecasts = [seasonal_base[i % seasonal_periods] * trend_factor for i in range(horizon)]
            else:
                forecasts = [actuals[-1]] * horizon

        elif model_name == "Random Forest Forecast":
            rolling_actuals = list(actuals)

            for _ in range(horizon):
                prediction = predict_random_forest_next(rolling_actuals)

                if prediction is None:
                    prediction = rolling_actuals[-1]

                forecasts.append(prediction)
                rolling_actuals.append(prediction)

        elif model_name == "XGBoost Forecast":
            rolling_actuals = list(actuals)

            for _ in range(horizon):
                prediction = predict_xgboost_next(rolling_actuals)

                if prediction is None:
                    prediction = rolling_actuals[-1]

                forecasts.append(prediction)
                rolling_actuals.append(prediction)

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
            model = ExponentialSmoothing(actuals, trend=None, seasonal=None)
            fitted = model.fit()
            forecasts = fitted.forecast(horizon).tolist()

        elif model_name == "Holt Trend":
            model = ExponentialSmoothing(actuals, trend="add", seasonal=None)
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


def build_horizon_rows(months, values):
    rows = []

    for index, value in enumerate(values):
        rows.append({
            "month": str(months[index]),
            "forecast": round(float(value), 2)
        })

    return rows


def get_model_residuals(model_name, sku_df, actuals, test_periods=12):
    actuals = list(actuals)
    residuals = []

    if len(actuals) < 12:
        return []

    start_index = max(6, len(actuals) - test_periods)

    for i in range(start_index, len(actuals)):
        train_actuals = actuals[:i]
        train_sku_df = sku_df.iloc[:i].copy()
        actual_value = float(actuals[i])

        try:
            forecast_values = generate_forecast_horizon(
                model_name,
                train_actuals,
                train_sku_df,
                horizon=1,
                seasonal_periods=12
            )

            if len(forecast_values) == 0:
                continue

            forecast_value = float(forecast_values[0])
            residual = actual_value - forecast_value
            residuals.append(float(residual))

        except Exception:
            continue

    return residuals


def build_forecast_range_rows(months, values, wmape, residuals=None):
    rows = []

    if residuals and len(residuals) >= 3:
        residual_series = pd.Series(residuals).astype(float)

        p10_residual = float(residual_series.quantile(0.10))
        p90_residual = float(residual_series.quantile(0.90))

        for index, value in enumerate(values):
            p50 = float(value)
            p10 = max(0, p50 + p10_residual)
            p90 = max(0, p50 + p90_residual)

            rows.append({
                "month": str(months[index]),
                "p10": round(float(p10), 2),
                "p50": round(float(p50), 2),
                "p90": round(float(p90), 2),
                "range_method": "Residual percentile"
            })

    else:
        error_factor = float(get_error_factor(wmape))

        for index, value in enumerate(values):
            p50 = float(value)
            p10 = max(0, p50 * (1 - error_factor))
            p90 = p50 * (1 + error_factor)

            rows.append({
                "month": str(months[index]),
                "p10": round(float(p10), 2),
                "p50": round(float(p50), 2),
                "p90": round(float(p90), 2),
                "range_method": "WMAPE fallback"
            })

    return rows


def backtest_forecast_range_reliability(model_name, sku_df, actuals, wmape, residuals=None, test_periods=6):
    actuals = list(actuals)

    if len(actuals) < 12:
        return {
            "coverage": None,
            "months_tested": 0,
            "inside_count": 0,
            "outside_count": 0,
            "status": "Insufficient history",
            "details": []
        }

    start_index = max(6, len(actuals) - test_periods)

    details = []
    inside_count = 0
    outside_count = 0

    if residuals and len(residuals) >= 3:
        residual_series = pd.Series(residuals).astype(float)
        p10_residual = float(residual_series.quantile(0.10))
        p90_residual = float(residual_series.quantile(0.90))
    else:
        p10_residual = None
        p90_residual = None

    for i in range(start_index, len(actuals)):
        train_actuals = actuals[:i]
        train_sku_df = sku_df.iloc[:i].copy()
        actual_value = float(actuals[i])
        month_label = str(sku_df.iloc[i]["month"])

        try:
            forecast_values = generate_forecast_horizon(
                model_name,
                train_actuals,
                train_sku_df,
                horizon=1,
                seasonal_periods=12
            )

            if len(forecast_values) == 0:
                continue

            p50 = float(forecast_values[0])

            if p10_residual is not None and p90_residual is not None:
                p10 = max(0, p50 + p10_residual)
                p90 = max(0, p50 + p90_residual)
                range_method = "Residual percentile"
            else:
                error_factor = float(get_error_factor(wmape))
                p10 = max(0, p50 * (1 - error_factor))
                p90 = p50 * (1 + error_factor)
                range_method = "WMAPE fallback"

            inside_range = bool(p10 <= actual_value <= p90)

            if inside_range:
                inside_count += 1
            else:
                outside_count += 1

            details.append({
                "month": month_label,
                "actual": round(float(actual_value), 2),
                "p10": round(float(p10), 2),
                "p50": round(float(p50), 2),
                "p90": round(float(p90), 2),
                "inside_range": inside_range,
                "range_method": range_method
            })

        except Exception:
            continue

    months_tested = len(details)

    if months_tested == 0:
        coverage = None
        status = "No valid reliability test"
    else:
        coverage = round(float((inside_count / months_tested) * 100), 2)

        if coverage < 70:
            status = "Too narrow / high risk"
        elif coverage <= 90:
            status = "Good"
        else:
            status = "Conservative / possibly too wide"

    return {
        "coverage": coverage,
        "months_tested": int(months_tested),
        "inside_count": int(inside_count),
        "outside_count": int(outside_count),
        "status": status,
        "details": details
    }


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

            forecasts.append(float(prediction))
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

            forecasts.append(float(prediction))
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

            forecasts.append(float(prediction))
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

        forecasts.append(float(prediction))
        actual_test.append(float(test["actual_units"]))

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

    required_columns = ["sku", "month", "month_number", "actual_units"]

    for column in required_columns:
        if column not in df.columns:
            return jsonify({
                "status": "error",
                "message": f"Missing required column: {column}"
            }), 400

    model_results = []
    best_models = []
    narratives = []
    demand_patterns = []
    forecast_horizons = []
    random_forest_feature_importance = []
    forecast_explanations = []
    forecast_range_reliability = []

    skus = df["sku"].unique()
    run_name = f"Forecast Run - {pd.Timestamp.now()}"

    run_id = save_forecast_run(
        run_name=run_name,
        sku_count=len(skus)
    )

    for sku in skus:
        sku_df = df[df["sku"] == sku].copy()

        sku_df["month_number"] = sku_df["month_number"].astype(float)
        sku_df["actual_units"] = sku_df["actual_units"].astype(float)

        sku_df = sku_df.sort_values("month_number").reset_index(drop=True)

        next_month = int(sku_df["month_number"].max()) + 1
        actuals = sku_df["actual_units"].tolist()

        last_month_label = sku_df.iloc[-1]["month"]
        future_months = generate_future_months(last_month_label, 12)

        seasonality_detected, seasonality_reason = detect_seasonality(actuals)
        demand_pattern = "Seasonal" if seasonality_detected else "Non-seasonal / trend-stable"

        demand_patterns.append({
            "sku": str(sku),
            "demand_pattern": demand_pattern,
            "seasonality_detected": bool(seasonality_detected),
            "reason": str(seasonality_reason)
        })

        rf_importance = get_random_forest_feature_importance(actuals)

        random_forest_feature_importance.append({
            "sku": str(sku),
            "features": rf_importance
        })

        lr_model = LinearRegression()
        lr_model.fit(sku_df[["month_number"]], sku_df["actual_units"])

        lr_prediction = lr_model.predict([[next_month]])[0]
        lr_wmape, lr_bias = backtest_linear_regression(sku_df)

        try:
            rf_prediction = predict_random_forest_next(actuals)
            rf_wmape, rf_bias = backtest_random_forest(actuals)
        except Exception:
            rf_prediction = None
            rf_wmape = None
            rf_bias = None

        try:
            xgb_prediction = predict_xgboost_next(actuals)
            xgb_wmape, xgb_bias = backtest_xgboost(actuals)
        except Exception:
            xgb_prediction = None
            xgb_wmape = None
            xgb_bias = None

        naive_prediction = actuals[-1]
        naive_wmape, naive_bias = backtest_naive(actuals)

        if len(actuals) >= 12:
            seasonal_naive_prediction = actuals[-12]
            seasonal_naive_wmape, seasonal_naive_bias = backtest_seasonal_naive(actuals, 12)
        else:
            seasonal_naive_prediction = None
            seasonal_naive_wmape = None
            seasonal_naive_bias = None

        if len(actuals) >= 24:
            trend_factor = get_trend_factor(actuals, 12)
            trend_adjusted_seasonal_naive_prediction = actuals[-12] * trend_factor
            tasn_wmape, tasn_bias = backtest_trend_adjusted_seasonal_naive(actuals, 12)
        else:
            trend_factor = None
            trend_adjusted_seasonal_naive_prediction = None
            tasn_wmape = None
            tasn_bias = None

        ma_prediction = sum(actuals[-3:]) / 3
        ma_wmape, ma_bias = backtest_moving_average(actuals, 3)

        try:
            es_model = ExponentialSmoothing(actuals, trend=None, seasonal=None)
            es_fitted = es_model.fit()
            es_prediction = es_fitted.forecast(1)[0]
            es_wmape, es_bias = backtest_exponential_smoothing(actuals)
        except Exception:
            es_prediction = None
            es_wmape = None
            es_bias = None

        try:
            holt_model = ExponentialSmoothing(actuals, trend="add", seasonal=None)
            holt_fitted = holt_model.fit()
            holt_prediction = holt_fitted.forecast(1)[0]
            holt_wmape, holt_bias = backtest_holt_trend(actuals)
        except Exception:
            holt_prediction = None
            holt_wmape = None
            holt_bias = None

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
                "sku": str(sku),
                "model": "Linear Regression",
                "prediction": round(float(lr_prediction), 2),
                "wmape": lr_wmape,
                "bias": lr_bias,
                "records_used": int(len(sku_df)),
                "slope": round(float(lr_model.coef_[0]), 2),
                "demand_pattern": demand_pattern
            },
            {
                "sku": str(sku),
                "model": "Random Forest Forecast",
                "prediction": None if rf_prediction is None else round(float(rf_prediction), 2),
                "wmape": rf_wmape,
                "bias": rf_bias,
                "records_used": int(len(sku_df)),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": str(sku),
                "model": "XGBoost Forecast",
                "prediction": None if xgb_prediction is None else round(float(xgb_prediction), 2),
                "wmape": xgb_wmape,
                "bias": xgb_bias,
                "records_used": int(len(sku_df)),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": str(sku),
                "model": "Naive Forecast",
                "prediction": round(float(naive_prediction), 2),
                "wmape": naive_wmape,
                "bias": naive_bias,
                "records_used": int(len(sku_df)),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": str(sku),
                "model": "Seasonal Naive Forecast",
                "prediction": None if seasonal_naive_prediction is None else round(float(seasonal_naive_prediction), 2),
                "wmape": seasonal_naive_wmape,
                "bias": seasonal_naive_bias,
                "records_used": int(len(sku_df)),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": str(sku),
                "model": "Trend-Adjusted Seasonal Naive",
                "prediction": None if trend_adjusted_seasonal_naive_prediction is None else round(float(trend_adjusted_seasonal_naive_prediction), 2),
                "wmape": tasn_wmape,
                "bias": tasn_bias,
                "records_used": int(len(sku_df)),
                "slope": None,
                "demand_pattern": demand_pattern,
                "trend_factor": None if trend_factor is None else round(float(trend_factor), 3)
            },
            {
                "sku": str(sku),
                "model": "3-Month Moving Average",
                "prediction": round(float(ma_prediction), 2),
                "wmape": ma_wmape,
                "bias": ma_bias,
                "records_used": int(len(sku_df)),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": str(sku),
                "model": "Exponential Smoothing",
                "prediction": None if es_prediction is None else round(float(es_prediction), 2),
                "wmape": es_wmape,
                "bias": es_bias,
                "records_used": int(len(sku_df)),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": str(sku),
                "model": "Holt Trend",
                "prediction": None if holt_prediction is None else round(float(holt_prediction), 2),
                "wmape": holt_wmape,
                "bias": holt_bias,
                "records_used": int(len(sku_df)),
                "slope": None,
                "demand_pattern": demand_pattern
            },
            {
                "sku": str(sku),
                "model": "Holt-Winters Seasonal",
                "prediction": None if hw_prediction is None else round(float(hw_prediction), 2),
                "wmape": hw_wmape,
                "bias": hw_bias,
                "records_used": int(len(sku_df)),
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
            result["rank"] = int(index)

        for result in unranked_results:
            result["rank"] = "-"

        model_results.extend(ranked_results + unranked_results)

        best_model = ranked_results[0]

        model_residuals = get_model_residuals(
            best_model["model"],
            sku_df,
            actuals,
            test_periods=12
        )

        best_models.append({
            "sku": str(sku),
            "best_model": best_model["model"],
            "prediction": best_model["prediction"],
            "wmape": best_model["wmape"],
            "bias": best_model["bias"],
            "rank": best_model["rank"],
            "demand_pattern": demand_pattern,
            "residual_points_used": int(len(model_residuals))
        })

        forecast_explanations.append(
            generate_forecast_explanation(
                str(sku),
                best_model["model"],
                demand_pattern,
                rf_importance
            )
        )

        reliability_result = backtest_forecast_range_reliability(
            best_model["model"],
            sku_df,
            actuals,
            best_model["wmape"],
            residuals=model_residuals,
            test_periods=6
        )

        forecast_range_reliability.append({
            "sku": str(sku),
            "model": best_model["model"],
            "coverage": reliability_result["coverage"],
            "months_tested": reliability_result["months_tested"],
            "inside_count": reliability_result["inside_count"],
            "outside_count": reliability_result["outside_count"],
            "status": reliability_result["status"],
            "details": reliability_result["details"]
        })

        horizon_values = generate_forecast_horizon(
            best_model["model"],
            actuals,
            sku_df,
            horizon=12,
            seasonal_periods=12
        )

        tasn_values = generate_forecast_horizon(
            "Trend-Adjusted Seasonal Naive",
            actuals,
            sku_df,
            horizon=12,
            seasonal_periods=12
        )

        random_forest_values = generate_forecast_horizon(
            "Random Forest Forecast",
            actuals,
            sku_df,
            horizon=12,
            seasonal_periods=12
        )

        xgboost_values = generate_forecast_horizon(
            "XGBoost Forecast",
            actuals,
            sku_df,
            horizon=12,
            seasonal_periods=12
        )

        forecast_horizons.append({
            "sku": str(sku),
            "model": best_model["model"],
            "forecast": build_horizon_rows(future_months, horizon_values),
            "forecast_range": build_forecast_range_rows(
                future_months,
                horizon_values,
                best_model["wmape"],
                residuals=model_residuals
            ),
            "tasn_forecast": build_horizon_rows(future_months, tasn_values),
            "random_forest_forecast": build_horizon_rows(future_months, random_forest_values),
            "xgboost_forecast": build_horizon_rows(future_months, xgboost_values)
        })

        narrative = generate_narrative(
            str(sku),
            best_model["model"],
            best_model["wmape"],
            best_model["bias"],
            best_model["prediction"],
            demand_pattern
        )

        narratives.append({
            "sku": str(sku),
            "narrative": narrative
        })

    return jsonify({
        "status": "success",
        "message": "Forecast model comparison completed",
        "model_results": model_results,
        "best_models": best_models,
        "narratives": narratives,
        "demand_patterns": demand_patterns,
        "forecast_horizons": forecast_horizons,
        "random_forest_feature_importance": random_forest_feature_importance,
        "forecast_explanations": forecast_explanations,
        "forecast_range_reliability": forecast_range_reliability
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
