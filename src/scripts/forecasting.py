import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def generate_predictions(model, periods):
    """
    Generate predictions from a fitted Exponential Smoothing model.

    Parameters:
        model (ExponentialSmoothing): The fitted Holt-Winters Exponential Smoothing model.
        periods (int): The number of periods to forecast.

    Returns:
        pd.Series: Predictions for the specified number of periods.
    """
    predictions = model.forecast(periods)
    return predictions


def forecast_sales(train_data, valid_data, seasonal_periods=12):
    """
    Evaluate different configurations of the Holt-Winters model and return the best model.

    Parameters:
        train_data (pd.Series): Training data for fitting the models.
        valid_data (pd.Series): Validation data for evaluating the models.
        seasonal_periods (int): Number of periods in a complete seasonal cycle.

    Returns:
        best_model (ExponentialSmoothing): The best fitting model based on validation data.
        best_config (str): Description of the best model configuration.
        best_mse (float): Mean squared error of the best model on the validation set.
    """
    configurations = [
        ('add', 'add'), ('add', 'mul'), ('mul', 'add'), ('mul', 'mul')
    ]
    best_mse = np.inf
    best_model = None
    best_config = None

    print("-------forecast_sales---------")
    train_data = train_data.set_index('date')
    train_data.index = pd.to_datetime(train_data.index)
    train_data.index.freq = 'MS'

    valid_data = valid_data.set_index('date')
    valid_data.index = pd.to_datetime(valid_data.index)
    valid_data.index.freq = 'MS'

    for trend, seasonal in configurations:
        try:
            model = ExponentialSmoothing(
                train_data['total_monthly_sales'], 
                trend=trend, 
                seasonal=seasonal, 
                seasonal_periods=seasonal_periods
            ).fit()

            forecast = model.forecast(len(valid_data))

            mse = ((valid_data['total_monthly_sales'] - forecast) ** 2).mean()

            if mse < best_mse:
                best_mse = mse
                best_model = model
                best_config = f'Trend: {trend}, Seasonal: {seasonal}'
        except Exception as e:
            print(f"Failed to fit model with configuration: Trend: {trend}, Seasonal: {seasonal}. Error: {e}")

    return best_model, best_config, best_mse

def in_sample_forecast(data):

    print(f"data:{data}")
    prophet_data = data.rename(columns={'date': 'ds', 'total_monthly_sales': 'y'}).reset_index(drop=True)
    model = Prophet()
    print(prophet_data)
    print(f"prophet_data:{prophet_data}")
    model.fit(prophet_data)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    return forecast, model


def out_of_sample_forecast(data, periods):
    """
    Generate out-of-sample forecasts using the Prophet model.

    Parameters:
        data (pd.DataFrame): The input data containing at least two columns: 'date' and 'total_monthly_sales'.
        periods (int): Number of future periods to forecast.

    Returns:
        forecast (pd.DataFrame): DataFrame containing the forecasted values.
        model (Prophet): The fitted Prophet model.
    """

    prophet_data = data.rename(columns={'date': 'ds', 'total_monthly_sales': 'y'}).reset_index(drop=True)
    model = Prophet()
    model.fit(prophet_data)

    future = model.make_future_dataframe(periods=periods, freq='M')

    forecast = model.predict(future)

    return forecast, model