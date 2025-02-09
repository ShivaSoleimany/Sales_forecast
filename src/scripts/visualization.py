import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_monthly_sales(data, title):
    """
    Plots monthly sales for the entire dataset or a specific shop.

    Parameters:
        data (pd.DataFrame): The dataset containing sales data.
        shop_id (int or None): The shop ID to filter by. If None, plots sales for the entire dataset.
    """
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(data.index, data.values)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sold Items")
    ax.grid(True)
    return fig

def decompose_series(series, model_type='additive', period=12):
    """
    Decompose a time series into its components.

    Parameters:
        series (pd.Series): Time series data.
        model_type (str): 'additive' or 'multiplicative' to specify the type of decomposition.
        period (int): The periodicity of the series (e.g., 12 for monthly data with annual cycles).

    Returns:
        Decomposition plot of the seasonal, trend, and residual components.
    """
    result = seasonal_decompose(series, model=model_type, period=period)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 8), sharex=True)

    result.trend.plot(ax=axes[0], title='Trend')
    result.seasonal.plot(ax=axes[1], title='Seasonality')
    result.resid.plot(ax=axes[2], title='Residuals')
    series.plot(ax=axes[3], title='Original Series')

    plt.tight_layout()
    return fig


def plot_forecast_vs_actual(train_data, valid_data, test_data, predictions, title="Train, Validation, Test, and Forecast Data"):
    """
    Plot the train data, validation data, test data, and predictions on a single time series plot.

    Parameters:
        train_data (pd.DataFrame): Training data with columns 'date' and 'monthly_sales'.
        valid_data (pd.DataFrame): Validation data with columns 'date' and 'monthly_sales'.
        test_data (pd.DataFrame): Test data with columns 'date' and 'monthly_sales'.
        predictions (pd.DataFrame): Predictions data with columns 'date' and 'monthly_sales'.
        title (str): Title of the plot.

    Returns:
        plt.Figure: A matplotlib figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    print(f"train_data:\n{train_data}")
    print(f"valid_data:\n{valid_data}")
    print(f"test_data:\n{test_data}")
    print(f"predictions:\n{predictions}")
    # Ensure 'date' column is in datetime format for all dataframes
    train_data['date'] = pd.to_datetime(train_data['date'])
    valid_data['date'] = pd.to_datetime(valid_data['date'])
    test_data['date'] = pd.to_datetime(test_data['date'])
    predictions['date'] = pd.to_datetime(predictions['date'])

    # Plotting the data
    ax.plot(train_data['date'], train_data['total_monthly_sales'], label='Train Data', color='blue', marker='o')
    ax.plot(valid_data['date'], valid_data['total_monthly_sales'], color='green', marker='o')
    ax.plot(test_data['date'], test_data['total_monthly_sales'], label='Test Data', color='green', marker='o')
    ax.plot(predictions['date'], predictions['total_monthly_sales'], label='Predictions', color='red', linestyle='--', marker='x')

    # Setting plot labels and title
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly Sales")
    ax.legend()
    plt.grid(True)

    return fig


def plot_forecast_vs_actual_with_full_data(full_data, predictions, title="Forecast vs Actual Data"):
    """
    Plot the actual data and the predictions for the test period, including all data.

    Parameters:
        full_data (pd.DataFrame): The full dataset including train and test data.
        predictions (pd.DataFrame): The predictions for the test period, with 'date' and 'total_monthly_sales'.
        title (str): Title of the plot.

    Returns:
        plt.Figure: A matplotlib figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    full_data.plot(ax=ax, label='Actual Data', color='blue', marker='o', linestyle='-', markersize=4)

    predictions.plot(ax=ax, label='Predictions', color='red', linestyle='--', marker='x', markersize=6)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Monthly Sales")
    ax.legend()
    plt.grid(True)

    return fig



def plot_extended_forecast(actual_data, forecast, title="Extended Forecast vs Actual Data"):
    """
    Plot the actual data alongside the forecast from Prophet.

    Parameters:
        actual_data (pd.DataFrame): DataFrame containing the actual data with columns 'ds' for dates and 'y' for values.
        forecast (pd.DataFrame): DataFrame containing the forecasted data from Prophet, including 'ds' (date), 'yhat' (forecast),
                                 'yhat_lower' (lower bound of forecast), and 'yhat_upper' (upper bound of forecast).
        title (str): Title of the plot.

    Returns:
        plt.Figure: A matplotlib figure object containing the extended forecast and actual plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    actual_data = actual_data.rename(columns={'date': 'ds', 'total_monthly_sales': 'y'}).reset_index(drop=True)
    ax.plot(actual_data['ds'], actual_data['y'], label='Actual Data', color='blue', marker='o')

    ax.plot(forecast['ds'], forecast['yhat'], label='Forecasted Data', color='red', linestyle='--', marker='x')

    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='red', alpha=0.3, label='Confidence Interval')

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Values")
    ax.legend()
    plt.grid(True)

    return fig
