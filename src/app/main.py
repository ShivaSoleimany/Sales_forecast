import pandas as pd
import os
import sys
import numpy as np
import streamlit as st

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, root_path)
from src.scripts.process_data import load_data, get_total_monthly_sales, split_time_series, filter_shops_by_months, preprocess_data_for_multiplicative_model, preprocess_data
from src.scripts.visualization import plot_monthly_sales, decompose_series, plot_forecast_vs_actual, plot_extended_forecast, plot_forecast_vs_actual_with_full_data
from src.scripts.forecasting import generate_predictions, forecast_sales, in_sample_forecast, out_of_sample_forecast



def main():

    st.set_page_config(page_title='Sales Visualization App', layout='centered')
    st.title("Sales Data Analysis")

    df = load_data("src/data/sales_train.csv")
    shops_df = load_data("src/data/shops.csv")

    df = preprocess_data(df)
    df = filter_shops_by_months(df, min_months=4)
    df = preprocess_data_for_multiplicative_model(df)
    st.write("Data loaded successfully!")

    shops_in_df = shops_df[shops_df['shop_id'].isin(df['shop_id'].unique())]
    shop_name_to_id = dict(zip(shops_in_df['shop_name'], shops_in_df['shop_id']))
    selected_shop_name = st.selectbox('Select a Shop', shops_in_df['shop_name'].unique())
    shop_id = shop_name_to_id[selected_shop_name]


    if st.button('Show Sales Plot'):

        monthly_sales, title = get_total_monthly_sales(df, shop_id=shop_id)

        print(f"monthly_sales:\n{monthly_sales}")
        monthly_sales_fig = plot_monthly_sales(monthly_sales, title)
        st.pyplot(monthly_sales_fig)

        st.write("#### Additive Decomposition")
        st.write("This additive decomposition shows the trend, seasonality, and residuals where changes are constant over time.")

        additive_fig = decompose_series(monthly_sales["total_monthly_sales"], model_type='additive')

        multiplicative_fig = decompose_series(monthly_sales["total_monthly_sales"], model_type='multiplicative')

        col1, col2 = st.columns(2)  # Create two columns for side by side layout
        with col1:
            st.pyplot(additive_fig)
            st.write("#### Additive Decomposition")
            st.write("This additive decomposition shows the trend, seasonality, and residuals where changes are constant over time.")

        with col2:
            st.pyplot(multiplicative_fig)
            st.write("#### Multiplicative Decomposition")
            st.write("This multiplicative decomposition shows the trend, seasonality, and residuals where changes vary proportionally over time.")


        in_sample_forecast_df, _ = in_sample_forecast(monthly_sales[["date", "total_monthly_sales"]])
        fig = plot_forecast_vs_actual(monthly_sales[["date", "total_monthly_sales"]], in_sample_forecast_df['yhat'])
        st.pyplot(fig)

        out_of_sample_forecast_df, _ = out_of_sample_forecast(monthly_sales[["date", "total_monthly_sales"]], periods=12)
        fig = plot_extended_forecast(monthly_sales[["date", "total_monthly_sales"]], out_of_sample_forecast_df)
        st.pyplot(fig)

        train_data, valid_data, test_data = split_time_series(monthly_sales)
        best_model, best_config, best_mse = forecast_sales(train_data, valid_data)

        periods = len(test_data) + len(valid_data)
        st.write(periods)
        predictions = generate_predictions(best_model, periods)

        predictions_df = pd.DataFrame({
            'date': predictions.index,
            'total_monthly_sales': predictions.values
        })


        monthly_sales_sorted = monthly_sales.sort_values(by='date')

        train_data_sorted = train_data.sort_values(by='date')

        predictions_df_sorted = predictions_df.sort_values(by="date")

        combined_df = pd.concat([monthly_sales_sorted, predictions_df], axis=0)
        combined_df = combined_df.sort_values(by='date')

        fig = plot_forecast_vs_actual(train_data, valid_data, test_data, predictions_df)
        st.pyplot(fig)

if __name__ == "__main__":

    main()
