import pandas as pd

def load_data(df_path):

    return pd.read_csv(df_path,  parse_dates = True)

def check_null(df, cols):

    return df[cols].isnull()

def preprocess_data(df):

    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df['date'] = df['date'].apply(lambda x: x.replace(day=1))

    df.fillna(method='ffill', inplace=True)

    df.interpolate(method='linear', inplace=True)

    return df


def preprocess_data_for_multiplicative_model(df, value_column='item_cnt_day'):
    """
    Preprocess data for multiplicative models by ensuring no zero or negative values.

    Parameters:
        df (pd.DataFrame): The input data frame containing sales data.
        value_column (str): The name of the column containing the values to adjust.

    Returns:
        pd.DataFrame: Adjusted DataFrame suitable for multiplicative models.
    """

    min_value = df[value_column].min()
    if min_value <= 0:
        offset = abs(min_value) + 1 
        df[value_column] += offset

    return df


def filter_shops_by_months(df, min_months=4):
    return df.groupby('shop_id').filter(lambda x: len(x) >= min_months).reset_index()


def get_total_monthly_sales(df, shop_id=None):

    if shop_id is not None:
        df = df[df['shop_id'] == shop_id]
        title = f"Monthly Sales for Shop {shop_id}"
    else:
        title = "Monthly Sales Across All Shops"

    total_monthly_sales = df.groupby(["date_block_num", "date"])["item_cnt_day"].sum().reset_index()

    total_monthly_sales.rename(columns={'item_cnt_day': 'total_monthly_sales'}, inplace=True)

    return total_monthly_sales, title

def split_time_series(data, train_frac=0.8, valid_frac=0.1, test_frac=0.1):
    """
    Splits a time series dataset into training, validation, and test sets.

    Parameters:
        data (pd.Series or pd.DataFrame): The full time series dataset.
        train_frac (float): Fraction of the dataset to include in the train split.
        valid_frac (float): Fraction of the dataset to include in the validation split.
        test_frac (float): Fraction of the dataset to include in the test split.

    Returns:
        train_set (pd.Series or pd.DataFrame): Training data subset.
        valid_set (pd.Series or pd.DataFrame): Validation data subset.
        test_set (pd.Series or pd.DataFrame): Test data subset.
    """
    assert train_frac + valid_frac + test_frac == 1, "The sum of fractions must be 1."

    size = len(data)
    train_end = int(size * train_frac)
    valid_end = train_end + int(size * valid_frac)

    train_set = data[:train_end]
    valid_set = data[train_end:valid_end]
    test_set = data[valid_end:]

    return train_set, valid_set, test_set