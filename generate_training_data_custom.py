import pandas as pd
import numpy as np

import argparse
import os.path as osp
import os
import logging

logging.getLogger().setLevel(logging.DEBUG)

FREQ_CHOICES = ["M", "SM", "W", "D", "H", "T", "min", "S", "L", "ms", "U", "us", "N"]


def get_datapoints_at_freq(df: pd.DataFrame, freq: str):
    timepoints = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
    rel_indices = [df.index.get_loc(tp, method="nearest") for tp in timepoints]
    rel_df = df.iloc[rel_indices]
    return rel_df


def generate_context_target(
    df: pd.DataFrame,
    context_window: int,
    year_feature: bool,
    month_feature: bool,
    day_feature: bool,
    time_feature: bool,
):
    """
    Generates samples from raw data
    @param df: the timeseries DataFrame
    @param context_window: the size of the context window for the past and future
    @year_feature: whether to use the year feature of the time data
    @month_feature: whether to use the month of year feature of the time data
    @day_feature: whether to use the day of week feature of the time data
    @time_feature: whether to use the time of day feature of the time data

    @return: (x, y)
        x: (M, context_window, N, F)
        y: (M, context_window, N, F)
    M - the number of samples
    N - the number of nodes in the dataset
    F - the added features (depends on *_feature params)
    """
    _, N = df.values.shape
    data = [df.values[..., None]]

    if year_feature:
        data.append(np.repeat(df.index.year.values[:, None, None], N, axis=1))

    if month_feature:
        data.append(np.repeat(df.index.month.values[:, None, None], N, axis=1))

    if day_feature:
        data.append(np.repeat(df.index.dayofweek.values[:, None, None], N, axis=1))

    if time_feature:
        seconds = df.index.hour * 3600 + df.index.minute * 60 + df.index.second
        data.append(np.repeat((seconds / 86400).values[:, None, None], N, axis=1))

    data = np.concatenate(data, axis=-1)

    window_idx = np.arange(context_window)
    min_t = context_window
    max_t = len(data) - context_window - 1
    context = np.empty(shape=([max_t - min_t + 1, context_window] + list(data.shape[1:])))
    target = np.empty(shape=([max_t - min_t + 1, context_window] + list(data.shape[1:])))

    for t in range(min_t, max_t):
        context[t - min_t] = data[t - context_window + window_idx, ...]
        target[t - min_t] = data[t + window_idx, ...]
        if t % 100 == 0:
            print(f"{t}/{max_t - min_t - 1}", end = "\r")

    return context, target


def main(args):
    if not osp.exists(args.ts_fp):
        logging.error("The time series fp %s does not exist", args.input_dir)
        return 1
    if osp.splitext(args.ts_fp)[-1] != ".csv":
        logging.error(
            "The time series fp is expected to be in CSV format but the extension is %s",
            osp.splitext(args.ts_fp),
        )
        return 1

    if (
        args.year is None
        and args.month is None
        and args.day is None
        and args.time is None
    ):
        logging.error("At least one time series feature needs to be added")

    time_series_df = None
    if args.time_col is not None:
        time_series_df = pd.read_csv(
            args.ts_fp, parse_dates=[args.time_col], index_col=[args.time_col]
        )
    else:
        time_series_df = pd.read_csv(args.ts_fp)
        if type(time_series_df.index) != pd.core.indexes.datetimes.DatetimeIndex:
            logging.error(
                f"The index of the DataFrame is not a DatetimeIndex and no time column was supplied"
            )
            return 1

    assert (
        type(time_series_df.index) == pd.core.indexes.datetimes.DatetimeIndex
    ), "Index is not DatetimeIndex"

    freq, freq_unit = args.freq, args.freq_unit
    if (freq is None) != (freq_unit is None):
        logging.error(
            f"Both `freq` and `freq_unit` need to be set if you want to subset. Terminating"
        )
        return 1

    if freq is not None and freq_unit is not None:
        time_series_df = get_datapoints_at_freq(time_series_df, f"{freq}{freq_unit}")

    context_data, target_data = generate_context_target(time_series_df, args.context_window, args.year, args.month, args.day, args.time)

    num_samples = context_data.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    os.makedirs(args.output_dir, exist_ok=True)

    context_train, target_train = context_data[:num_train], target_data[:num_train]
    np.savez_compressed(
        os.path.join(args.output_dir, "train.npz"),
        x=context_train,
        y=target_train)

    context_val, target_val = (
        context_data[num_train:num_train + num_val],
        target_data[num_train:num_train + num_val],
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "val.npz"),
        x=context_val,
        y=target_val)

    context_test, target_test = context_data[-num_test:], target_data[-num_test:]
    np.savez_compressed(
        os.path.join(args.output_dir, "test.npz"),
        x=context_test,
        y=target_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--ts_fp",
        type=str,
        required=True,
        metavar="FP",
        help="The filepath to the .csv file that contains the time series data."
        " The DataFrame has columns that represent nodes and the optional time column."
        " The samples in the DataFrame are samples taken at different times that correspond"
        " to the time column or the DataTimeIndex. If the time column is not provided,"
        " the script expects the index to be a DatetimeIndex.",
    )
    parser.add_argument(
        "--time_col",
        type=str,
        metavar="C",
        help="Specify the time column in the dataset."
        " That column is then converted into DatetimeIndex and used for the time series processing."
        " If this is not supplied, the script expects that the index of the time series data"
        " is already a DatetimeIndex.",
    )
    DEFAULT_TIME_WINDOW = 12
    parser.add_argument(
        "-c",
        "--context_window",
        type=int,
        metavar="N",
        default=DEFAULT_TIME_WINDOW,
        help="Symmetrical time window in DatetimeIndex freq. This is used for the depth"
        " of context for the model. It is telling how many time steps of the past will"
        " we take into consideration to predict how many time steps in the future."
        " The actual window in terms of time will depend on the freq of time steps in the index.",
    )
    freq_selector = parser.add_argument_group(
        "Frequency selector",
        "What subset of datapoints in the time series dataframe to select."
        " This interacts with context_window because it dictates the freq of"
        " time steps in the index. We take datapoints sequentially at `freq`"
        " starting from the first datapoint. If a datapoint at timepoint doesn't exist"
        " we take the nearest earlier datapoint.",
    )
    freq_selector.add_argument(
        "--freq_unit",
        type=str,
        choices=FREQ_CHOICES,
        help=" Refer to https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases"
        " for definitions of aliases.",
    )
    freq_selector.add_argument(
        "-f",
        "--freq",
        type=int,
        metavar="N",
        help="The number of `freq_unit` to set the freq to."
        " For eg, if `freq=2` and `freq_unit='H'` then we will take one datapoint"
        " every 2 hours starting from the first datapoint's timepoint.",
    )
    ts_features = parser.add_argument_group(
        "Time series features",
        "What features to extract from DatetimeIndex to add it as the time features",
    )
    ts_features.add_argument(
        "-y", "--year", action="store_true", help="Whether to add the year feature."
    )
    ts_features.add_argument(
        "-m",
        "--month",
        action="store_true",
        help="Whether to add the month of the year feature.",
    )
    ts_features.add_argument(
        "-d",
        "--day",
        action="store_true",
        help="Whether to add the day of the week feature.",
    )
    ts_features.add_argument(
        "-t",
        "--time",
        action="store_true",
        help="Whether to add the time of the day feature."
        " It's scaled to [0,1] using seconds granularity.",
    )
    args = parser.parse_args()
    main(args)
