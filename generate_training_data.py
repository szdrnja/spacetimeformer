from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(df: pd.DataFrame,
                                   context_offsets: np.array,
                                   target_offsets: np.array,
                                   add_time_in_day=True,
                                   add_day_in_week=True,
                                   scaler=None):
    """
    Generates samples from raw data
    @param df: the timeseries DataFrame to create training dataset from
    @param context_offsets: the
    @param target_offsets:
    @param add_time_in_day:
    @param add_day_in_week:
    @param scaler:
    @return:
    # x: (N, input_length, num_nodes, input_dim)
    # y: (N, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    data_list = [data]
    if add_time_in_day:
        time_ind = (df.index.values -
                    df.index.values.astype("datetime64[D]")) / np.timedelta64(
                        1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        data_list.append(time_in_day)
    if add_day_in_week:
        day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
        day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
        data_list.append(day_in_week)

    data = np.concatenate(data_list, axis=-1)

    # t is the index of the last observation.
    min_t = abs(min(context_offsets))
    max_t = abs(num_samples - abs(max(target_offsets)))  # Exclusive
    context = np.empty(shape=([max_t - min_t + 1, len(context_offsets)] +
                        list(data.shape[1:])))
    target = np.empty(shape=([max_t - min_t + 1, len(target_offsets)] +
                        list(data.shape[1:])))
    for t in range(min_t, max_t):
        context[t - min_t] = data[t + context_offsets, ...]
        target[t - min_t] = data[t + target_offsets, ...]
        if t % 100 == 0:
            print(f"{t}/{max_t - min_t - 1}", end = "\r")

    return context, target


def generate_train_val_test(args):
    df = pd.read_hdf(args.input_filepath)
    # 0 is the latest observed sample.
    context_offsets = np.arange(-11, 1, 1)
    # Predict the next one hour
    target_offsets = np.arange(1, 13, 1)
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    context_data, target_data = generate_graph_seq2seq_io_data(
        df,
        context_offsets=context_offsets,
        target_offsets=target_offsets,
        add_time_in_day=True,
        add_day_in_week=True,
    )

    print(f"context_data.shape: {context_data.shape}, target_data.shape: {target_data.shape}")

    num_samples = min([1000, context_data.shape[0]])
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    os.makedirs(args.output_dir, exist_ok=True)

    context_train, target_train = context_data[:num_train], target_data[:num_train]
    np.savez_compressed(
        os.path.join(args.output_dir, "train.npz"),
        x=context_train,
        y=target_train,
        x_offsets=context_offsets.reshape(list(context_offsets.shape) + [1]),
        y_offsets=target_offsets.reshape(list(target_offsets.shape) + [1]))

    context_val, target_val = (
        context_data[num_train:num_train + num_val],
        target_data[num_train:num_train + num_val],
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "val.npz"),
        x=context_val,
        y=target_val,
        x_offsets=context_offsets.reshape(list(context_offsets.shape) + [1]),
        y_offsets=target_offsets.reshape(list(target_offsets.shape) + [1]))

    context_test, target_test = context_data[-num_test:], target_data[-num_test:]
    np.savez_compressed(
        os.path.join(args.output_dir, "test.npz"),
        x=context_test,
        y=target_test,
        x_offsets=context_offsets.reshape(list(context_offsets.shape) + [1]),
        y_offsets=target_offsets.reshape(list(target_offsets.shape) + [1]))


def main(args):
    print("Generating training data")
    generate_train_val_test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="data/",
        help="Output directory.",
    )
    parser.add_argument(
        "-i",
        "--input_filepath",
        type=str,
        help="Raw traffic readings.",
    )
    args = parser.parse_args()
    main(args)
