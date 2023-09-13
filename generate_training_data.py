from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(df,
                                   x_offsets,
                                   y_offsets,
                                   add_time_in_day=True,
                                   add_day_in_week=False,
                                   scaler=None):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
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
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = None, None
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    x = np.empty(shape=([max_t - min_t + 1, len(x_offsets)] +
                        list(data.shape[1:])))
    y = np.empty(shape=([max_t - min_t + 1, len(y_offsets)] +
                        list(data.shape[1:])))
    for t in range(min_t, max_t):
        x[t - min_t] = data[t + x_offsets, ...]
        y[t - min_t] = data[t + y_offsets, ...]
        if t % 100 == 0:
            print(f"{t}/{max_t - min_t - 1}", end = "\r")

    return x, y


def generate_train_val_test(args):
    df = pd.read_hdf(args.input_filepath)
    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1), )))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=True,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    os.makedirs(args.output_dir)

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    np.savez_compressed(
        os.path.join(args.output_dir, "train.npz"),
        x=x_train,
        y=y_train,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))
    # val
    x_val, y_val = (
        x[num_train:num_train + num_val],
        y[num_train:num_train + num_val],
    )
    np.savez_compressed(
        os.path.join(args.output_dir, "val.npz"),
        x=x_val,
        y=y_val,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))

    # test
    x_test, y_test = x[-num_test:], y[-num_test:]
    np.savez_compressed(
        os.path.join(args.output_dir, "test.npz"),
        x=x_test,
        y=y_test,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]))


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
