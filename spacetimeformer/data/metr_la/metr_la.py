import numpy as np
import os

import torch
from torch.utils.data import TensorDataset

from ..base import BaseData

class METR_LA_Data(BaseData):
    def _read(self, split):
        with np.load(os.path.join(self.path, f"{split}.npz")) as f:
            x = f["x"]
            y = f["y"]
        return x, y

    def _split_set(self, data: np.ndarray):
        """
        @param data: Is the data to be split up with shape (N, time_steps, M, 9)
          Where the first dimension are the datapoints, the second represent the
          prediciton window (how many time steps we're looking back and forward),
          the third one is the number of nodes (sources of truth such as scanners or
          power plants), and the last dimension is the actual data. In the last dimension
          the first datapoint is the value to predict, the second is the time of day
          normalized to 0-1, and the rest is the one-hot-enc of the day of the week

        @return x, y
          x - in the shape of (N, time_steps, 2) holds the time of day and day of week
            scaled to [-1, 1]
          y - in the shape of (N, time_steps, 1) holds the value to predict

        In this method we scale time of day and day of week to [-1,1] and split the data into
        x and y.
        """
        # time features are the same across the nodes.
        # just grab the first one
        x = np.squeeze(data[:, :, 0, 1:])

        # normalize time of day
        time = 2 * x[:, :, 0] - 1

        # convert one-hot day of week feature to scalar [-1, 1]
        day_of_week = x[:, :, 1:]
        day_of_week = np.argmax(day_of_week, axis=-1).astype(np.float32)
        day_of_week = 2 * day_of_week / 6 - 1
        x = np.concatenate((time[..., None], day_of_week[..., None]), axis=-1)

        y = data[:, :, :, 0]
        return x, y

    def __init__(self, path):
        self.path = path

        context_train, target_train = self._read("train")
        context_val, target_val = self._read("val")
        context_test, target_test = self._read("test")

        x_c_train, y_c_train = self._split_set(context_train)
        x_t_train, y_t_train = self._split_set(target_train)

        x_c_val, y_c_val = self._split_set(context_val)
        x_t_val, y_t_val = self._split_set(target_val)

        x_c_test, y_c_test = self._split_set(context_test)
        x_t_test, y_t_test = self._split_set(target_test)

        self.max_per_node = y_c_train.max((0, 1))

        y_c_train = self.scale(y_c_train)
        y_t_train = self.scale(y_t_train)

        y_c_val = self.scale(y_c_val)
        y_t_val = self.scale(y_t_val)

        y_c_test = self.scale(y_c_test)
        y_t_test = self.scale(y_t_test)

        self.train_data = (x_c_train, y_c_train, x_t_train, y_t_train)
        self.val_data = (x_c_val, y_c_val, x_t_val, y_t_val)
        self.test_data = (x_c_test, y_c_test, x_t_test, y_t_test)

    def scale(self, x):
        return x / self.max_per_node

    def inverse_scale(self, x):
        return x * self.max_per_node

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str, default="./data/metr_la/")
        parser.add_argument("--context_points", type=int, default=12)

        parser.add_argument("--target_points", type=int, default=12)


def METR_LA_Torch(data: METR_LA_Data, split: str):
    assert split in ["train", "val", "test"]
    if split == "train":
        tensors = data.train_data
    elif split == "val":
        tensors = data.val_data
    else:
        tensors = data.test_data
    tensors = [torch.from_numpy(x).float() for x in tensors]
    return TensorDataset(*tensors)


if __name__ == "__main__":
    data = METR_LA_Data(path="./data/pems-bay/")
    dset = METR_LA_Torch(data, "test")
    breakpoint()
