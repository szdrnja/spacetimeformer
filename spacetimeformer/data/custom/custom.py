import logging
import os
import os.path as osp

import numpy as np
import torch
from ...logger import Logger
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset

from ..base import BaseData

logger = Logger(osp.basename(__file__))
logger.setLevel(logging.NOTSET)

class CustomData(BaseData):
    """
        Expects a npz file that has 'x' and 'y' that represent the context and target.
    """
    def __init__(self, path):
        self.path = path

        context_train, target_train = self.__read("train")
        context_val, target_val = self.__read("val")
        context_test, target_test = self.__read("test")

        x_c_train, y_c_train = self.__split_set(context_train)
        x_t_train, y_t_train = self.__split_set(target_train)

        x_c_val, y_c_val = self.__split_set(context_val)
        x_t_val, y_t_val = self.__split_set(target_val)

        x_c_test, y_c_test = self.__split_set(context_test)
        x_t_test, y_t_test = self.__split_set(target_test)

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

    def __read(self, split):
        with np.load(os.path.join(self.path, f"{split}.npz")) as f:
            context = f["x"]
            target = f["y"]
        return context, target

    def __split_set(self, data: np.ndarray):
        """
        @param data: Is the data to be split up with shape (N, time_steps, M, K)
          Where the first dimension are the datapoints, the second represent the
          prediciton window (how many time steps we're looking back and forward),
          the third one is the number of nodes (sources of truth such as scanners or
          power plants), and the last dimension is the actual data. In the last dimension
          the first datapoint is the value to predict, the second is the time of day,
          and the rest is either the one-hot-enc of the day of the week or a single value
          that represents the day of the week. K can be either 9 or 3.

        @return x, y
          x - in the shape of (N, time_steps, 2) holds the time of day and day of week
            scaled to [-1, 1]
          y - in the shape of (N, time_steps, 1) holds the value to predict

        In this method we scale time of day and day of week to [-1,1] and split the data into
        x and y.
        """
        logger.debug(f"data shape is {data.shape}")
        EXPECTED_K = [3]
        if data.shape[-1] not in EXPECTED_K:
            raise TypeError(f"Expected data's last axis to be one of {EXPECTED_K} but got {data.shape[-1]}")

        time_day = np.squeeze(data[:, :, 0, 1:])

        # normalize time of day
        time = 2 * normalize(time_day[:, :, 0], "max", axis=0) - 1

        # normalize day of week
        day_of_week = time_day[:, :, 1]

        day_of_week = 2 * normalize(day_of_week, norm="max", axis=0) - 1

        x = np.concatenate((time[..., None], day_of_week[..., None]), axis=-1)
        y = data[:, :, :, 0]
        return x, y

    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--data_path", type=str, help="The directory where the npz files can be found.")
        parser.add_argument("--context_points", type=int, default=12, help="The size of the context sequence.")

        parser.add_argument("--target_points", type=int, default=12, help="The size of the target sequence.")


def CustomDataTorch(data: CustomData, split: str):
    assert split in ["train", "val", "test"]
    if split == "train":
        tensors = data.train_data
    elif split == "val":
        tensors = data.val_data
    else:
        tensors = data.test_data
    tensors = [torch.from_numpy(x).float() for x in tensors]
    return TensorDataset(*tensors)
