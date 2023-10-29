import math

import torch
from torch import nn
from einops import rearrange


class LinearModel(nn.Module):
    """
    Implements basic linear model with repeating linear feed-forward
    layers.
    """

    def __init__(
        self, context_points: int, shared_weights: bool = False, d_yt: int = 7
    ):
        """
        Initializes a LinearModel. Creates the weights and the bias
        Tensors and initalizes them with a unifrom dist using
        eps=sqrt(1/context_points).

        If shared_weights is True then only one layer is created, otherwise
        the number of layers is set to d_yt

        @param context_points: the number of nodes per layer
        @param shared_weights: whether the weights are shared between layers
        @param d_yt: the number of values to predict per datapoint
        """
        super().__init__()

        if not shared_weights:
            assert d_yt is not None
            layer_count = d_yt
        else:
            layer_count = 1

        self.weights = nn.Parameter(
            torch.ones((context_points, layer_count)), requires_grad=True
        )
        self.bias = nn.Parameter(torch.ones((layer_count)), requires_grad=True)

        d = math.sqrt(1.0 / context_points)
        self.weights.data.uniform_(-d, d)
        self.bias.data.uniform_(-d, d)

        self.window = context_points
        self.shared_weights = shared_weights
        self.d_yt = d_yt

    def forward(
        self, y_c: torch.Tensor, pred_len: int, d_yt: int = None
    ) -> torch.Tensor:
        """
        @param y_c: y tensor used for training.
            Shape is (batch, length, d_yc)
        @param pred_len: the number of timesteps to predict
        @param d_yt: the number of values to predict per datapoint

        @returns torch.Tensor that is the output of the feed-forward operatoin.
            Shape is (batch, pred_len, d_yt)
        """
        batch, _, _ = y_c.shape
        d_yt = d_yt or self.d_yt

        output = torch.zeros(batch, pred_len, d_yt).to(y_c.device)

        for i in range(pred_len):
            inp = torch.cat((y_c[:, i:, :d_yt], output[:, :i, :]), dim=1)
            output[:, i, :] = self._inner_forward(inp)

        return output

    def _inner_forward(self, input_t: torch.Tensor) -> torch.Tensor:
        """
        @param input_t: the input tensor. Shape is (batch, length, dy).

        @returns torch.Tensor that is the output of a feed-forward
            linear operation
        """
        batch = input_t.shape[0]
        if self.shared_weights:
            input_t = rearrange(
                input_t,
                "batch length dy -> (batch dy) length 1",
            )

        output_t = (self.weights * input_t[:, -self.window :, :]).sum(1) + self.bias

        if self.shared_weights:
            output_t = rearrange(
                output_t,
                "(batch dy) 1 -> batch dy",
                batch=batch,
            )

        return output_t
