import torch
from torch import nn


class Time2Vec(nn.Module):
    """
    Time2Vec: Learning a Vector Representation of Time
    ref: https://arxiv.org/pdf/1907.05321.pdf

    For a given scalar notion of time t, t2v(t) is a vector of size k+1 defined as follows:

    t2v(t)[i] = w_i @ t + phi_i if i == 0 else F(w_i @ t + phi_i)

    where
        t2v(t)[i] is the i-th element of t2v(t)
        F is a periodic activation function
            It helps capture periodic behaviors without the need for
            feature engineering
        phi_is and w_is are learnable parameters
    """

    def __init__(self, input_dim=6, embed_dim=512, act_function=torch.sin):
        """
        @param input_dim: the dimension of the inputs
        @param embed_dim: the dimension of the output embedding
        @param act_function: the activation function for t2v.
            Needs to be a periodic function.
        """
        assert embed_dim % input_dim == 0
        super(Time2Vec, self).__init__()
        self.enabled = embed_dim > 0
        if self.enabled:
            self.embed_dim = embed_dim // input_dim
            self.input_dim = input_dim
            self.embed_weight = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.embed_bias = nn.parameter.Parameter(
                torch.randn(self.input_dim, self.embed_dim)
            )
            self.act_function = act_function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the t2v embedding to x.
        For more information about t2v refer to class comment

        @param x: tensor used as an input.
            Shape is (batch, seq_len, input_dim).

        @return tensor that's the output
        """
        if self.enabled:
            x = torch.diag_embed(x)
            # x.shape = (bs, sequence_length, input_dim, input_dim)
            x_affine = torch.matmul(x, self.embed_weight) + self.embed_bias
            # x_affine.shape = (bs, sequence_length, input_dim, time_embed_dim)
            x_affine[..., 1:] = self.act_function(x_affine[..., 1:])
            output = x_affine.view(x_affine.size(0), x_affine.size(1), -1)
            # output.shape = (bs, sequence_length, input_dim * time_embed_dim)
        else:
            output = x
        return output
