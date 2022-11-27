import os
import torch
import torch.nn.functional as F
import torch.nn as nn

from squash import squash

USE_CUDA = os.environ.get("USE_CUDA") or False

class SecondaryCaps(nn.Module):
    """Secondary Capsule Layer."""
    def __init__(self, in_channels, out_channels, out_dimension, in_dimension):
        """Constructor of SecondaryCaps.

        Args:
            in_channels (int): Number of input channels. Each channel is a vector.
            out_channels (int): Number of output channels. Each channel is a vector.
            out_dimension (int): The dimension of output vectors.
            in_dimension (int): The dimension of input vectors.
        """
        super(SecondaryCaps, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_dimension = out_dimension
        self.W = nn.Parameter(torch.randn(1, in_channels, out_channels, out_dimension, in_dimension))

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, in_dimension)
        batch_size = x.size(0)
        W = torch.cat([self.W] * batch_size, dim=0)
        # Shape of W: (batch_size, in_channels, out_channels, out_dimension, in_dimension)
        x = torch.stack([x] * self.out_channels, dim=2).unsqueeze(-1)
        # Shape of W: (batch_size, in_channels, out_channels, in_dimension, 1)
        u_hat = torch.matmul(W, x)
        # Shape of u_hat: (batch_size, in_channels, out_channels, out_dimension, 1)
        return self.routing(u_hat)

    def routing(self, u_hat):
        # Shape of u_hat: (batch_size, in_channels, out_channels, out_dimension, 1)
        batch_size = u_hat.size(0)
        b = torch.zeros(batch_size, self.in_channels, self.out_channels, 1)
        if USE_CUDA:
            b = b.cuda()
        num_iterations = 3
        for i in range(num_iterations):
            c = F.softmax(b).unsqueeze(-1)
            # Shape of c: (batch_size, in_channels, out_channels, 1, 1)
            s =  (c * u_hat).sum(dim=1).squeeze(-1)
            # Shape of s: (batch_size, out_channels, out_dimension)
            v = squash(s).unsqueeze(-1)
            # Shape of v: (batch_size, out_channels, out_dimension, 1)
            a = torch.matmul(u_hat.transpose(3,4), torch.stack([v] * self.in_channels, dim=1))
            # Shape of v: (batch_size, in_channels, out_channels, 1, 1)
            b = b + a.squeeze(-1)
            # Shape of v: (batch_size, in_channels, out_channels, 1)
        return v.squeeze(-1)
