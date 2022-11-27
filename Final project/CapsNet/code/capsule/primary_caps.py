import torch
import torch.nn as nn
from squash import squash

class PrimaryCaps(nn.Module):
    """Primary Capsule Layer."""
    def __init__(self, in_channels, out_channels, out_dimension, kernel_size, num_capsules):
        """Constructor of PrimaryCaps.

        Args:
            in_channels (int): Number of input channels. Each channel is a 2D tensor.
            out_channels (int): Number of outputs. Each output is vector.
            out_dimension (int): The dimension of output vectors.
            kernel_size (int or tuple): The size of the kernel of each convolution capsule inside this layer
            num_capsules (int): The number of convolution capsules inside this layer.
        """
        super(PrimaryCaps, self).__init__()
        self.out_channels = out_channels
        self.conv_capsules = nn.ModuleList([nn.Conv2d(in_channels=in_channels, out_channels=out_dimension, kernel_size=kernel_size, stride=2) for _ in range(num_capsules)])

    def forward(self, x):
        # Shape of x: (batch_size, in_channels, out_dimension, width, height)
        batch_size = x.size(0)
        out = torch.stack([capsule(x) for capsule in self.conv_capsules], dim=1)
        # Shape of out: (batch_size, num_capsules, out_dimension, out_width, out_height)
        # out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(batch_size, self.out_channels, -1)
        out = squash(out)
        return out
