import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "capsule"))

from primary_caps import PrimaryCaps
from secondary_caps import SecondaryCaps
from decoder import Decoder

class CapsNet(nn.Module):
    def __init__(self):
        super(CapsNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9)
        self.primary_caps = PrimaryCaps(in_channels=256, out_channels=32*6*6, kernel_size=9, num_capsules=32, out_dimension=8)
        self.secondary_caps = SecondaryCaps(in_channels=32*6*6, out_channels=10, out_dimension=16, in_dimension=8)
        self.decoder = Decoder()
        self.mse_loss = nn.MSELoss()    # Mean squared error

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(out)
        out = self.primary_caps(out)
        out = self.secondary_caps(out)
        reconstructions, masked = self.decoder(out)
        return out, reconstructions, masked

    def loss(self, data, x, target, reconstructions):
        return self.margin_loss(x, target) + self.reconstructions_loss(data, reconstructions)

    def margin_loss(self, x, target):
        """
        Computes margin loss
        See section 3 of "Dynamic Routing Between Capsules"
        """
        lambda_ = 0.5 # _ because lambda is a keyword
        batch_size = x.size(0)

        v = torch.sqrt((x**2).sum(dim=2, keepdim=True))
        m_plus = 0.9
        m_minus = 0.1
        # Using the fact relu(x) == max(0, x)
        left = F.relu(m_plus - v).view(batch_size, -1)
        right = F.relu(v - m_minus).view(batch_size, -1)

        loss = target * left + lambda_ * (1.0 - target) * right
        loss = loss.sum(dim=1).mean()
        return loss

    def reconstructions_loss(self, data, reconstructions):
        """
        Return reconstructions_loss
        data - the batch of images feed into the input of capsule network
        reconstructions - The reconstructed set of images by the decoder
        """
        flat_reconstructions = reconstructions.view(reconstructions.size(0), -1)
        flat_data = data.view(reconstructions.size(0), -1)
        loss = self.mse_loss(flat_reconstructions, flat_data)
        return loss * 0.0005 # To normalize it's contribution to total loss
