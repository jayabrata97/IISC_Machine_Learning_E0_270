import os
import torch
import torch.nn as nn

USE_CUDA = os.environ.get("USE_CUDA") or False

class CifarDecoder(nn.Module):
    def __init__(self):
        super(CifarDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=160, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024*3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Shape of x: (batch_size, num_classes, dimension)
        classes = (x**2).sum(2)
        # Shape of classes: (batch_size, num_classes)
        _, max_indices = classes.max(dim=1)
        # Shape of max_indices: (classes)

        masked = torch.eye(10)

        if USE_CUDA:
            masked = masked.cuda()

        # Make output of capsules zero except the one which produces longest output
        masked = masked.index_select(dim=0, index=max_indices)
        filtered = x.unsqueeze(-1) * masked[:, :, None, None]

        # Flatten filtered data and pass through reconstruction layers
        reconstructions = self.layers(filtered.view(x.size(0), -1))
        reconstructions = reconstructions.view(-1, 3, 32, 32)

        return reconstructions, masked
