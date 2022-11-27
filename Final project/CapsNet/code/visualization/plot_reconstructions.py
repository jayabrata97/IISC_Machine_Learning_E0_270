import os, sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mnist_loader import mnist_loader
from caps_net import CapsNet

# Changing current working directory to the directory of source code
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

caps_net = CapsNet()
model_path = os.path.join(os.path.dirname(__file__), '../model_mnist')
data_path = os.path.join(os.path.dirname(__file__), '../data/')
caps_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

_, test_loader = mnist_loader(10, download_path=data_path)

# Get 5 images from the dataset
data = torch.stack([test_loader.dataset[i][0] for i in range(5) ], dim=0);

_, reconstructions, _ = caps_net(data)

fig = plt.figure()
for img_index in range(1, 6):
    fig.add_subplot(2, 5, img_index)
    plt.imshow(data[img_index - 1, 0].data.cpu().numpy(), cmap="binary")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    fig.add_subplot(2, 5, img_index + 5)
    plt.imshow(reconstructions[img_index - 1, 0].data.cpu().numpy(), cmap="binary")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
plt.show()
