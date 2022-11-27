import os, sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cifar_loader import cifar_loader
from cifar_net import CifarNet

# Changing current working directory to the directory of source code
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

cifar_net = CifarNet()
model_path = os.path.join(os.path.dirname(__file__), '../model_cifar')
data_path = os.path.join(os.path.dirname(__file__), '../data/')
cifar_net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

_, test_loader = cifar_loader(10, download_path=data_path)

# Get 5 images from the dataset
data = torch.stack([test_loader.dataset[i][0] for i in range(5) ], dim=0);

_, reconstructions, _ = cifar_net(data)

fig = plt.figure()
for img_index in range(1, 6):
    fig.add_subplot(2, 5, img_index)
    plt.imshow(data[img_index - 1].data.cpu().numpy().transpose((1,2,0)),interpolation="none")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    fig.add_subplot(2, 5, img_index + 5)
    plt.imshow(reconstructions[img_index - 1].data.cpu().numpy().transpose((1,2,0)),interpolation="none")
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
plt.show()
