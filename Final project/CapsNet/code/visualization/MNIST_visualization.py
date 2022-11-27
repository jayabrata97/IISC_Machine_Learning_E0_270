import os,sys
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")) 

from mnist_loader import mnist_loader
from caps_net import CapsNet

# Changing current working directory to the directory of source code
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
os.chdir('..')

caps_net = CapsNet()
print(os.path.join(os.path.dirname(__file__), 'model_mnist')) 
caps_net.load_state_dict(torch.load('model_mnist', map_location=torch.device('cpu')))

_, test_loader = mnist_loader(10)

# Get 50 images from the dataset
data = torch.cat([test_loader.dataset[i][0] for i in range(50) ], dim=0) 
print(data.shape) 
data=data.reshape(1400,28)
print(data.shape)
data2=data.numpy()
print(data2.shape)
data3=np.hstack(data2.reshape(10,140,28)) 
print(data3.shape)


data4=torch.stack([test_loader.dataset[i][0] for i in range(50)],dim=0)
print(data4.shape)
_, reconstructions, _ = caps_net(data4)
print(reconstructions.shape)
data5=reconstructions.view(1400,-1)
print(data5.shape)
data6=data5.detach().numpy()
print(data6.shape)
data7=np.hstack(data6.reshape(10,140,28))
print(data7.shape)

data8=np.concatenate((data3,data7))
print(data8.shape)

plt.imshow(data8,cmap="gray")
plt.axis('off')
plt.show()












