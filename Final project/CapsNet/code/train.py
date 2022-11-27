import os
import sys
from datetime import datetime
from time import perf_counter

import torch
from torch.optim import Adam

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-G", "--gpu", help="Use gpu for training", action="store_true")
parser.add_argument("-E", "--epochs", help="Number of epochs", type=int, default=100)
parser.add_argument("-B", "--batch_size", help="Batch size", type=int, default=100)
args = parser.parse_args()
# Do it before loading anything
USE_CUDA = False
if args.gpu:
    USE_CUDA = True
    os.environ["USE_CUDA"] = str(USE_CUDA)

from mnist_loader import mnist_loader
from caps_net import CapsNet

from trainer import Trainer

# Creating an instance of CapsNet
caps_net = CapsNet()
if USE_CUDA:
    caps_net = caps_net.cuda()

n_epochs = args.epochs
batch_size = args.batch_size
config = {"model_name":"model_mnist","epochs":n_epochs,"batch_size":args.batch_size,"gpu":USE_CUDA}
train_loader, test_loader = mnist_loader(batch_size)
trainer = Trainer(caps_net, train_loader, test_loader, config)
trainer.train()
