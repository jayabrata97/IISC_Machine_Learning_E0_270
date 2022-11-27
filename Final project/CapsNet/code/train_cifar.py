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

from cifar_loader import cifar_loader
from cifar_net import CifarNet

from trainer import Trainer

# Creating an instance of CapsNet
cifar_net = CifarNet()
if USE_CUDA:
    cifar_net = cifar_net.cuda()

n_epochs = args.epochs
batch_size = args.batch_size
config = {"model_name":"model_cifar", "epochs":n_epochs, "batch_size":args.batch_size, "gpu":USE_CUDA}
train_loader, test_loader = cifar_loader(batch_size)
trainer = Trainer(cifar_net, train_loader, test_loader, config)
trainer.train()
