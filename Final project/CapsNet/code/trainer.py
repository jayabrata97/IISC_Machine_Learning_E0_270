import os
from datetime import datetime
from time import perf_counter
import torch
from torch.optim import Adam

DEBUG = os.environ.get("DEBUG") or False

class Trainer():
    """ Trainer class.

    Reusable class for training the models.
    Create instances by passing model, loaders and config.
    """
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

    def train(self):
        if (DEBUG):
            print("Training in DEBUG mode")
        model = self.model
        optimizer = Adam(model.parameters())
        n_epochs = self.config["epochs"]
        batch_size = self.config["batch_size"]
        accuracy = 0
        t_start = perf_counter()
        n_test_batches, n_train_batches = len(self.test_loader), len(self.train_loader)
        print("Starting training of {} model".format(self.config["model_name"]))
        print("Using GPU") if self.config["gpu"] else print("Using CPU")
        start_date_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        print("Start Time:\t\t{} (GPU Server Time)".format(start_date_time))
        print("Number of epochs:\t{}".format(n_epochs))
        print("Batch size:\t\t{}\n".format(batch_size))
        for epoch in range(n_epochs):
            print("Starting epoch {}".format(epoch+1))
            print("------------------")
            model.train() # Set training mode
            train_loss = 0
            correct = 0.
            total_count = 0.
            for batch_id, (data, target) in enumerate(self.train_loader):
                target = torch.sparse.torch.eye(10).index_select(dim=0, index=target) # Make target a 10 element vector

                if self.config["gpu"]:
                    data, target = data.cuda(), target.cuda()

                optimizer.zero_grad()
                output, reconstructions, masked = model(data)
                loss = model.loss(data, output, target, reconstructions)
                loss.backward()
                optimizer.step()

                train_loss += loss.data.item()
                correct += torch.sum(torch.argmax(masked.data, 1) == torch.argmax(target.data, 1)).cpu().numpy().squeeze()
                total_count += len(data)
                if (batch_id % 10 == 0):
                    print("Training - Epoch: {}/{}, Batch: {}/{}, Loss: {}, Accuracy: {}".format(epoch+1, n_epochs, batch_id+1, n_train_batches, train_loss, correct/total_count))
                if (DEBUG and batch_id >= 100):
                    break
            print("Total loss = {}, accuracy = {}".format(train_loss, correct/total_count))


            torch.save(model.state_dict(), self.config["model_name"])

            model.eval() # Set evaluation mode
            test_loss = 0
            correct = 0.
            total_count = 0.
            for batch_id, (data, target) in enumerate(self.test_loader):
                target = torch.sparse.torch.eye(10).index_select(dim=0, index=target)
                if self.config["gpu"]:
                    data, target = data.cuda(), target.cuda()

                output, reconstructions, masked = model(data)
                loss = model.loss(data, output, target, reconstructions)

                test_loss += loss.data.item()
                correct += torch.sum(torch.argmax(masked.data, 1) == torch.argmax(target.data, 1)).cpu().numpy().squeeze()
                total_count += len(data)
                accuracy = correct/total_count
                if (batch_id % 10 == 0):
                    print("Testing - Epoch: {}/{}, Batch: {}/{}, Loss: {}, Accuracy: {}".format(epoch+1, n_epochs, batch_id+1, n_test_batches, test_loss, accuracy))
                if (DEBUG and batch_id >= 100):
                    break
            print("Total test loss = {}, accuracy = {}".format(test_loss, accuracy))
            print("")
            if (DEBUG):
                break
        t_end = perf_counter()

        print("Summary")
        print("=======")
        print("Number of epochs:\t{}".format(n_epochs))
        print("Batch size:\t\t{}".format(batch_size))
        print("Validation Accuracy:\t{}%".format(accuracy*100))
        print("Start Time:\t\t{} (GPU Server Time)".format(start_date_time))
        print("End Time:\t\t{} (GPU Server Time)".format(datetime.now().strftime("%Y-%m-%d, %H:%M:%S")))
        print("Time elapsed:\t\t{} seconds".format(t_end - t_start))
