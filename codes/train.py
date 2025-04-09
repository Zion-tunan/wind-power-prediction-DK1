"""
train.py

This script defines a reusable training loop function `fit()` for time series forecasting models.
It trains the model for one epoch, calculates both training and validation loss,
and saves the model if it achieves the best validation loss so far.

Usage:
- Call `fit()` inside a loop over `config.epochs`.
- Compatible with PyTorch DataLoader and torch.nn models.

Includes:
- Training loop with tqdm progress bar
- Validation loss calculation
- Model checkpointing based on validation performance
"""

import torch
from Config import config
from tqdm import tqdm
import numpy as np
config = config()

# General-purpose training function
def fit(epoch, model, loss_function, optimizer, train_loader, test_loader, bst_loss):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)
    for data in train_bar:
        x_train, y_train = data
        optimizer.zero_grad()
        y_train_pred = model(x_train)
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, config.epochs, loss)

    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    test_running_loss = 0
    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            x_test, y_test = data
            y_test_pred = model(x_test)
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))
            test_running_loss += test_loss.item()
    epoch_test_loss = test_running_loss / len(test_loader.dataset)

    if epoch_test_loss < bst_loss:
        bst_loss = epoch_test_loss
        torch.save(model.state_dict(), config.save_path)

    return epoch_loss, epoch_test_loss