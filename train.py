from pkgutil import get_loader
import seaborn as sns; sns.set(style='darkgrid')
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import copy
from utils.data_loader import get_dataloaders
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.CNN_TUMOR import CNN_TUMOR
from config.config import BASE_DIR, cnn_model

loss_func = nn.NLLLoss(reduction="sum")

opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)


def get_lr(optimizer):
    """
    Return the learning rate of the first parameter group as a scalar.
    """
    if not optimizer.param_groups:
        return None
    return optimizer.param_groups[0].get("lr", None)


def loss_epoch(model, loss_fn, dataloader, optimizer=None, device=None):
    """
    Run one epoch over dataloader.
    If optimizer is provided, run training steps; otherwise run evaluation.
    Returns (avg_loss, accuracy).
    """
    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    if optimizer is not None:
        model.train()
    else:
        model.eval()

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        if optimizer is not None:
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(X)
                loss = loss_fn(outputs, y)

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = total_correct / max(1, total_samples)
    return avg_loss, accuracy



def Train_Val(model, params, verbose=False):
    # Get the parameters
    epochs = params["epochs"]
    loss_func = params["f_loss"]
    opt = params["optimiser"]
    train_dl = params["train"]
    val_dl = params["val"]
    lr_scheduler = params["lr_change"]
    weight_path = params["weight_path"]

    # history of loss values in each epoch
    loss_history = {"train": [], "val": []}
    # histroy of metric values in each epoch
    metric_history = {"train": [], "val": []}
    # a deep copy of weights for the best performing model
    best_model_wts = copy.deepcopy(model.state_dict())
    # initialize best loss to a large value
    best_loss = float('inf')

    # Train Model n_epochs (the progress of training by printing the epoch number and the associated learning rate. It can be helpful for debugging, monitoring the learning rate schedule, or gaining insights into the training process.)

    for epoch in tqdm(range(epochs)):

        # Get the Learning Rate
        current_lr = get_lr(opt)
        if (verbose):
            print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))

        # Train Model Process

        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)

        # collect losses
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        # Evaluate Model Process

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)

        # store best model
        if (val_loss < best_loss):
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            # store weights into a local file
            torch.save(model.state_dict(), weight_path)
            if (verbose):
                print("Copied best model weights!")

        # collect loss and metric for validation dataset
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)

        # learning rate schedule
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            if (verbose):
                print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        if (verbose):
            print(f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100 * val_metric:.2f}")
            print("-" * 10)

            # load best model weights
    model.load_state_dict(best_model_wts)

    return model, loss_history, metric_history

# Define various parameters used for training and evaluation of a cnn_model

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders(batch_size=64, num_workers=0)

    params_train = {
        "train": train_loader,
        "val": test_loader,
        "epochs": 60,
        "optimiser": optim.Adam(cnn_model.parameters(), lr=3e-4),
        "lr_change": ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20),
        "f_loss": nn.NLLLoss(reduction="sum"),
        "weight_path": "weights.pt",
    }

    # train and validate the model
    cnn_model, loss_hist, metric_hist = Train_Val(cnn_model, params_train, verbose=True)



epochs=params_train["epochs"]
fig,ax = plt.subplots(1,2,figsize=(12,5))

sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["train"],ax=ax[0],label='loss_hist["train"]')
sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["val"],ax=ax[0],label='loss_hist["val"]')
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["train"],ax=ax[1],label='Acc_hist["train"]')
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist["val"],ax=ax[1],label='Acc_hist["val"]')