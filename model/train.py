import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
from torchinfo import summary
import yaml
import json
import sys
import copy
import csv
from torch.cuda.amp import autocast, GradScaler
sys.path.append("..")
from lib.utils import (
    print_log,
    seed_everything,
    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.metrics import MAE
from lib.data_prepare import get_dataloaders_from_index_data
from model.STLformer import STLformer
# ! X shape: (B, T, N, C)

@torch.no_grad()
def eval_model(model, valset_loader, criterion):
    model.eval()
    batch_loss_list = []
    for x_batch, y_batch in valset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out


def train_one_epoch(
    model, trainset_loader, valset_loader, optimizer, scheduler, scaler, criterion, clip_grad, log=None
):
    global cfg, global_iter_count, global_target_length

    model.train()
    batch_loss_list = []

    # ------------------------------- open amp ------------------------------- #
    for x_batch, y_batch in trainset_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        with autocast():
            out_batch = model(x_batch)
            out_batch = SCALER.inverse_transform(out_batch)
            loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

    # ------------------------------- close amp ------------------------------- #
    # for x_batch, y_batch in trainset_loader:
    #     x_batch = x_batch.to(DEVICE)
    #     y_batch = y_batch.to(DEVICE)
    #     out_batch = model(x_batch)
    #     out_batch = SCALER.inverse_transform(out_batch)
    #     loss = criterion(out_batch, y_batch)
    #     batch_loss_list.append(loss.item())
    #     optimizer.zero_grad()
    #     loss.backward()
    #     if clip_grad:
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    #     optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    val_mae = MAE(*predict(model, valset_loader))
    scheduler.step(val_mae)

    return epoch_loss


def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    scaler,
    criterion,
    clip_grad=0,
    max_epochs=200,
    verbose=1,
    plot=False,
    log=None,
    save=None,
):
    model = model.to(DEVICE)

    min_val_loss = np.inf
    train_loss_list = []
    val_loss_list = []
    torch.cuda.empty_cache()
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, trainset_loader, valset_loader, optimizer, scheduler, scaler, criterion, clip_grad, log=log
        )
        train_loss_list.append(train_loss)
        val_loss = eval_model(model, valset_loader, criterion)
        val_loss_list.append(val_loss)
        LR = optimizer.param_groups[0]['lr']
        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                f" \tLR = {LR}",
                "Train Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model


@torch.no_grad()
def test_model(model, testset_loader, log=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="pems08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    args = parser.parse_args()
    seed = int(42)
    seed_everything(seed)
    set_cpu_num(1)
    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = args.dataset
    dataset = dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = STLformer.__name__
    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]

    # ------------------------------- make log file ------------------------------ #

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log = os.path.join(log_path, f"{model_name}-{dataset}-{now}.log")
    log = open(log, "a")
    log.seek(0)
    log.truncate()

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        SCALER,
    ) = get_dataloaders_from_index_data(
        data_path,
        batch_size=cfg.get("batch_size"),
        log=log,
    )
    print_log(log=log)

    # -------------------------------- load model -------------------------------- #

    model = STLformer(**cfg["model_args"])

    # --------------------------- set model saving path -------------------------- #

    save_path = f"../saved_models/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save = os.path.join(save_path, f"{model_name}-{dataset}-{now}.pt")

    # ---------------------- set loss, optimizer, scheduler, scaler ---------------------- #

    if dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0015),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='min', factor=cfg.get("lr_decay_rate", 0.1),
                                                                   patience=cfg.get("patience", 20),
                                                                   threshold=0.001, threshold_mode='rel',
                                                                   cooldown=0, min_lr=2e-6, eps=1e-08)
    scaler = GradScaler()

    # --------------------------- print model structure -------------------------- #

    print_log("---------", model_name, "---------", log=log)
    print_log(
        json.dumps(cfg, ensure_ascii=False, indent=4, cls=CustomJSONEncoder), log=log
    )
    print_log(
        summary(
            model,
            [
                cfg["batch_size"],
                cfg["in_steps"],
                cfg["num_nodes"],
                next(iter(trainset_loader))[0].shape[-1],
            ],
            verbose=0,
        ),
        log=log,
    )
    print_log(log=log)

    # --------------------------- train and test model --------------------------- #

    print_log(f"Loss: {criterion._get_name()}", log=log)
    print_log(log=log)

    model = train(
        model,
        trainset_loader,
        valset_loader,
        optimizer,
        scheduler,
        scaler,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        verbose=1,
        log=log,
        save=save,
    )

    print_log(f"Saved Model: {save}", log=log)

    test_model(model, testset_loader, log=log)

    log.close()


