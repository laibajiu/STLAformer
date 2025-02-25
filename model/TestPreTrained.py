import sys
sys.path.append("..")
import torch
import os
import time
from lib.data_prepare import get_dataloaders_from_index_data
from model.STLformer import STLformer
from lib.metrics import RMSE_MAE_MAPE
import numpy as np
import yaml
from train import predict
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
@torch.no_grad()
def predict(model, loader, scaler):
    model.eval()
    y = []
    out = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = scaler.inverse_transform(out_batch)
        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()
    y = np.vstack(y).squeeze()

    return y, out

@torch.no_grad()
def test_model(model, testset_loader, model_path, scaler, log=None):
    model.eval()

    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.to(DEVICE)
    start = time.time()
    y_true, y_pred = predict(model, testset_loader, scaler)
    end = time.time()
    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    print("All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    ))
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        print(f"Step {i + 1} RMSE = {rmse:.5f}, MAE = {mae:.5f}, MAPE = {mape:.5f}")
    print("Inference time: %.2f s" % (end - start))


if __name__ == "__main__":
    dataset = "PEMS08"
    model_path = "../pre-trained/" + dataset + ".pt"
    data_path = f"../data/{dataset}"
    with open('STLformer.yaml', 'r') as f:
        cfg = yaml.safe_load(f)[dataset]

    _, _, testset_loader, SCALER = get_dataloaders_from_index_data(
        data_path,
        batch_size=cfg.get("batch_size")
    )
    model = STLformer(**cfg["model_args"])
    test_model(model, testset_loader, model_path, SCALER)
