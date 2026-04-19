# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# ===========================
# MODIFIED FOR MC DROPOUT INFERENCE
# - Enables dropout layers during inference
# - Runs multiple stochastic forward passes
# - Computes mean and variance of predictions
# ===========================

from physicsnemo.datapipes.cae.mesh_datapipe import MeshDatapipe
from physicsnemo.distributed import DistributedManager
import vtk
from physicsnemo.models.unet import UNet
from omegaconf import DictConfig
import torch
import hydra
import torch.nn.functional as F
from physicsnemo.utils import load_checkpoint
from hydra.utils import to_absolute_path
import os
import numpy as np
from vtk.util.numpy_support import numpy_to_vtk
from physicsnemo.sym.utils.io.vtk import var_to_polyvtk

# -----------------------------------------------------------------------------
# MC DROPOUT HELPER: enable dropout layers during inference
# -----------------------------------------------------------------------------
def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout3d):
            m.train()  # keep dropout ON

# -----------------------------------------------------------------------------
# MC DROPOUT INFERENCE FUNCTION
# -----------------------------------------------------------------------------
def mc_forward(model, x, mc_samples=100):
    model.eval()            # keep batchnorm frozen
    enable_mc_dropout(model)

    preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            preds.append(model(x))

    return torch.stack(preds)  # [T, C, X, Y, Z]

@hydra.main(version_base="1.2", config_path="conf", config_name="config_inference")
def main(cfg: DictConfig) -> None:
    print("Inference Started (MC Dropout Enabled)!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # LOAD MODEL
    # ----------------------------
    model = UNet(**cfg.model).to(device)
    load_checkpoint(to_absolute_path(cfg.checkpoint), model)

    # ----------------------------
    # LOAD DATA
    # ----------------------------
    datapipe = MeshDatapipe(**cfg.data)
    dataloader = torch.utils.data.DataLoader(datapipe, batch_size=1)

    mc_samples = cfg.get("mc_samples", 100)

    os.makedirs(cfg.output_dir, exist_ok=True)

    for i, sample in enumerate(dataloader):
        x = sample["input"].to(device)

        # ----------------------------
        # MC DROPOUT FORWARD PASSES
        # ----------------------------
        preds = mc_forward(model, x, mc_samples=mc_samples)

        mean_pred = preds.mean(dim=0).squeeze(0)
        var_pred = preds.var(dim=0).squeeze(0)

        # ----------------------------
        # SAVE MEAN + UNCERTAINTY
        # ----------------------------
        mean_np = mean_pred.cpu().numpy()
        var_np = var_pred.cpu().numpy()

        vtk_data = {
            "prediction_mean": mean_np,
            "prediction_variance": var_np,
        }

        bounds = cfg.domain.bounds
        output_file = os.path.join(cfg.output_dir, f"mc_output_{i}.vtu")

        var_to_polyvtk(vtk_data, bounds=bounds, filename=output_file)

        print(f"Saved MC output to {output_file}")

if __name__ == "__main__":
    main()
