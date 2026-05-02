import os
import argparse
import numpy as np
import pandas as pd
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# IMPORTANT: MC Dropout model must use the same custom UNet used in train_dropout.py
from physicsnemo.models.unet import UNet

# -----------------------------
# Default settings for UNet + MC Dropout inference
# -----------------------------
NX, NY, NZ = 960, 96, 80
DEFAULT_VTU_FILE = "/home/iaziz6/Imo_Training/Training/internal_202.vtu"
DEFAULT_MODEL_FILE = "/home/iaziz6/Imo_Training/Training/checkpoints_dropout/UNet.0.16.mdlus"
DEFAULT_OUTPUT_PREFIX = "internal_115_mc_dropout_epoch16"
DEFAULT_NUM_THREADS = 4
DEFAULT_MC_SAMPLES = 30


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MC Dropout inference and compute RMSE metrics plus uncertainty."
    )
    parser.add_argument("--vtu_file", default=DEFAULT_VTU_FILE, help="Path to input/test .vtu file")
    parser.add_argument("--model_file", default=DEFAULT_MODEL_FILE, help="Path to trained MC Dropout .mdlus model file")
    parser.add_argument("--output_prefix", default=DEFAULT_OUTPUT_PREFIX, help="Prefix for output CSV/NPZ files")
    parser.add_argument("--num_threads", type=int, default=DEFAULT_NUM_THREADS, help="CPU thread count")
    parser.add_argument("--mc_samples", type=int, default=DEFAULT_MC_SAMPLES, help="Number of stochastic MC Dropout forward passes")
    return parser.parse_args()


def configure_threads(num_threads):
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)
    print("Torch threads:", torch.get_num_threads())


def make_positional_encoding(nx=NX, ny=NY, nz=NZ):
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    z = np.linspace(-1, 1, nz)

    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")

    x_freq_sin = np.sin(xv * 72 * np.pi / 2)
    x_freq_cos = np.cos(xv * 72 * np.pi / 2)
    y_freq_sin = np.sin(yv * 8 * np.pi / 2)
    y_freq_cos = np.cos(yv * 8 * np.pi / 2)
    z_freq_sin = np.sin(zv * 8 * np.pi / 2)
    z_freq_cos = np.cos(zv * 8 * np.pi / 2)

    pos = np.stack(
        [
            xv, x_freq_sin, x_freq_cos,
            yv, y_freq_sin, y_freq_cos,
            zv, z_freq_sin, z_freq_cos,
        ],
        axis=0,
    )
    return torch.from_numpy(pos).float().unsqueeze(0)


def require_array(point_data, name):
    arr = point_data.GetArray(name)
    if arr is None:
        raise KeyError(f"Required VTU point-data array '{name}' was not found.")
    return vtk_to_numpy(arr)


def load_case_from_vtu(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"VTU file not found: {filename}")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    mesh = reader.GetOutput()
    if mesh is None or mesh.GetPoints() is None:
        raise RuntimeError(f"Could not read mesh/points from VTU file: {filename}")

    pts = vtk_to_numpy(mesh.GetPoints().GetData())
    point_data = mesh.GetPointData()

    T = require_array(point_data, "T")
    U = require_array(point_data, "U")
    p = require_array(point_data, "p")
    wall_distance = require_array(point_data, "wallDistance")
    valid = require_array(point_data, "vtkValidPointMask")

    if len(pts) != NX * NY * NZ:
        raise ValueError(
            f"Point count mismatch: got {len(pts)}, expected {NX * NY * NZ}. "
            "NX, NY, NZ may be wrong for this case."
        )

    wall_distance_grid = np.reshape(wall_distance, (NX, NY, NZ), order="F")
    valid_grid = np.reshape(valid, (NX, NY, NZ), order="F")

    return mesh, pts, T, U, p, wall_distance_grid, valid, valid_grid


def build_model(device, model_file):
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Must match train_dropout.py architecture.
    model = UNet(
        in_channels=10,
        out_channels=5,
        model_depth=5,
        feature_map_channels=[32, 32, 64, 64, 128, 128, 256, 256, 512, 512],
        num_conv_blocks=2,
    ).to(device)

    model.load(model_file)
    print(f"Loaded MC Dropout model weights from: {model_file}")
    return model


def enable_mc_dropout(model):
    """Keep dropout active while leaving the rest of the model in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()


def compute_metrics(y_true, y_pred, mask):
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    err = y_pred_m - y_true_m

    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))

    denom = np.sqrt(np.mean(y_true_m ** 2)) + 1e-12
    rel_rmse_pct = float(100.0 * rmse / denom)

    return mse, rmse, mae, max_abs, rel_rmse_pct


def uncertainty_summary(std_field, mask):
    std_m = std_field[mask]
    return float(np.mean(std_m)), float(np.max(std_m))


@torch.no_grad()
def main():
    args = parse_args()
    configure_threads(args.num_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("VTU file:", args.vtu_file)
    print("Model file:", args.model_file)
    print("MC samples:", args.mc_samples)

    mesh, pts, T_true, U_true, p_true, wall_distance_grid, valid_flat, valid_grid = load_case_from_vtu(args.vtu_file)

    wall_tensor = torch.from_numpy(wall_distance_grid).float().unsqueeze(0).unsqueeze(0).to(device)
    pos_tensor = make_positional_encoding().to(device)
    invar = torch.cat([wall_tensor, pos_tensor], dim=1)

    model = build_model(device, args.model_file)
    enable_mc_dropout(model)

    preds = []
    for sample_id in range(args.mc_samples):
        pred = model(invar).detach().cpu().numpy()[0]
        preds.append(pred)
        print(f"Completed MC sample {sample_id + 1}/{args.mc_samples}", flush=True)

    preds = np.stack(preds, axis=0)  # (S, 5, NX, NY, NZ)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)

    ux_pred = pred_mean[0].ravel(order="F")
    uy_pred = pred_mean[1].ravel(order="F")
    uz_pred = pred_mean[2].ravel(order="F")
    T_pred = pred_mean[3].ravel(order="F")
    p_pred = pred_mean[4].ravel(order="F")

    ux_std = pred_std[0].ravel(order="F")
    uy_std = pred_std[1].ravel(order="F")
    uz_std = pred_std[2].ravel(order="F")
    T_std = pred_std[3].ravel(order="F")
    p_std = pred_std[4].ravel(order="F")

    mask = valid_flat.astype(bool)

    metrics = []
    fields = [
        ("Ux", U_true[:, 0], ux_pred, ux_std),
        ("Uy", U_true[:, 1], uy_pred, uy_std),
        ("Uz", U_true[:, 2], uz_pred, uz_std),
        ("T", T_true, T_pred, T_std),
        ("p", p_true, p_pred, p_std),
    ]
    for name, true, pred, std in fields:
        mse, rmse, mae, max_abs, rel_rmse = compute_metrics(true, pred, mask)
        mean_std, max_std = uncertainty_summary(std, mask)
        metrics.append([name, mse, rmse, mae, max_abs, rel_rmse, mean_std, max_std])

    metrics_df = pd.DataFrame(
        metrics,
        columns=[
            "Field", "MSE", "RMSE", "MAE", "MaxAbsError", "RelRMSE_percent",
            "Mean_MC_Std", "Max_MC_Std",
        ],
    )

    metrics_csv = f"{args.output_prefix}_metrics.csv"
    pred_csv = f"{args.output_prefix}_mean_std_true_vs_pred.csv"
    pred_npz = f"{args.output_prefix}_mean_std_true_vs_pred.npz"

    metrics_df.to_csv(metrics_csv, index=False)

    out_df = pd.DataFrame(
        {
            "x": pts[:, 0],
            "y": pts[:, 1],
            "z": pts[:, 2],
            "Ux_true": U_true[:, 0],
            "Uy_true": U_true[:, 1],
            "Uz_true": U_true[:, 2],
            "T_true": T_true,
            "p_true": p_true,
            "Ux_pred_mean": ux_pred,
            "Uy_pred_mean": uy_pred,
            "Uz_pred_mean": uz_pred,
            "T_pred_mean": T_pred,
            "p_pred_mean": p_pred,
            "Ux_pred_std": ux_std,
            "Uy_pred_std": uy_std,
            "Uz_pred_std": uz_std,
            "T_pred_std": T_std,
            "p_pred_std": p_std,
            "vtkValidPointMask": valid_flat,
        }
    )
    out_df.to_csv(pred_csv, index=False)

    np.savez(
        pred_npz,
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        Ux_true=U_true[:, 0],
        Uy_true=U_true[:, 1],
        Uz_true=U_true[:, 2],
        T_true=T_true,
        p_true=p_true,
        Ux_pred_mean=ux_pred,
        Uy_pred_mean=uy_pred,
        Uz_pred_mean=uz_pred,
        T_pred_mean=T_pred,
        p_pred_mean=p_pred,
        Ux_pred_std=ux_std,
        Uy_pred_std=uy_std,
        Uz_pred_std=uz_std,
        T_pred_std=T_std,
        p_pred_std=p_std,
        vtkValidPointMask=valid_flat,
    )

    print("\nSaved:")
    print(f"  {metrics_csv}")
    print(f"  {pred_csv}")
    print(f"  {pred_npz}")
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
