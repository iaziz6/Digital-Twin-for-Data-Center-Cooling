import os
import numpy as np
import pandas as pd
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# Try to use the same UNet class used by train_ensemble.py.
# If unet_mc_dropout.py is not available locally, fall back to standard PhysicsNeMo UNet.
try:
    from unet_mc_dropout import UNet
    UNET_SOURCE = "unet_mc_dropout.UNet"
except Exception:
    from physicsnemo.models.unet import UNet
    UNET_SOURCE = "physicsnemo.models.unet.UNet"

# -----------------------------
# user settings for 3x3 ensemble inference
# -----------------------------
NX, NY, NZ = 960, 96, 80
VTU_FILE = "/mnt/c/Users/iaziz6/Downloads/Training/internal_202.vtu"

# Put your three ensemble model files here after downloading/copying them locally.
MODEL_FILES = [
    "/mnt/c/Users/iaziz6/Downloads/Training/checkpoints_ensemble_3x3/ensemble_0/UNet.0.3.mdlus",
    "/mnt/c/Users/iaziz6/Downloads/Training/checkpoints_ensemble_3x3/ensemble_1/UNet.0.3.mdlus",
    "/mnt/c/Users/iaziz6/Downloads/Training/checkpoints_ensemble_3x3/ensemble_2/UNet.0.3.mdlus",
]

OUTPUT_PREFIX = "ensemble_3x3_internal_202"
NUM_THREADS = 15

# CPU threading
os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["TORCH_NUM_THREADS"] = str(NUM_THREADS)
torch.set_num_threads(NUM_THREADS)

print(f"Requested CPU threads: {NUM_THREADS}")
print(f"Torch threads: {torch.get_num_threads()}")
print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
print(f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
print(f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}")
print(f"NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS')}")
print(f"TORCH_NUM_THREADS={os.environ.get('TORCH_NUM_THREADS')}")
print(f"UNet source: {UNET_SOURCE}")


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
    )  # (9, NX, NY, NZ)

    return torch.from_numpy(pos).float().unsqueeze(0)  # (1, 9, NX, NY, NZ)


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
        raise FileNotFoundError(f"Ensemble model file not found: {model_file}")

    model = UNet(
        in_channels=10,
        out_channels=5,
        model_depth=5,
        feature_map_channels=[32, 32, 64, 64, 128, 128, 256, 256, 512, 512],
        num_conv_blocks=2,
    ).to(device)

    model.load(model_file)
    model.eval()
    print(f"Loaded ensemble member weights from {model_file}")
    return model


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


def field_uncertainty(std_field, mask):
    std_m = std_field[mask]
    return float(np.mean(std_m)), float(np.max(std_m))


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("VTU file:", VTU_FILE)
    print("Number of ensemble members:", len(MODEL_FILES))
    for k, mf in enumerate(MODEL_FILES):
        print(f"Model {k}: {mf}")

    mesh, pts, T_true, U_true, p_true, wall_distance_grid, valid_flat, valid_grid = load_case_from_vtu(VTU_FILE)

    wall_tensor = torch.from_numpy(wall_distance_grid).float().unsqueeze(0).unsqueeze(0).to(device)
    pos_tensor = make_positional_encoding().to(device)

    # input = wallDistance + positional encodings
    invar = torch.cat([wall_tensor, pos_tensor], dim=1)  # (1, 10, NX, NY, NZ)

    preds = []
    for model_file in MODEL_FILES:
        model = build_model(device, model_file)
        pred = model(invar).detach().cpu().numpy()[0]  # (5, NX, NY, NZ)
        preds.append(pred)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preds = np.stack(preds, axis=0)  # (Nens, 5, NX, NY, NZ)
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)

    ux_pred = mean_pred[0].ravel(order="F")
    uy_pred = mean_pred[1].ravel(order="F")
    uz_pred = mean_pred[2].ravel(order="F")
    T_pred = mean_pred[3].ravel(order="F")
    p_pred = mean_pred[4].ravel(order="F")

    ux_std = std_pred[0].ravel(order="F")
    uy_std = std_pred[1].ravel(order="F")
    uz_std = std_pred[2].ravel(order="F")
    T_std = std_pred[3].ravel(order="F")
    p_std = std_pred[4].ravel(order="F")

    mask = valid_flat.astype(bool)

    metrics = []
    metrics.append(["Ux", *compute_metrics(U_true[:, 0], ux_pred, mask), *field_uncertainty(ux_std, mask)])
    metrics.append(["Uy", *compute_metrics(U_true[:, 1], uy_pred, mask), *field_uncertainty(uy_std, mask)])
    metrics.append(["Uz", *compute_metrics(U_true[:, 2], uz_pred, mask), *field_uncertainty(uz_std, mask)])
    metrics.append(["T", *compute_metrics(T_true, T_pred, mask), *field_uncertainty(T_std, mask)])
    metrics.append(["p", *compute_metrics(p_true, p_pred, mask), *field_uncertainty(p_std, mask)])

    metrics_df = pd.DataFrame(
        metrics,
        columns=[
            "Field", "MSE", "RMSE", "MAE", "MaxAbsError", "RelRMSE_percent",
            "Mean_Ensemble_Std", "Max_Ensemble_Std",
        ],
    )

    metrics_csv = f"{OUTPUT_PREFIX}_metrics.csv"
    pred_csv = f"{OUTPUT_PREFIX}_true_vs_pred_mean_std.csv"
    pred_npz = f"{OUTPUT_PREFIX}_true_vs_pred_mean_std.npz"

    metrics_df.to_csv(metrics_csv, index=False)

    out_df = pd.DataFrame({
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
    })
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
    print("\nMetrics using ensemble mean prediction:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
