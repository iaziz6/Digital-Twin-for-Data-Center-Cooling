import os
import numpy as np
import pandas as pd
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from physicsnemo.models.unet import UNet

# -----------------------------
# user settings for BASELINE UNet inference
# -----------------------------
NX, NY, NZ = 960, 96, 80
VTU_FILE = "/mnt/c/Users/iaziz6/Downloads/Training/internal_170.vtu"
MODEL_FILE = "/mnt/c/Users/iaziz6/Downloads/Training/checkpoints_baseline_unet/UNet.0.4.mdlus"
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

    T = vtk_to_numpy(point_data.GetArray("T"))
    U = vtk_to_numpy(point_data.GetArray("U"))
    p = vtk_to_numpy(point_data.GetArray("p"))
    wall_distance = vtk_to_numpy(point_data.GetArray("wallDistance"))
    valid = vtk_to_numpy(point_data.GetArray("vtkValidPointMask"))

    if len(pts) != NX * NY * NZ:
        raise ValueError(
            f"Point count mismatch: got {len(pts)}, expected {NX*NY*NZ}. "
            "NX, NY, NZ may be wrong for this case."
        )

    wall_distance_grid = np.reshape(wall_distance, (NX, NY, NZ), order="F")
    valid_grid = np.reshape(valid, (NX, NY, NZ), order="F")

    return mesh, pts, T, U, p, wall_distance_grid, valid, valid_grid


def build_model(device):
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Baseline UNet model file not found: {MODEL_FILE}")

    model = UNet(
        in_channels=10,
        out_channels=5,
        model_depth=5,
        feature_map_channels=[32, 32, 64, 64, 128, 128, 256, 256, 512, 512],
        num_conv_blocks=2,
    ).to(device)

    model.load(MODEL_FILE)
    model.eval()
    print(f"Loaded baseline UNet model weights from {MODEL_FILE}")
    return model

def compute_metrics(y_true, y_pred, mask):
    y_true_m = y_true[mask]
    y_pred_m = y_pred[mask]
    err = y_pred_m - y_true_m

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))

    denom = np.sqrt(np.mean(y_true_m ** 2)) + 1e-12
    rel_rmse_pct = float(100.0 * rmse / denom)

    return rmse, mae, max_abs, rel_rmse_pct


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("VTU file:", VTU_FILE)
    print("Baseline model file:", MODEL_FILE)

    mesh, pts, T_true, U_true, p_true, wall_distance_grid, valid_flat, valid_grid = load_case_from_vtu(VTU_FILE)

    wall_tensor = torch.from_numpy(wall_distance_grid).float().unsqueeze(0).unsqueeze(0).to(device)
    pos_tensor = make_positional_encoding().to(device)

    # input = wallDistance + positional encodings
    invar = torch.cat([wall_tensor, pos_tensor], dim=1)  # (1, 10, NX, NY, NZ)

    model = build_model(device)

    pred = model(invar).detach().cpu().numpy()[0]  # (5, NX, NY, NZ)

    ux_pred = pred[0].ravel(order="F")
    uy_pred = pred[1].ravel(order="F")
    uz_pred = pred[2].ravel(order="F")
    T_pred  = pred[3].ravel(order="F")
    p_pred  = pred[4].ravel(order="F")

    mask = valid_flat.astype(bool)

    metrics = []
    metrics.append(["Ux", *compute_metrics(U_true[:, 0], ux_pred, mask)])
    metrics.append(["Uy", *compute_metrics(U_true[:, 1], uy_pred, mask)])
    metrics.append(["Uz", *compute_metrics(U_true[:, 2], uz_pred, mask)])
    metrics.append(["T",  *compute_metrics(T_true,       T_pred,  mask)])
    metrics.append(["p",  *compute_metrics(p_true,       p_pred,  mask)])

    metrics_df = pd.DataFrame(
        metrics,
        columns=["Field", "RMSE", "MAE", "MaxAbsError", "RelRMSE_percent"]
    )
    metrics_df.to_csv("baseline_unet_metrics.csv", index=False)

    out_df = pd.DataFrame({
        "x": pts[:, 0],
        "y": pts[:, 1],
        "z": pts[:, 2],
        "Ux_true": U_true[:, 0],
        "Uy_true": U_true[:, 1],
        "Uz_true": U_true[:, 2],
        "T_true": T_true,
        "p_true": p_true,
        "Ux_pred": ux_pred,
        "Uy_pred": uy_pred,
        "Uz_pred": uz_pred,
        "T_pred": T_pred,
        "p_pred": p_pred,
        "vtkValidPointMask": valid_flat,
    })
    out_df.to_csv("baseline_unet_true_vs_pred.csv", index=False)

    np.savez(
        "baseline_unet_true_vs_pred.npz",
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        Ux_true=U_true[:, 0],
        Uy_true=U_true[:, 1],
        Uz_true=U_true[:, 2],
        T_true=T_true,
        p_true=p_true,
        Ux_pred=ux_pred,
        Uy_pred=uy_pred,
        Uz_pred=uz_pred,
        T_pred=T_pred,
        p_pred=p_pred,
        vtkValidPointMask=valid_flat,
    )

    print("\nSaved:")
    print("  baseline_unet_metrics.csv")
    print("  baseline_unet_true_vs_pred.csv")
    print("  baseline_unet_true_vs_pred.npz")
    print("\nMetrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
