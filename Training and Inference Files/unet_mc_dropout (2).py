# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# ===========================
# MODIFIED FOR MC DROPOUT
# MC Dropout added by inserting nn.Dropout3d layers inside Conv3DBlock
# and propagating `dropout_p` through Encoder / Decoder / UNet
# ===========================

import importlib
import torch
import torch.nn as nn
# Local WSL-compatible fallback: this file only needs standard torch.nn.Module
# for inference. The PhysicsNeMo core Module/metadata/version utilities are
# not required to load the architecture and checkpoint weights.
Module = nn.Module

try:
    import importlib.metadata as importlib_metadata
    te_version = importlib_metadata.version("transformer_engine")
    TE_AVAILABLE = True
except Exception:
    TE_AVAILABLE = False

if TE_AVAILABLE:
    try:
        te = importlib.import_module("transformer_engine.pytorch")
    except Exception:
        te = None
else:
    te = None

# ---------------------------------------------------------------------------------
# CHANGED: Added Dropout3d support via dropout_p argument
# ---------------------------------------------------------------------------------
class Conv3DBlock(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        norm="batch",
        activation="relu",
        dropout_p: float = 0.0,  # <-- MC DROPOUT PARAMETER
    ):
        super().__init__()

        layers = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=(norm is None),
            )
        ]

        if norm == "batch":
            layers.append(nn.BatchNorm3d(out_channels))
        elif norm == "layer":
            layers.append(nn.LayerNorm(out_channels))

        layers.append(nn.ReLU(inplace=True))

        # ----------------------------
        # MC DROPOUT INSERTED HERE
        # ----------------------------
        if dropout_p > 0.0:
            layers.append(nn.Dropout3d(p=dropout_p))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

# ---------------------------------------------------------------------------------
# CHANGED: Propagate dropout_p into Conv3DBlock
# ---------------------------------------------------------------------------------
class Encoder3DBlock(Module):
    def __init__(self, in_channels, out_channels, dropout_p: float = 0.0):
        super().__init__()
        self.conv1 = Conv3DBlock(in_channels, out_channels, dropout_p=dropout_p)
        self.conv2 = Conv3DBlock(out_channels, out_channels, dropout_p=dropout_p)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip

# ---------------------------------------------------------------------------------
# CHANGED: Propagate dropout_p into Conv3DBlock
# ---------------------------------------------------------------------------------
class Decoder3DBlock(Module):
    def __init__(self, in_channels, out_channels, dropout_p: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = Conv3DBlock(in_channels, out_channels, dropout_p=dropout_p)
        self.conv2 = Conv3DBlock(out_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# ---------------------------------------------------------------------------------
# CHANGED: Single dropout_p control knob in UNet
# ---------------------------------------------------------------------------------
class UNet(Module):
    def __init__(self, in_channels, out_channels, dropout_p: float = 0.1, **kwargs):
        super().__init__()

        self.enc1 = Encoder3DBlock(in_channels, 32, dropout_p=dropout_p)
        self.enc2 = Encoder3DBlock(32, 64, dropout_p=dropout_p)
        self.enc3 = Encoder3DBlock(64, 128, dropout_p=dropout_p)

        self.bottleneck = Conv3DBlock(128, 256, dropout_p=dropout_p)

        self.dec3 = Decoder3DBlock(256, 128, dropout_p=dropout_p)
        self.dec2 = Decoder3DBlock(128, 64, dropout_p=dropout_p)
        self.dec1 = Decoder3DBlock(64, 32, dropout_p=dropout_p)

        self.final = nn.Conv3d(32, out_channels, kernel_size=1)

    def load(self, path, map_location="cpu", strict=True):
        """Compatibility replacement for PhysicsNeMo Module.load().

        Supports common checkpoint formats:
        - raw PyTorch state_dict
        - dicts containing model/state_dict/model_state_dict/module keys
        - state_dicts with optional _module. or module. prefixes
        """
        checkpoint = torch.load(path, map_location=map_location)

        if isinstance(checkpoint, dict):
            state_dict = None
            for key in ("state_dict", "model_state_dict", "model", "module"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    state_dict = value
                    break
            if state_dict is None:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        cleaned = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("_module.", "module.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value

        try:
            return self.load_state_dict(cleaned, strict=strict)
        except RuntimeError:
            # Retry in non-strict mode to tolerate metadata/buffer naming differences.
            return self.load_state_dict(cleaned, strict=False)

    def save(self, path):
        """Small compatibility helper matching the expected Module API."""
        torch.save(self.state_dict(), path)

    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)

        x = self.bottleneck(x)

        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)

        return self.final(x)
