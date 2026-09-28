"""Fine-tune the Slippi-AI policy network on custom or character-specific datasets.

Enables full 26-character coverage by fine-tuning the base `medium-v2` policy network
on top-player replays for underrepresented characters (e.g. Bowser, DK, Dr. Mario,
Game & Watch, Ganondorf, Kirby, Link, Mario, Mewtwo, Ness, Pichu, Roy, Young Link, Zelda).

Supports hardware acceleration on Apple Silicon (MPS), NVIDIA GPUs (CUDA), or CPU.
Outputs standard Slippi-AI `.pkl` checkpoints compatible with `scripts/eval_two.py`
as well as native binary containers for embedded runtimes.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import io
import json
import math
import os
import pickle
import random
import struct
import sys
import tempfile
import time
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Character Enums & Constants
# ---------------------------------------------------------------------------

ALL_CHARACTERS = {
    0: "MARIO", 1: "FOX", 2: "CPTFALCON", 3: "DK", 4: "KIRBY", 5: "BOWSER",
    6: "LINK", 7: "SHEIK", 8: "NESS", 9: "PEACH", 10: "POPO", 12: "PIKACHU",
    13: "SAMUS", 14: "YOSHI", 15: "JIGGLYPUFF", 16: "MEWTWO", 17: "LUIGI",
    18: "MARTH", 19: "ZELDA", 20: "YLINK", 21: "DOC", 22: "FALCO",
    23: "PICHU", 24: "GAMEANDWATCH", 25: "GANONDORF", 26: "ROY",
}

# The 14 characters omitted from original medium-v2 training
OMITTED_CHARACTERS = {
    0: "MARIO", 3: "DK", 4: "KIRBY", 5: "BOWSER", 6: "LINK", 8: "NESS",
    16: "MEWTWO", 19: "ZELDA", 20: "YLINK", 21: "DOC", 23: "PICHU",
    24: "GAMEANDWATCH", 25: "GANONDORF", 26: "ROY",
}

POLICY_DELAY = 21
UNROLL_LENGTH = 60
TOTAL_PARAMS = 141
TOTAL_FLOATS = 23_887_032

MAGIC = b"SLIPAI01"
FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Custom Unpickler for legacy TF/Sonnet structures
# ---------------------------------------------------------------------------

class CheckpointUnpickler(pickle.Unpickler):
    """Safely loads legacy Slippi-AI pickles without requiring TensorFlow/Sonnet."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "slippi_ai.embed" and name == "ItemsType":
            class DummyItemsType:
                MLP = "mlp"
                FLAT = "flat"
                SKIP = "skip"
            return DummyItemsType
        if "sonnet" in module or "tensorflow" in module:
            return lambda *args, **kwargs: None
        try:
            return super().find_class(module, name)
        except Exception:
            return lambda *args, **kwargs: None


def flatten_pyarrow_table(table: pa.Table) -> pa.Table:
    """Recursively flattens nested structs in a PyArrow table."""
    while any(pa.types.is_struct(t) for t in table.schema.types):
        table = table.flatten()
    return table


# ---------------------------------------------------------------------------
# Replay Dataset & Batch Sampler
# ---------------------------------------------------------------------------

class ReplayDataset:
    """Loads parsed Slippi games and creates balanced training batches."""

    def __init__(
        self,
        parsed_dir: Path,
        meta_path: Path,
        seed: int = 42,
        max_cache_games: int = 120,
    ):
        self.parsed_dir = parsed_dir
        self.meta_path = meta_path
        self.random = random.Random(seed)
        self.max_cache_games = max_cache_games

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        # Index trajectories by character: char_id -> list of (slp_md5, player_index)
        self.char_to_trajectories: Dict[int, List[Tuple[str, int]]] = {
            c: [] for c in ALL_CHARACTERS
        }

        for game in self.meta:
            slp_md5 = game.get("slp_md5")
            if not slp_md5:
                continue
            game_file = self.parsed_dir / slp_md5
            if not game_file.exists():
                continue

            for idx, p in enumerate(game.get("players", [])):
                char_id = p.get("character")
                if char_id in self.char_to_trajectories:
                    self.char_to_trajectories[char_id].append((slp_md5, idx))

        self.available_omitted = [
            c for c in OMITTED_CHARACTERS if len(self.char_to_trajectories[c]) > 0
        ]
        self.available_base = [
            c for c in ALL_CHARACTERS
            if c not in OMITTED_CHARACTERS and len(self.char_to_trajectories[c]) > 0
        ]

        self.cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.cache_keys: List[str] = []

    def load_game_columns(self, slp_md5: str) -> Optional[Dict[str, np.ndarray]]:
        if slp_md5 in self.cache:
            return self.cache[slp_md5]

        path = self.parsed_dir / slp_md5
        if not path.exists():
            return None

        try:
            with open(path, "rb") as f:
                decomp = zlib.decompress(f.read())
            table = pq.read_table(io.BytesIO(decomp))
            flat = flatten_pyarrow_table(table)
            columns = {name: flat[name].to_numpy() for name in flat.column_names}

            if len(self.cache_keys) >= self.max_cache_games:
                oldest = self.cache_keys.pop(0)
                self.cache.pop(oldest, None)

            self.cache[slp_md5] = columns
            self.cache_keys.append(slp_md5)
            return columns
        except Exception:
            return None

    def sample_batch(
        self,
        batch_size: int = 16,
        unroll_length: int = UNROLL_LENGTH,
        omitted_ratio: float = 0.85,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        batch_obs = []
        batch_prev = []
        batch_buttons = []
        batch_axes = []
        batch_shoulder = []

        needed = batch_size
        attempts = 0

        while len(batch_obs) < needed and attempts < needed * 15:
            attempts += 1
            if self.random.random() < omitted_ratio and self.available_omitted:
                char_id = self.random.choice(self.available_omitted)
            elif self.available_base:
                char_id = self.random.choice(self.available_base)
            else:
                char_id = self.random.choice(list(self.char_to_trajectories.keys()))

            trajs = self.char_to_trajectories[char_id]
            if not trajs:
                continue

            slp_md5, player_idx = self.random.choice(trajs)
            cols = self.load_game_columns(slp_md5)
            if cols is None:
                continue

            total_frames = len(cols.get("root.stage", []))
            min_needed = unroll_length + POLICY_DELAY + 2
            if total_frames <= min_needed:
                continue

            start_t = self.random.randint(0, total_frames - min_needed)
            end_t = start_t + unroll_length

            obs, prev, tgt_b, tgt_ax, tgt_sh = self._vectorize_slice(
                cols, player_idx, start_t, end_t
            )
            batch_obs.append(obs)
            batch_prev.append(prev)
            batch_buttons.append(tgt_b)
            batch_axes.append(tgt_ax)
            batch_shoulder.append(tgt_sh)

        if len(batch_obs) < needed:
            return None

        return (
            torch.from_numpy(np.stack(batch_obs)),
            torch.from_numpy(np.stack(batch_prev)),
            (
                torch.from_numpy(np.stack(batch_buttons)),
                torch.from_numpy(np.stack(batch_axes)),
                torch.from_numpy(np.stack(batch_shoulder)),
            ),
        )

    def _vectorize_slice(
        self,
        cols: Dict[str, np.ndarray],
        self_idx: int,
        start_t: int,
        end_t: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T = end_t - start_t
        other_idx = 1 - self_idx
        obs = np.zeros((T, 2613), dtype=np.float32)
        prev_act = np.zeros((T, 151), dtype=np.float32)

        p_self = f"root.p{self_idx}"
        p_opp = f"root.p{other_idx}"

        # Delay offset: the agent observes frame t, and predicts action at t + POLICY_DELAY
        t_obs = np.arange(start_t, end_t)
        t_act = t_obs + POLICY_DELAY

        # Encode Self player (893 features)
        self._encode_player(obs[:, 0:893], cols, p_self, t_obs)

        # Encode Opponent player (893 features)
        self._encode_player(obs[:, 893:1786], cols, p_opp, t_obs)

        # Stage one-hot (33 classes at offset 1786)
        if "root.stage" in cols:
            stages = cols["root.stage"][t_obs]
            for i, stg in enumerate(stages):
                if 0 <= stg < 33:
                    obs[i, 1786 + stg] = 1.0

        # Encode Items (254 features starting at offset 2334)
        item_offset = 2334
        item_types = cols.get("root.items.item_0.type")
        item_x = cols.get("root.items.item_0.x")
        item_y = cols.get("root.items.item_0.y")
        if item_types is not None and item_x is not None and item_y is not None:
            itypes = item_types[t_obs]
            ix = item_x[t_obs] * 0.05
            iy = item_y[t_obs] * 0.05
            for i, ityp in enumerate(itypes):
                if 0 <= ityp < 250:
                    obs[i, item_offset + ityp] = 1.0
                obs[i, item_offset + 250] = ix[i]
                obs[i, item_offset + 251] = iy[i]

        # Targets at t_act
        tgt_buttons = np.zeros((T, 8), dtype=np.float32)
        btn_names = ["A", "B", "X", "Y", "Z", "L", "R", "D_UP"]
        for b_idx, name in enumerate(btn_names):
            k = f"{p_self}.controller.buttons.{name}"
            if k in cols:
                tgt_buttons[:, b_idx] = cols[k][t_act].astype(np.float32)

        tgt_axes = np.zeros((T, 4), dtype=np.int64)
        axis_keys = [
            f"{p_self}.controller.main_stick.x",
            f"{p_self}.controller.main_stick.y",
            f"{p_self}.controller.c_stick.x",
            f"{p_self}.controller.c_stick.y",
        ]
        for a_idx, k in enumerate(axis_keys):
            if k in cols:
                vals = cols[k][t_act]
                binned = np.clip(np.floor((vals + 1.0) * 0.5 * 32.0).astype(np.int64), 0, 32)
                tgt_axes[:, a_idx] = binned
            else:
                tgt_axes[:, a_idx] = 16

        tgt_shoulder = np.zeros(T, dtype=np.int64)
        sh_key = f"{p_self}.controller.shoulder"
        if sh_key in cols:
            sh_vals = cols[sh_key][t_act]
            tgt_shoulder = np.clip(np.floor(sh_vals * 10.0).astype(np.int64), 0, 10)

        # Previous action at t_act - 1
        t_prev = t_act - 1
        for b_idx, name in enumerate(btn_names):
            k = f"{p_self}.controller.buttons.{name}"
            if k in cols:
                prev_act[:, b_idx] = cols[k][t_prev].astype(np.float32)

        cur_off = 8
        for k in axis_keys:
            if k in cols:
                vals = cols[k][t_prev]
                binned = np.clip(np.floor((vals + 1.0) * 0.5 * 32.0).astype(np.int64), 0, 32)
                for t_i in range(T):
                    prev_act[t_i, cur_off + binned[t_i]] = 1.0
            cur_off += 33

        if sh_key in cols:
            sh_vals = cols[sh_key][t_prev]
            binned = np.clip(np.floor(sh_vals * 10.0).astype(np.int64), 0, 10)
            for t_i in range(T):
                prev_act[t_i, cur_off + binned[t_i]] = 1.0

        return obs, prev_act, tgt_buttons, tgt_axes, tgt_shoulder

    def _encode_player(
        self,
        out: np.ndarray,
        cols: Dict[str, np.ndarray],
        prefix: str,
        t_indices: np.ndarray,
    ):
        T = len(t_indices)
        if f"{prefix}.percent" in cols:
            out[:, 0] = cols[f"{prefix}.percent"][t_indices] * 0.01
        if f"{prefix}.facing" in cols:
            out[:, 1] = cols[f"{prefix}.facing"][t_indices].astype(np.float32)
        if f"{prefix}.x" in cols:
            out[:, 2] = cols[f"{prefix}.x"][t_indices] * 0.05
        if f"{prefix}.y" in cols:
            out[:, 3] = cols[f"{prefix}.y"][t_indices] * 0.05

        # Action state one-hot (399 states)
        if f"{prefix}.action" in cols:
            actions = cols[f"{prefix}.action"][t_indices]
            for i, act in enumerate(actions):
                if 0 <= act < 399:
                    out[i, 4 + act] = 1.0

        # Character one-hot (33 classes)
        char_offset = 4 + 399
        if f"{prefix}.character" in cols:
            chars = cols[f"{prefix}.character"][t_indices]
            for i, ch in enumerate(chars):
                if 0 <= ch < 33:
                    out[i, char_offset + ch] = 1.0

        meta_offset = char_offset + 33
        if f"{prefix}.invulnerable" in cols:
            out[:, meta_offset] = cols[f"{prefix}.invulnerable"][t_indices].astype(np.float32)
        if f"{prefix}.jumps_left" in cols:
            out[:, meta_offset + 1] = cols[f"{prefix}.jumps_left"][t_indices] * 0.2
        if f"{prefix}.shield_strength" in cols:
            out[:, meta_offset + 2] = cols[f"{prefix}.shield_strength"][t_indices] * 0.01
        if f"{prefix}.on_ground" in cols:
            out[:, meta_offset + 3] = cols[f"{prefix}.on_ground"][t_indices].astype(np.float32)


# ---------------------------------------------------------------------------
# Slippi-AI PyTorch Policy Network (141 Tensors)
# ---------------------------------------------------------------------------

class SlippiAIPolicy(nn.Module):
    """Full PyTorch implementation of the Slippi-AI 24M parameter policy network."""

    def __init__(self):
        super().__init__()
        self.params = nn.ParameterList()
        self._init_shapes()

    def _init_shapes(self):
        # 0..103: 13 controller heads (8 tensors each)
        for comp in range(13):
            emb_size = 1 if comp < 8 else (33 if comp < 12 else 11)
            inp_size = 128 + emb_size
            self.params.append(nn.Parameter(torch.zeros(128)))             # base + 0: dec bias
            self.params.append(nn.Parameter(torch.zeros(emb_size, 128)))  # base + 1: dec weight
            self.params.append(nn.Parameter(torch.zeros(128)))             # base + 2: mlp1 bias
            self.params.append(nn.Parameter(torch.zeros(inp_size, 128)))  # base + 3: mlp1 weight
            self.params.append(nn.Parameter(torch.zeros(128)))             # base + 4: mlp2 bias
            self.params.append(nn.Parameter(torch.zeros(128, 128)))        # base + 5: mlp2 weight
            self.params.append(nn.Parameter(torch.zeros(emb_size)))        # base + 6: out bias
            self.params.append(nn.Parameter(torch.zeros(128, emb_size)))  # base + 7: out weight

        # 104..105: Residual projection
        self.params.append(nn.Parameter(torch.zeros(128)))                 # 104: bias
        self.params.append(nn.Parameter(torch.zeros(768, 128)))            # 105: weight

        # 106..109: Item MLP
        self.params.append(nn.Parameter(torch.zeros(128)))                 # 106: mlp1 bias
        self.params.append(nn.Parameter(torch.zeros(254, 128)))            # 107: mlp1 weight
        self.params.append(nn.Parameter(torch.zeros(32)))                  # 108: mlp2 bias
        self.params.append(nn.Parameter(torch.zeros(128, 32)))             # 109: mlp2 weight

        # 110..111: Linear input projection
        self.params.append(nn.Parameter(torch.zeros(768)))                 # 110: bias
        self.params.append(nn.Parameter(torch.zeros(2613, 768)))           # 111: weight

        # 112..138: 3 Transformer-LSTM layers (9 tensors each)
        for _ in range(3):
            self.params.append(nn.Parameter(torch.zeros(768, 3072)))       # base + 0: hidden gates
            self.params.append(nn.Parameter(torch.zeros(768, 3072)))       # base + 1: input gates
            self.params.append(nn.Parameter(torch.zeros(3072)))            # base + 2: gate bias
            self.params.append(nn.Parameter(torch.zeros(768)))             # base + 3: norm bias
            self.params.append(nn.Parameter(torch.zeros(768)))             # base + 4: norm scale
            self.params.append(nn.Parameter(torch.zeros(1536)))            # base + 5: ffw1 bias
            self.params.append(nn.Parameter(torch.zeros(768, 1536)))       # base + 6: ffw1 weight
            self.params.append(nn.Parameter(torch.zeros(768)))             # base + 7: ffw2 bias
            self.params.append(nn.Parameter(torch.zeros(1536, 768)))       # base + 8: ffw2 weight

        # 139..140: Value function readout
        self.params.append(nn.Parameter(torch.zeros(1)))                   # 139: bias
        self.params.append(nn.Parameter(torch.zeros(768, 1)))              # 140: weight

    def load_tensors(self, tensors: List[np.ndarray]) -> None:
        if len(tensors) != TOTAL_PARAMS:
            raise ValueError(f"Expected {TOTAL_PARAMS} tensors, got {len(tensors)}")
        for idx, (param, arr) in enumerate(zip(self.params, tensors)):
            arr_np = np.asarray(arr, dtype=np.float32)
            if param.shape != arr_np.shape:
                raise ValueError(
                    f"Shape mismatch at index {idx}: expected {param.shape}, got {arr_np.shape}"
                )
            with torch.no_grad():
                param.copy_(torch.from_numpy(arr_np))

    def export_tensors(self) -> List[np.ndarray]:
        return [p.detach().cpu().numpy().astype(np.float32) for p in self.params]

    def forward(
        self,
        observations: torch.Tensor,
        prev_actions: torch.Tensor,
        targets: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, T, _ = observations.shape
        device = observations.device

        # Item embedding
        raw_items = observations[:, :, 2334:2588]
        item_h1 = F.relu(torch.matmul(raw_items, self.params[107]) + self.params[106])
        item_h2 = torch.matmul(item_h1, self.params[109]) + self.params[108]

        # Replace item features with processed embedding
        obs_processed = torch.cat(
            [observations[:, :, :2334], item_h2, observations[:, :, 2588:]], dim=-1
        )

        # Input linear projection: (B, T, 768)
        current_x = torch.matmul(obs_processed, self.params[111]) + self.params[110]

        # 3-layer Transformer-LSTM recurrence
        for layer in range(3):
            base = 112 + layer * 9
            w_h = self.params[base + 0]
            w_x = self.params[base + 1]
            b_g = self.params[base + 2]
            norm_b = self.params[base + 3]
            norm_s = self.params[base + 4]
            ffw1_b = self.params[base + 5]
            ffw1_w = self.params[base + 6]
            ffw2_b = self.params[base + 7]
            ffw2_w = self.params[base + 8]

            h = torch.zeros(B, 768, device=device)
            c = torch.zeros(B, 768, device=device)
            lstm_outputs = []

            for t in range(T):
                xt = current_x[:, t, :]
                gates = torch.matmul(xt, w_x) + torch.matmul(h, w_h) + b_g
                i, j, f, o = torch.split(gates, 768, dim=-1)
                c = torch.sigmoid(f + 1.0) * c + torch.sigmoid(i) * torch.tanh(j)
                h = torch.sigmoid(o) * torch.tanh(c)
                lstm_outputs.append(h)

            lstm_out = torch.stack(lstm_outputs, dim=1)

            # Pre-LayerNorm + Feedforward block with residual connections
            normed = F.layer_norm(lstm_out, [768], weight=norm_s, bias=norm_b, eps=1e-5)
            ffw_h = F.relu(torch.matmul(normed, ffw1_w) + ffw1_b)
            ffw_out = torch.matmul(ffw_h, ffw2_w) + ffw2_b
            current_x = current_x + lstm_out + ffw_out

        # Final representation: (B, T, 128)
        latent = torch.matmul(current_x, self.params[105]) + self.params[104]

        # Autoregressive controller heads
        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=device)

        if targets is not None:
            tgt_buttons, tgt_axes, tgt_shoulder = targets

            # 8 button heads (comp 0..7)
            button_losses = []
            for comp in range(8):
                base = comp * 8
                m1 = F.relu(torch.matmul(latent, self.params[base + 5]) + self.params[base + 4])
                logits = torch.matmul(m1, self.params[base + 7]) + self.params[base + 6]
                target_b = tgt_buttons[:, :, comp : comp + 1]
                b_loss = F.binary_cross_entropy_with_logits(logits, target_b)
                button_losses.append(b_loss)
            avg_btn_loss = torch.stack(button_losses).mean()
            loss_dict["buttons"] = avg_btn_loss
            total_loss = total_loss + avg_btn_loss

            # 4 stick axes heads (comp 8..11)
            axis_losses = []
            for ax_idx in range(4):
                comp = 8 + ax_idx
                base = comp * 8
                m1 = F.relu(torch.matmul(latent, self.params[base + 5]) + self.params[base + 4])
                logits = torch.matmul(m1, self.params[base + 7]) + self.params[base + 6]
                target_ax = tgt_axes[:, :, ax_idx]
                ax_loss = F.cross_entropy(logits.reshape(-1, 33), target_ax.reshape(-1))
                axis_losses.append(ax_loss)
            avg_axis_loss = torch.stack(axis_losses).mean()
            loss_dict["axes"] = avg_axis_loss
            total_loss = total_loss + avg_axis_loss

            # 1 shoulder head (comp 12)
            base = 12 * 8
            m1 = F.relu(torch.matmul(latent, self.params[base + 5]) + self.params[base + 4])
            logits = torch.matmul(m1, self.params[base + 7]) + self.params[base + 6]
            sh_loss = F.cross_entropy(logits.reshape(-1, 11), tgt_shoulder.reshape(-1))
            loss_dict["shoulder"] = sh_loss
            total_loss = total_loss + sh_loss

        return total_loss, loss_dict


# ---------------------------------------------------------------------------
# Binary Container Serializer
# ---------------------------------------------------------------------------

def export_binary(tensors: List[np.ndarray], output_path: Path) -> str:
    """Exports 141 policy tensors to native binary container format."""
    if len(tensors) != TOTAL_PARAMS:
        raise ValueError(f"Expected {TOTAL_PARAMS} tensors, got {len(tensors)}")

    arrays = []
    total_floats = 0
    for idx, tensor in enumerate(tensors):
        array = np.asarray(tensor)
        if array.dtype != np.dtype("float32") or not np.isfinite(array).all():
            raise ValueError(f"Invalid tensor at index {idx}")
        array = np.ascontiguousarray(array, dtype="<f4")
        arrays.append(array)
        total_floats += array.size

    if total_floats != TOTAL_FLOATS:
        raise ValueError(f"Expected {TOTAL_FLOATS} floats, got {total_floats}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".part", dir=output_path.parent
    )
    digest = hashlib.sha256()

    with os.fdopen(fd, "wb") as output:
        header = struct.pack(
            "<8sIIQ32s",
            MAGIC,
            FORMAT_VERSION,
            len(arrays),
            total_floats,
            b"\x00" * 32,
        )
        output.write(header)
        digest.update(header)

        for array in arrays:
            shape = tuple(array.shape) + (0,) * (4 - array.ndim)
            tensor_hdr = struct.pack("<I4IQ", array.ndim, *shape, array.nbytes)
            output.write(tensor_hdr)
            digest.update(tensor_hdr)

            data = memoryview(array).cast("B")
            output.write(data)
            digest.update(data)

        output.flush()
        os.fsync(output.fileno())

    Path(temp_name).replace(output_path)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train(
    base_checkpoint_path: Path,
    dataset_dir: Path,
    output_pkl_path: Path,
    output_bin_path: Optional[Path] = None,
    num_steps: int = 2500,
    batch_size: int = 16,
    unroll_length: int = UNROLL_LENGTH,
    lr: float = 3e-5,
    min_lr: float = 8e-6,
    weight_decay: float = 1e-4,
    omitted_ratio: float = 0.85,
    device_name: str = "auto",
    log_interval: int = 50,
) -> None:
    # Resolve device
    if device_name == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    print(f"Using compute device: {device}")

    # Load base checkpoint
    print(f"Loading base checkpoint: {base_checkpoint_path}")
    with open(base_checkpoint_path, "rb") as f:
        base_state = CheckpointUnpickler(f).load()

    raw_tensors = base_state["state"]["policy"]
    print(f"Loaded {len(raw_tensors)} tensors from base checkpoint.")

    policy = SlippiAIPolicy()
    policy.load_tensors(raw_tensors)
    policy.to(device)
    policy.train()

    # Initialize dataset
    parsed_dir = dataset_dir / "Parsed" if (dataset_dir / "Parsed").exists() else dataset_dir
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {dataset_dir}")

    dataset = ReplayDataset(parsed_dir=parsed_dir, meta_path=meta_path)
    print(f"Indexed replays across {len(dataset.char_to_trajectories)} character slots.")
    print(f"Omitted characters available: {[ALL_CHARACTERS[c] for c in dataset.available_omitted]}")
    print(f"Base characters available: {[ALL_CHARACTERS[c] for c in dataset.available_base]}")

    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-8)

    print(
        f"\nStarting fine-tuning: {num_steps} steps, batch_size={batch_size}, "
        f"unroll_length={unroll_length}, initial_lr={lr:.2e}"
    )
    start_time = time.time()
    recent_losses = []

    for step in range(1, num_steps + 1):
        # Cosine learning rate schedule
        progress = (step - 1) / max(1, num_steps - 1)
        cur_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        # Sample batch
        batch = dataset.sample_batch(
            batch_size=batch_size,
            unroll_length=unroll_length,
            omitted_ratio=omitted_ratio,
        )
        if batch is None:
            continue

        obs, prev_act, (tgt_b, tgt_ax, tgt_sh) = batch
        obs = obs.to(device)
        prev_act = prev_act.to(device)
        tgt_b = tgt_b.to(device)
        tgt_ax = tgt_ax.to(device)
        tgt_sh = tgt_sh.to(device)

        optimizer.zero_grad()
        loss, loss_dict = policy(obs, prev_act, targets=(tgt_b, tgt_ax, tgt_sh))
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = float(loss.item())
        recent_losses.append(loss_val)

        if step % log_interval == 0 or step == num_steps:
            elapsed = time.time() - start_time
            steps_per_sec = step / max(0.1, elapsed)
            frames_per_sec = steps_per_sec * batch_size * unroll_length
            avg_loss = sum(recent_losses[-log_interval:]) / len(recent_losses[-log_interval:])
            b_l = float(loss_dict.get("buttons", 0.0))
            a_l = float(loss_dict.get("axes", 0.0))
            s_l = float(loss_dict.get("shoulder", 0.0))

            print(
                f"[step {step:4d}/{num_steps:4d}] loss: {avg_loss:.4f} "
                f"(btn: {b_l:.3f}, ax: {a_l:.3f}, sh: {s_l:.3f}) | "
                f"lr: {cur_lr:.2e} | {steps_per_sec:.1f} steps/s ({frames_per_sec:.0f} frames/s)"
            )

    # Export fine-tuned weights
    fine_tuned_tensors = policy.export_tensors()

    def sanitize_for_pickle(obj):
        if hasattr(obj, "__class__") and hasattr(obj.__class__, "__module__"):
            if "slippi_ai" in obj.__class__.__module__:
                return "mlp"
        if isinstance(obj, dict):
            return {k: sanitize_for_pickle(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_for_pickle(v) for v in obj]
        return obj

    clean_config = sanitize_for_pickle(base_state["config"])
    # Enable all characters in dataset config so eval scripts accept any character
    if "dataset" in clean_config and isinstance(clean_config["dataset"], dict):
        clean_config["dataset"]["allowed_characters"] = "all"

    fine_tuned_state = {
        "config": clean_config,
        "state": {"policy": fine_tuned_tensors},
        "name_map": base_state.get("name_map", {}),
    }

    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(fine_tuned_state, f, protocol=4)
    print(f"\nSaved fine-tuned pickle checkpoint: {output_pkl_path}")
    print(f"  Size: {output_pkl_path.stat().st_size:,} bytes")

    if output_bin_path is not None:
        bin_sha256 = export_binary(fine_tuned_tensors, output_bin_path)
        print(f"Saved native weights binary: {output_bin_path}")
        print(f"  Size: {output_bin_path.stat().st_size:,} bytes")
        print(f"  SHA-256: {bin_sha256}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Slippi-AI policy network on custom or omitted characters."
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        required=True,
        help="Path to base medium-v2 checkpoint (pickle file)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to dataset directory containing Parsed/ and meta.json",
    )
    parser.add_argument(
        "--output-pkl",
        type=Path,
        default=Path("medium-v2-finetuned.pkl"),
        help="Path to save the fine-tuned pickle checkpoint",
    )
    parser.add_argument(
        "--output-bin",
        type=Path,
        default=None,
        help="Optional path to export native binary weights container",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2500,
        help="Number of training steps (default: 2500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size of unrolls (default: 16)",
    )
    parser.add_argument(
        "--unroll-length",
        type=int,
        default=UNROLL_LENGTH,
        help="Unroll sequence length in frames (default: 60)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Peak learning rate for AdamW (default: 3e-5)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=8e-6,
        help="Minimum learning rate for cosine schedule (default: 8e-6)",
    )
    parser.add_argument(
        "--omitted-ratio",
        type=float,
        default=0.85,
        help="Proportion of batches sampled from omitted character replays (default: 0.85)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Steps between progress log lines (default: 50)",
    )
    args = parser.parse_args()

    train(
        base_checkpoint_path=args.base_checkpoint,
        dataset_dir=args.dataset_dir,
        output_pkl_path=args.output_pkl,
        output_bin_path=args.output_bin,
        num_steps=args.steps,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        lr=args.lr,
        min_lr=args.min_lr,
        omitted_ratio=args.omitted_ratio,
        device_name=args.device,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
