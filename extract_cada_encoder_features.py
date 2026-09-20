#!/usr/bin/env python3
"""Extract mean-pooled CADA encoder embeddings on RouteFinder test npz.

Run from the `routefinder/` directory on the remote machine (GPU):

    python extract_cada_encoder_features.py --n_size 50
    python extract_cada_encoder_features.py --n_size 100

Default checkpoints match the CADA README:
    CADA/50/result/2024-1111-1139/checkpoint-300.pt
    CADA/100/result/2024-1121-1355/checkpoint-300.pt

Outputs
    encoder_embeddings/cada/all_{size}_mean_cada_embeddings.npz
    encoder_embeddings/cada/{type}_{size}_mean_cada_embeddings.npz
    optional CSV: problem_id, type, cada_0 ... cada_{H-1}

Pooling is mean over encoder tokens (depot + customers, dim 128).
This is encoder-only: no decoder, no 8-fold augmentation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

PROBLEM_TYPES = [
    "cvrp",
    "ovrp",
    "ovrpb",
    "ovrpbl",
    "ovrpbltw",
    "ovrpbtw",
    "ovrpl",
    "ovrpltw",
    "ovrptw",
    "vrpb",
    "vrpbl",
    "vrpbltw",
    "vrpbtw",
    "vrpl",
    "vrpltw",
    "vrptw",
]

DEFAULT_CKPT = {
    50: "CADA/50/result/2024-1111-1139/checkpoint-300.pt",
    100: "CADA/100/result/2024-1121-1355/checkpoint-300.pt",
}


def _add_cada_to_path(n_size: int, script_dir: Path) -> Path:
    cada_root = script_dir / "CADA"
    size_dir = cada_root / str(n_size)
    if not size_dir.is_dir():
        raise FileNotFoundError(f"CADA size dir not found: {size_dir}")
    # size dir first so `model` / `envs` resolve to 50 vs 100
    for p in (str(size_dir), str(cada_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return size_dir


def _constraint_tag(problem: str, batch: int, size: int) -> torch.Tensor:
    """Same C / O / TW / L / B packing as CADA env.dataset()."""
    keep_mask = torch.zeros((batch, 5), dtype=torch.bool)
    for i, tag in enumerate(["c", "o", "tw", "l", "b"]):
        keep_mask[:, i] = tag in problem
    keep_mask[:, 0:1] = ~keep_mask[:, 1:2]
    size_col = torch.full((batch, 1), size / 2000.0, dtype=torch.float32)
    return torch.cat((keep_mask.float(), size_col), dim=-1)


def _load_model(n_size: int, checkpoint: str, device: torch.device, size_dir: Path):
    from model import VRPModel

    cfg_path = size_dir / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    model_params = dict(cfg["model_params"])
    model_params.setdefault("p_num", 5 if n_size == 50 else 1)
    model_params["sqrt_embedding_dim"] = model_params["embedding_dim"] ** 0.5
    args = SimpleNamespace(model_params=model_params)
    model = VRPModel(args)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, int(model_params["embedding_dim"])


def _pool(h: torch.Tensor, pooling: str) -> torch.Tensor:
    # h: [B, N+1, H]
    if pooling == "mean":
        return h.mean(dim=1)
    if pooling == "mean_customers":
        return h[:, 1:, :].mean(dim=1)
    raise ValueError(pooling)


def extract(
    n_size: int,
    checkpoint: str,
    data_root: str,
    problem: str,
    batch_size: int,
    device_str: str,
    pooling: str,
    out_dir: str,
    csv_path: str,
):
    script_dir = Path(__file__).resolve().parent
    size_dir = _add_cada_to_path(n_size, script_dir)

    from envs.fill_missing_fields import fill_missing_vrp_fields
    from utils.functions import load_npz_to_tensordict

    if device_str.startswith("cuda") and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA not available, using CPU")

    ckpt_path = checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = str(script_dir / ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Download CADA checkpoints (HuggingFace Goodyee/CaDA) into CADA/{50,100}/result/."
        )

    print("Loading", ckpt_path)
    model, hidden = _load_model(n_size, ckpt_path, device, size_dir)

    types = PROBLEM_TYPES if problem == "all" else [problem]
    os.makedirs(out_dir, exist_ok=True)

    all_emb, all_ids, all_types = [], [], []

    with torch.inference_mode():
        for ptype in tqdm(types, desc="problems"):
            npz_path = os.path.join(data_root, ptype, "test", f"{n_size}.npz")
            if not os.path.exists(npz_path):
                print(f"[WARN] missing {npz_path}")
                continue

            td_all = load_npz_to_tensordict(npz_path)
            td_all = fill_missing_vrp_fields(td_all)
            n = int(td_all.batch_size[0])
            chunks = []
            offset = 0
            for start in range(0, n, batch_size):
                td = td_all[start : start + batch_size].to(device)
                bsz = int(td.batch_size[0])
                td["p_s_tag"] = _constraint_tag(ptype, bsz, n_size).to(device)
                prompt = model.prompt_net(td)["prompt"]
                node_embed = model.encoder(td, prompt)
                pooled = _pool(node_embed, pooling)
                chunks.append(pooled.detach().cpu().float().numpy())
                offset += bsz

            emb = np.concatenate(chunks, axis=0)
            ids = [f"{ptype}_{n_size}_{i:04d}" for i in range(emb.shape[0])]
            type_out = os.path.join(out_dir, f"{ptype}_{n_size}_{pooling}_cada_embeddings.npz")
            np.savez_compressed(
                type_out,
                problem_id=np.array(ids),
                type=np.array([ptype] * len(ids)),
                embedding=emb,
            )
            print(f"Saved {type_out} {emb.shape}")
            all_emb.append(emb)
            all_ids.extend(ids)
            all_types.extend([ptype] * len(ids))

    if not all_emb:
        raise FileNotFoundError(f"No datasets under {data_root}")

    all_emb = np.concatenate(all_emb, axis=0)
    all_out = os.path.join(out_dir, f"all_{n_size}_{pooling}_cada_embeddings.npz")
    np.savez_compressed(
        all_out,
        problem_id=np.array(all_ids),
        type=np.array(all_types),
        embedding=all_emb,
    )
    print(f"\n[DONE] {all_out} {all_emb.shape}")

    if csv_path:
        import pandas as pd

        cols = [f"cada_{i}" for i in range(all_emb.shape[1])]
        df = pd.DataFrame(all_emb, columns=cols)
        df.insert(0, "type", all_types)
        df.insert(0, "problem_id", all_ids)
        csv_parent = os.path.dirname(os.path.abspath(csv_path))
        if csv_parent:
            os.makedirs(csv_parent, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print("CSV", csv_path, df.shape)

    return all_out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_size", type=int, choices=[50, 100], default=50)
    p.add_argument("--checkpoint", type=str, default="", help="Override default CADA checkpoint path")
    p.add_argument("--data_root", type=str, default="data", help="RouteFinder data/{type}/test/{size}.npz")
    p.add_argument("--problem", type=str, default="all")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--pooling", choices=["mean", "mean_customers"], default="mean")
    p.add_argument("--out_dir", type=str, default="encoder_embeddings/cada")
    p.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV path, e.g. ../cada_features_size50_mean_pooling.csv",
    )
    args = p.parse_args()
    ckpt = args.checkpoint or DEFAULT_CKPT[args.n_size]
    csv_path = args.csv
    if not csv_path:
        csv_path = f"../cada_features_size{args.n_size}_mean_pooling.csv"
    extract(
        n_size=args.n_size,
        checkpoint=ckpt,
        data_root=args.data_root,
        problem=args.problem,
        batch_size=args.batch_size,
        device_str=args.device,
        pooling=args.pooling,
        out_dir=args.out_dir,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()
