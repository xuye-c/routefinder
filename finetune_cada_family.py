#!/usr/bin/env python3
"""Fine-tune a pretrained CADA checkpoint into 6 family specialists.

Families match constraint_family_cluster_6way_{50,100}.csv:

  0 open_TW              ovrp*tw
  1 closed_TW            vrp*tw
  2 open_backhaul_noTW   ovrpb, ovrpbl
  3 open_plain_noTW      ovrp, ovrpl
  4 closed_backhaul_noTW vrpb, vrpbl
  5 closed_plain_noTW    cvrp, vrpl

Run from the routefinder/ directory on the cluster (needs CADA/{50,100}
on PYTHONPATH via this script). Example:

    python -u finetune_cada_family.py --n_size 50
    python -u finetune_cada_family.py --n_size 100 --clusters 0,1,2,3,4,5

Each family reloads the same pretrained checkpoint, trains on that
family's instances (stratified holdout), writes checkpoints + metrics
under CADA/{n}/result/family-ft-..., and prints a banner so one Slurm
.out shows all six runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tensordict import TensorDict

FAMILY_NAME = {
    0: "open_TW",
    1: "closed_TW",
    2: "open_backhaul_noTW",
    3: "open_plain_noTW",
    4: "closed_backhaul_noTW",
    5: "closed_plain_noTW",
}

FAMILY_TYPES = {
    0: ("ovrpbltw", "ovrpbtw", "ovrpltw", "ovrptw"),
    1: ("vrpbltw", "vrpbtw", "vrpltw", "vrptw"),
    2: ("ovrpb", "ovrpbl"),
    3: ("ovrp", "ovrpl"),
    4: ("vrpb", "vrpbl"),
    5: ("cvrp", "vrpl"),
}

ALL_TEST_PROBLEMS = (
    "cvrp",
    "ovrp",
    "vrpb",
    "vrpl",
    "vrptw",
    "ovrptw",
    "ovrpb",
    "ovrpl",
    "ovrpbl",
    "ovrpbtw",
    "ovrpltw",
    "ovrpbltw",
    "vrpbl",
    "vrpbtw",
    "vrpltw",
    "vrpbltw",
)

DEFAULT_CKPT_DIR = {
    50: "2024-1111-1139",
    100: "2024-1121-1355",
}

KEEP_KEYS = (
    "locs",
    "demand_linehaul",
    "demand_backhaul",
    "distance_limit",
    "service_time",
    "open_route",
    "time_windows",
    "vehicle_capacity",
    "capacity_original",
    "speed",
    "p_s_tag",
    "opt_cost",
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune CADA on 6 constraint families")
    p.add_argument("--n_size", type=int, required=True, choices=[50, 100])
    p.add_argument(
        "--clusters",
        type=str,
        default="0,1,2,3,4,5",
        help="Comma-separated family ids, default all six",
    )
    p.add_argument(
        "--cluster_csv",
        type=str,
        default=None,
        help="problem_id,type,cluster CSV. Default: constraint_family_cluster_6way_{n}.csv",
    )
    p.add_argument("--epoch", type=int, default=300, help="Pretrained checkpoint epoch")
    p.add_argument(
        "--path_id",
        type=str,
        default=None,
        help="Checkpoint folder under CADA/{n}/result/, e.g. 2024-1111-1139",
    )
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--lr_decay_epoch", type=int, default=0, help="0 => epochs-2")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--loss", type=str, default="rl", choices=["rl", "po"])
    p.add_argument("--po_alpha", type=float, default=0.05)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--skip_zero_shot", action="store_true")
    p.add_argument(
        "--result_root",
        type=str,
        default=None,
        help="Parent dir for all family runs. Default: CADA/{n}/result/family-ft-n{n}-{stamp}",
    )
    return p.parse_args()


def parse_clusters(s: str):
    ids = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    for c in ids:
        if c not in FAMILY_NAME:
            raise ValueError(f"unknown family id {c}; expected 0-5")
    return ids


def routefinder_root() -> Path:
    return Path(__file__).resolve().parent


def add_cada_to_path(n_size: int) -> Path:
    root = routefinder_root()
    size_dir = root / "CADA" / str(n_size)
    if not size_dir.is_dir():
        raise FileNotFoundError(f"CADA size dir not found: {size_dir}")
    for p in (str(size_dir), str(root / "CADA")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    return size_dir


def resolve_data_dir(explicit):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"data_dir not found: {path}")
        return path
    root = routefinder_root()
    candidates = [
        root / "data",
        root / "Routing-POLAR-master" / "data",
        Path.home() / "routefinder" / "data",
        Path.home() / "routefinder" / "Routing-POLAR-master" / "data",
    ]
    for c in candidates:
        if (c / "cvrp" / "test").is_dir():
            return c.resolve()
    raise FileNotFoundError(
        "Could not find RouteFinder data/{type}/test. Pass --data_dir. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def resolve_cluster_csv(n_size, explicit):
    name = f"constraint_family_cluster_6way_{n_size}.csv"
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"cluster csv not found: {path}")
        return path
    candidates = [
        routefinder_root() / name,
        routefinder_root().parent / name,
        Path.cwd() / name,
        Path.home() / "routefinder" / name,
        Path.home() / "wfl_sw" / name,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def constraint_tag(problem: str, batch: int, size: int) -> torch.Tensor:
    keep_mask = torch.zeros((batch, 5), dtype=torch.bool)
    for i, tag in enumerate(["c", "o", "tw", "l", "b"]):
        keep_mask[:, i] = tag in problem
    keep_mask[:, 0:1] = ~keep_mask[:, 1:2]
    size_col = torch.full((batch, 1), size / 2000.0, dtype=torch.float32)
    return torch.cat((keep_mask.float(), size_col), dim=-1)


def parse_problem_id(problem_id, n_size):
    prefix = f"_{n_size}_"
    if prefix not in problem_id:
        raise ValueError(f"problem_id {problem_id} does not match n_size={n_size}")
    ptype, idx_s = problem_id.rsplit(prefix, 1)
    return ptype, int(idx_s)


def load_cluster_rows(csv_path, n_size, cluster):
    if csv_path is None:
        rows = []
        for ptype in FAMILY_TYPES[cluster]:
            for idx in range(1000):
                rows.append(
                    {
                        "problem_id": f"{ptype}_{n_size}_{idx:04d}",
                        "type": ptype,
                        "index": idx,
                        "cluster": cluster,
                    }
                )
        return rows

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            if int(rec["cluster"]) != cluster:
                continue
            ptype, idx = parse_problem_id(rec["problem_id"], n_size)
            if rec["type"] != ptype:
                raise ValueError(
                    f"type mismatch for {rec['problem_id']}: csv type={rec['type']}"
                )
            rows.append(
                {
                    "problem_id": rec["problem_id"],
                    "type": ptype,
                    "index": idx,
                    "cluster": cluster,
                }
            )
    if not rows:
        raise ValueError(f"No rows for cluster={cluster} in {csv_path}")
    return rows


def stratified_split(rows, holdout_frac, seed):
    by_type = defaultdict(list)
    for i, row in enumerate(rows):
        by_type[row["type"]].append(i)
    rng = random.Random(seed)
    holdout = set()
    for indices in by_type.values():
        rng.shuffle(indices)
        n_hold = max(1, int(round(len(indices) * holdout_frac)))
        n_hold = min(n_hold, len(indices) - 1) if len(indices) > 1 else 0
        holdout.update(indices[:n_hold])
    train_idx = [i for i in range(len(rows)) if i not in holdout]
    hold_idx = [i for i in range(len(rows)) if i in holdout]
    if not train_idx:
        raise ValueError("Train split is empty; lower --holdout_frac")
    return train_idx, hold_idx


def select_keys(td):
    out = {}
    for key in KEEP_KEYS:
        if key in td.keys():
            out[key] = td[key]
    return TensorDict(out, batch_size=td.batch_size)


def subset_td(td, indices):
    sample = next(v for v in td.values() if torch.is_tensor(v))
    if torch.is_tensor(indices):
        indices = indices.to(device="cpu", dtype=torch.long)
    else:
        indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
    indices = indices.to(sample.device)
    return TensorDict(
        {k: td[k][indices] for k in td.keys() if torch.is_tensor(td[k])},
        batch_size=[int(indices.numel())],
    )


def stack_tds(pieces):
    keys = list(pieces[0].keys())
    for td in pieces[1:]:
        keys = [k for k in keys if k in td.keys()]
    batch = sum(td.batch_size[0] for td in pieces)
    return TensorDict(
        {k: torch.cat([td[k] for td in pieces], dim=0) for k in keys},
        batch_size=[batch],
    )


def maybe_opt_cost(td, data_dir: Path, ptype: str, n_size: int):
    if "opt_cost" in td.keys() and not torch.all(td["opt_cost"] == 0):
        return td
    sol = data_dir / ptype / "test" / f"{n_size}_sol_pyvrp.npz"
    if not sol.is_file():
        return td
    blob = np.load(sol)
    key = "costs" if "costs" in blob.files else ("cost" if "cost" in blob.files else None)
    if key is None:
        return td
    costs = torch.as_tensor(blob[key], dtype=torch.float32)
    n = td.batch_size[0]
    if costs.numel() < n:
        return td
    td["opt_cost"] = costs[:n].abs()
    return td


def load_family_tensordict(rows, n_size, data_dir: Path, fill_missing_vrp_fields, load_npz_to_tensordict):
    by_type = defaultdict(list)
    for i, row in enumerate(rows):
        by_type[row["type"]].append((row["index"], i))

    ordered = [None] * len(rows)
    for ptype, items in by_type.items():
        npz_path = data_dir / ptype / "test" / f"{n_size}.npz"
        if not npz_path.is_file():
            raise FileNotFoundError(f"missing {npz_path}")
        td = load_npz_to_tensordict(str(npz_path)).to("cpu")
        td = fill_missing_vrp_fields(td)
        td = maybe_opt_cost(td, data_dir, ptype, n_size)
        n_inst = td.batch_size[0]
        tags = constraint_tag(ptype, n_inst, n_size)
        td["p_s_tag"] = tags
        td = select_keys(td)
        for local_idx, row_i in items:
            if local_idx < 0 or local_idx >= n_inst:
                raise IndexError(f"{ptype} index {local_idx} out of range (n={n_inst})")
            ordered[row_i] = subset_td(td, [local_idx])
    return stack_tds(ordered)


def compute_po_loss(reward, log_likelihood, po_alpha):
    preference = reward[:, :, None] > reward[:, None, :]
    log_prob = po_alpha * log_likelihood
    log_prob_pair = log_prob[:, :, None] - log_prob[:, None, :]
    return -torch.mean(F.logsigmoid(log_prob_pair) * preference)


def gap_mean(cost, opt):
    if opt is None:
        return None
    scale = 1000.0
    agree = torch.round(cost * scale) == torch.round(opt * scale)
    gap = (cost - opt) * 100.0 / opt.clamp(min=1e-30)
    gap = torch.where(agree, torch.zeros_like(gap), gap)
    return float(gap.mean().item())


@torch.inference_mode()
def evaluate(model, env, td_all, device, augmentation, batch_size, problem_ids=None):
    model.eval()
    n = td_all.batch_size[0]
    costs, aug_costs, opts = [], [], []
    has_opt = "opt_cost" in td_all.keys()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        inp = subset_td(td_all, list(range(start, end)))
        p_s_tag = inp["p_s_tag"].clone()
        opt_cost = inp["opt_cost"].clone() if has_opt else None
        td = env.reset(td=inp.to(device))
        td["p_s_tag"] = p_s_tag.to(device)
        td_aug = augmentation(td)
        out = model(td_aug, env)
        bsz = inp.batch_size[0]
        reward = out["reward"].view(-1, augmentation.num_augment, bsz)
        all_reward, _ = reward.max(dim=0)
        score = -all_reward[0, :].float()
        aug_score = -all_reward.max(dim=0).values.float()
        costs.append(score.detach().cpu())
        aug_costs.append(aug_score.detach().cpu())
        if opt_cost is not None:
            opts.append(opt_cost.cpu())
        del td, td_aug, out, reward

    cost = torch.cat(costs)
    aug_cost = torch.cat(aug_costs)
    opt = torch.cat(opts) if opts else None
    metrics = {
        "n": int(n),
        "no_aug_cost": float(cost.mean().item()),
        "aug_cost": float(aug_cost.mean().item()),
        "no_aug_gap": gap_mean(cost.abs(), opt.abs()) if opt is not None else None,
        "aug_gap": gap_mean(aug_cost.abs(), opt.abs()) if opt is not None else None,
    }
    per_row = []
    for i in range(n):
        rec = {
            "problem_id": problem_ids[i] if problem_ids is not None else str(i),
            "cost": float(cost[i].item()),
            "aug_cost": float(aug_cost[i].item()),
        }
        if opt is not None:
            rec["opt_cost"] = float(opt[i].item())
        per_row.append(rec)
    return metrics, per_row


def log_metrics(prefix, metrics):
    gap = metrics["no_aug_gap"]
    agap = metrics["aug_gap"]
    gap_s = f"gap={gap:.3f}%" if gap is not None else "gap=NA"
    agap_s = f"aug_gap={agap:.3f}%" if agap is not None else "aug_gap=NA"
    log(
        f"{prefix} n={metrics['n']} "
        f"cost={metrics['no_aug_cost']:.4f} {gap_s} | "
        f"aug_cost={metrics['aug_cost']:.4f} {agap_s}"
    )


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def train_one_epoch(model, env, optimizer, td_train, device, args, epoch, clip_grad_norms):
    model.train()
    n = td_train.batch_size[0]
    perm = torch.randperm(n, device="cpu")
    losses, costs = [], []
    seen = 0
    while seen < n:
        end = min(seen + args.batch_size, n)
        batch = subset_td(td_train, perm[seen:end].cpu())
        p_s_tag = batch["p_s_tag"].clone()
        td = env.reset(td=batch.to(device))
        td["p_s_tag"] = p_s_tag.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(td, env)
        batch_n = td.batch_size[0]
        reward = out["reward"].view(-1, batch_n)
        log_likelihood = out["log_likelihood"].view(-1, batch_n)
        if args.loss == "po":
            loss = compute_po_loss(reward, log_likelihood, args.po_alpha)
        else:
            advantage = reward - reward.mean(dim=0, keepdims=True)
            loss = -(advantage * log_likelihood).mean()
        score_mean = (-reward).max(dim=0).values.mean()
        loss.backward()
        clip_grad_norms(optimizer.param_groups, args.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
        costs.append(float(score_mean.item()))
        seen = end
        del td, out, reward, log_likelihood, loss
    log(
        f"Epoch {epoch:03d} train loss={np.mean(losses):.4f} "
        f"best_cost={np.mean(costs):.4f} seen={n}"
    )
    return float(np.mean(losses)), float(np.mean(costs))


def load_pretrained(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    del ckpt
    log(f"Loaded pretrained weights from {ckpt_path}")


def run_one_family(cluster, args, shared):
    name = FAMILY_NAME[cluster]
    fam_dir = os.path.join(args.result_root, f"family{cluster}-{name}")
    os.makedirs(fam_dir, exist_ok=True)

    log("")
    log("=" * 78)
    log(f"FAMILY {cluster}/5  {name}")
    log("=" * 78)

    rows = load_cluster_rows(args.cluster_csv, args.n_size, cluster)
    train_idx, hold_idx = stratified_split(rows, args.holdout_frac, args.seed)
    log(
        f"cluster size={len(rows)} types={sorted({r['type'] for r in rows})} "
        f"train={len(train_idx)} holdout={len(hold_idx)}"
    )

    env = shared["env_cls"](**args.env)
    td_all = load_family_tensordict(
        rows,
        args.n_size,
        Path(args.data_dir),
        shared["fill_missing_vrp_fields"],
        shared["load_npz_to_tensordict"],
    )
    td_train = subset_td(td_all, train_idx)
    td_hold = subset_td(td_all, hold_idx)
    hold_ids = [rows[i]["problem_id"] for i in hold_idx]

    save_json(
        os.path.join(fam_dir, "split.json"),
        {
            "n_size": args.n_size,
            "cluster": cluster,
            "family": name,
            "csv": str(args.cluster_csv) if args.cluster_csv else None,
            "holdout_frac": args.holdout_frac,
            "seed": args.seed,
            "types": sorted({r["type"] for r in rows}),
            "train_ids": [rows[i]["problem_id"] for i in train_idx],
            "holdout_ids": hold_ids,
        },
    )

    model = shared["model_cls"](args).to(args.device)
    load_pretrained(model, args.ckpt_path, args.device)
    augmentation = shared["aug_cls"]()
    history = []

    def run_eval(tag, epoch):
        metrics, per_row = evaluate(
            model,
            env,
            td_hold,
            args.device,
            augmentation,
            args.eval_batch_size,
            problem_ids=hold_ids,
        )
        log_metrics(f"[{tag} epoch {epoch} holdout family{cluster} {name}]", metrics)
        record = {"tag": tag, "epoch": epoch, "cluster": cluster, "family": name, **metrics}
        history.append(record)
        save_json(os.path.join(fam_dir, "metrics.json"), history)
        write_csv(os.path.join(fam_dir, f"holdout_{tag}_e{epoch}.csv"), per_row)
        return metrics

    if not args.skip_zero_shot:
        run_eval("zero_shot", 0)

    if args.eval_only:
        log(f"eval_only: family {cluster} done")
        return history[-1] if history else {}

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch > 0 else max(args.epochs - 2, 1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[decay_epoch], gamma=args.lr_gamma
    )
    log(
        f"optim lr={args.lr} wd={args.weight_decay} "
        f"decay_epoch={decay_epoch} gamma={args.lr_gamma} "
        f"grad_clip={args.grad_clip} loss={args.loss} po_alpha={args.po_alpha}"
    )

    last = None
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model,
            env,
            optimizer,
            td_train,
            args.device,
            args,
            epoch,
            shared["clip_grad_norms"],
        )
        scheduler.step()
        last = run_eval("finetune", epoch)
        ckpt_out = os.path.join(fam_dir, f"tuned-family{cluster}-{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "cluster": cluster,
                "family": name,
                "n_size": args.n_size,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "holdout_metrics": last,
            },
            ckpt_out,
        )
        log(f"saved {ckpt_out}")

    log(f"Fine-tuning complete for family {cluster} {name}")
    log(f"Artifacts in {fam_dir}")
    return last or {}


def main():
    args = parse_args()
    clusters = parse_clusters(args.clusters)
    if args.path_id is None:
        args.path_id = DEFAULT_CKPT_DIR[args.n_size]
    if args.batch_size is None:
        args.batch_size = 128 if args.n_size == 50 else 64

    size_dir = add_cada_to_path(args.n_size)
    from envs.env import MTVRPEnv
    from envs.fill_missing_fields import fill_missing_vrp_fields
    from envs.transformer import StateAugmentation
    from model import VRPModel
    from utils.functions import clip_grad_norms, load_npz_to_tensordict

    cfg_path = size_dir / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    args.env["generator_params"]["num_loc"] = args.n_size
    args.env["data_dir"] = str(resolve_data_dir(args.data_dir))
    args.data_dir = args.env["data_dir"]
    args.env["test_size"] = [args.n_size]
    args.env["test_problem"] = list(ALL_TEST_PROBLEMS)
    args.env["test_distribution"] = ["uniform"]
    args.model_params = dict(args.model_params)
    args.model_params.setdefault("p_num", 5 if args.n_size == 50 else 1)
    args.model_params["sqrt_embedding_dim"] = args.model_params["embedding_dim"] ** 0.5
    args.log = log
    args.mute = False
    args.ddp = False
    args.rank = 0

    if torch.cuda.is_available():
        args.device = "cuda"
        torch.cuda.set_device(0)
        log(f"Using device: cuda ({torch.cuda.get_device_name()})")
    else:
        args.device = "cpu"
        log("WARNING: CUDA not available, running on CPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    csv_path = resolve_cluster_csv(args.n_size, args.cluster_csv)
    args.cluster_csv = str(csv_path) if csv_path else None

    args.ckpt_path = str(
        size_dir / "result" / args.path_id / f"checkpoint-{args.epoch}.pt"
    )
    if not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt_path}")
    with open(args.ckpt_path, "rb") as f:
        magic = f.read(32)
    if magic.startswith(b"version https://git-lfs.github.com"):
        raise RuntimeError(
            f"{args.ckpt_path} is a Git LFS pointer. Run git lfs pull on the cluster."
        )

    stamp = time.strftime("%Y-%m%d-%H%M", time.localtime())
    if args.result_root:
        args.result_root = os.path.abspath(os.path.expanduser(args.result_root))
    else:
        args.result_root = str(size_dir / "result" / f"family-ft-n{args.n_size}-{stamp}")
    os.makedirs(args.result_root, exist_ok=True)
    log_path = os.path.join(args.result_root, "run.log")

    class _Tee:
        def __init__(self, stream, path):
            self.stream = stream
            self.fh = open(path, "a", encoding="utf-8")

        def write(self, data):
            self.stream.write(data)
            self.fh.write(data)
            self.fh.flush()
            return len(data)

        def flush(self):
            self.stream.flush()
            self.fh.flush()

    sys.stdout = _Tee(sys.stdout, log_path)
    sys.stderr = _Tee(sys.stderr, log_path)

    log("=" * 78)
    log("CADA family fine-tune (6 specialists)")
    log(f"n_size      : {args.n_size}")
    log(f"clusters    : {clusters} -> {[FAMILY_NAME[c] for c in clusters]}")
    log(f"csv         : {args.cluster_csv or '(type mapping, no csv)'}")
    log(f"checkpoint  : {args.ckpt_path}")
    log(f"data_dir    : {args.data_dir}")
    log(f"result_root : {args.result_root}")
    log(f"run.log     : {log_path}")
    log(
        f"epochs={args.epochs} lr={args.lr} batch={args.batch_size} "
        f"holdout={args.holdout_frac} loss={args.loss} seed={args.seed}"
    )
    log("=" * 78)

    shared = {
        "env_cls": MTVRPEnv,
        "model_cls": VRPModel,
        "aug_cls": StateAugmentation,
        "fill_missing_vrp_fields": fill_missing_vrp_fields,
        "load_npz_to_tensordict": load_npz_to_tensordict,
        "clip_grad_norms": clip_grad_norms,
    }

    summary = []
    for cluster in clusters:
        last = run_one_family(cluster, args, shared)
        summary.append(
            {
                "cluster": cluster,
                "family": FAMILY_NAME[cluster],
                **{k: last.get(k) for k in ("n", "no_aug_cost", "aug_cost", "no_aug_gap", "aug_gap")},
            }
        )
        if args.device == "cuda":
            torch.cuda.empty_cache()

    log("")
    log("=" * 78)
    log("SUMMARY (last holdout eval per family)")
    log("=" * 78)
    for row in summary:
        gap = row.get("aug_gap")
        gap_s = f"{gap:.3f}%" if isinstance(gap, float) else "NA"
        log(
            f"  family{row['cluster']} {row['family']:22s} "
            f"n={row.get('n')} aug_cost={row.get('aug_cost')} aug_gap={gap_s}"
        )
    save_json(os.path.join(args.result_root, "summary.json"), summary)
    write_csv(os.path.join(args.result_root, "summary.csv"), summary)
    log(f"Wrote {args.result_root}/summary.json and summary.csv")
    log("All families done")


if __name__ == "__main__":
    main()
