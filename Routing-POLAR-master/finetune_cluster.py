#!/usr/bin/env python3
"""Fine-tune a POLAR checkpoint on one UMAP cluster of the labeled test set.

Scheme B: instances come from data/{type}/test/{n}.npz, indexed by
umap_cluster_polar_encf_{n}.csv. Each cluster is split into train/holdout.
Training does not use PyVRP local search. Gaps use *_sol_pyvrp.npz costs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tensordict import TensorDict

from envs.mtvrp.env import MTVRPEnv
from envs.transformer import StateAugmentation
from models.model import VRPModel
from utils.functions import clip_grad_norms, get_torch_device
from utils.metrics import gap_percent_mean_torch


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

KEEP_KEYS = (
    "locs",
    "demand_linehaul",
    "demand_backhaul",
    "backhaul_class",
    "distance_limit",
    "service_time",
    "open_route",
    "time_windows",
    "vehicle_capacity",
    "capacity_original",
    "speed",
    "opt_cost",
    "p_s_tag",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster fine-tune POLAR (scheme B)")
    parser.add_argument("--n_size", type=int, required=True, choices=[50, 100])
    parser.add_argument("--cluster", type=int, required=True)
    parser.add_argument(
        "--cluster_csv",
        type=str,
        default=None,
        help="CSV with columns problem_id,type,cluster. "
        "Default: umap_cluster_polar_encf_{n_size}.csv in CWD or parent dirs.",
    )
    parser.add_argument("--epoch", type=int, default=300, help="Pretrained checkpoint epoch")
    parser.add_argument(
        "--path_id",
        type=str,
        default=None,
        help='Checkpoint folder under result/, e.g. "n=50/2026-0728-0719"',
    )
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.1,
        help="MultiStepLR decay factor",
    )
    parser.add_argument(
        "--lr_decay_epoch",
        type=int,
        default=0,
        help="Epoch to decay LR (1-based). 0 => epochs-2",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--loss", type=str, default="po", choices=["po", "rl"])
    parser.add_argument("--po_alpha", type=float, default=0.05)
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluate the pretrained checkpoint on the holdout split and exit",
    )
    parser.add_argument(
        "--skip_zero_shot",
        action="store_true",
        help="Do not evaluate before fine-tuning",
    )
    return parser.parse_args()


def default_path_id(n_size):
    if n_size == 50:
        return "n=50/2026-0728-0719"
    return "n=100/2026-0729-1221"


def default_batch_size(n_size):
    return 128 if n_size == 50 else 64


def resolve_cluster_csv(n_size, explicit):
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"cluster csv not found: {path}")
        return path

    name = f"umap_cluster_polar_encf_{n_size}.csv"
    candidates = [
        os.path.abspath(name),
        os.path.abspath(os.path.join(os.path.dirname(__file__), name)),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", name)),
        os.path.abspath(os.path.join(os.path.expanduser("~"), "routefinder", name)),
        os.path.abspath(os.path.join(os.path.expanduser("~"), "wfl_sw", name)),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Could not find {name}. Pass --cluster_csv. Looked in:\n  "
        + "\n  ".join(candidates)
    )


def load_yaml_config(args):
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg.items():
        setattr(args, key, value)

    args.env["generator_params"]["num_loc"] = args.n_size
    args.env["data_dir"] = args.data_dir
    args.env["test_size"] = [args.n_size]
    args.env["test_problem"] = list(ALL_TEST_PROBLEMS)
    args.env["test_distribution"] = ["uniform"]
    args.model_params["sqrt_embedding_dim"] = args.model_params["embedding_dim"] ** 0.5
    args.trainer_params["po_B"] = None
    args.trainer_params["use_ls"] = False
    args.mute = False
    args.ddp = False
    args.rank = 0
    args.wandb = ""
    args.skip = False


def setup_device(args):
    args.device = str(get_torch_device())
    torch.set_default_device(torch.device(args.device))
    if args.device == "cpu":
        print("WARNING: CUDA not available, running on CPU")
    else:
        torch.cuda.set_device(0)
        print(f"Using device: {args.device} ({torch.cuda.get_device_name()})")


def seed_all(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def parse_problem_id(problem_id, n_size):
    prefix = f"_{n_size}_"
    if prefix not in problem_id:
        raise ValueError(f"problem_id {problem_id} does not match n_size={n_size}")
    ptype, idx_s = problem_id.rsplit(prefix, 1)
    return ptype, int(idx_s)


def load_cluster_rows(csv_path, n_size, cluster):
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
    if "backhaul_class" not in out:
        out["backhaul_class"] = torch.ones(
            (td.batch_size[0], 1), dtype=torch.float32, device=td.device
        )
    return TensorDict(out, batch_size=td.batch_size)


def load_cluster_tensordict(env, rows, n_size):
    datasets = env.dataset(phase="test")
    by_type = defaultdict(list)
    for i, row in enumerate(rows):
        by_type[row["type"]].append((row["index"], i))

    ordered = [None] * len(rows)
    for ptype, items in by_type.items():
        name = f"{n_size}_{ptype}_uniform"
        if name not in datasets:
            raise KeyError(
                f"Missing dataset '{name}'. Available: {sorted(datasets.keys())}"
            )
        td = select_keys(datasets[name])
        n_inst = td.batch_size[0]
        for local_idx, row_i in items:
            if local_idx < 0 or local_idx >= n_inst:
                raise IndexError(
                    f"{ptype} index {local_idx} out of range (n={n_inst})"
                )
            ordered[row_i] = subset_td(td, [local_idx])

    stacked = stack_tds(ordered)
    if "opt_cost" not in stacked.keys():
        raise KeyError(
            "opt_cost missing after load. Confirm *_sol_pyvrp.npz exists next to test npz."
        )
    if torch.all(stacked["opt_cost"] == 0):
        print("WARNING: all opt_cost values are 0; gaps will be NaN")
    return stacked


def stack_tds(pieces):
    keys = list(pieces[0].keys())
    for td in pieces[1:]:
        keys = [k for k in keys if k in td.keys()]
    batch = sum(td.batch_size[0] for td in pieces)
    return TensorDict(
        {k: torch.cat([td[k] for td in pieces], dim=0) for k in keys},
        batch_size=[batch],
    )


def subset_td(td, indices):
    # torch.set_default_device(cuda) would otherwise put indices on GPU
    # while loaded npz TensorDicts stay on CPU.
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


def compute_po_loss(reward, log_likelihood, po_alpha):
    preference = reward[:, :, None] > reward[:, None, :]
    log_prob = po_alpha * log_likelihood
    log_prob_pair = log_prob[:, :, None] - log_prob[:, None, :]
    return -torch.mean(F.logsigmoid(log_prob_pair) * preference)


@torch.inference_mode()
def evaluate(model, env, td_all, args, augmentation, batch_size):
    model.eval()
    n = td_all.batch_size[0]
    costs, aug_costs, opts = [], [], []

    use_amp = args.device == "cuda"
    amp_dtype = args.amp_dtype

    for start in range(0, n, batch_size):
        inp = subset_td(td_all, list(range(start, min(start + batch_size, n))))
        opt_cost = inp["opt_cost"].clone()
        p_s_tag = inp["p_s_tag"].clone() if "p_s_tag" in inp.keys() else None
        td = env.reset(td=inp.to(args.device))
        if p_s_tag is not None:
            td["p_s_tag"] = p_s_tag.to(args.device)

        with torch.amp.autocast(
            device_type=args.device, dtype=amp_dtype, enabled=use_amp
        ):
            td_aug = augmentation(td)
            out = model(td_aug, env)
            reward = out["reward"].view(-1, augmentation.num_augment, inp.batch_size[0])
            all_reward, _ = reward.max(dim=0)
            score = -all_reward[0, :].float()
            aug_score = -all_reward.max(dim=0).values.float()

        costs.append(score.detach().cpu())
        aug_costs.append(aug_score.detach().cpu())
        opts.append(opt_cost.cpu())

        del td, td_aug, out, reward
        if hasattr(model, "encoded_nodes"):
            model.encoded_nodes = None

    cost = torch.cat(costs)
    aug_cost = torch.cat(aug_costs)
    opt = torch.cat(opts)
    gap = gap_percent_mean_torch(cost.abs(), opt.abs())
    aug_gap = gap_percent_mean_torch(aug_cost.abs(), opt.abs())
    return {
        "n": int(n),
        "no_aug_cost": float(cost.mean().item()),
        "aug_cost": float(aug_cost.mean().item()),
        "no_aug_gap": gap,
        "aug_gap": aug_gap,
    }


def log_metrics(prefix, metrics):
    print(
        f"{prefix} n={metrics['n']} "
        f"cost={metrics['no_aug_cost']:.4f} gap={metrics['no_aug_gap']:.3f}% | "
        f"aug_cost={metrics['aug_cost']:.4f} aug_gap={metrics['aug_gap']:.3f}%"
    )


def save_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def train_one_epoch(model, env, optimizer, scaler, td_train, args, epoch, use_scaler):
    model.train()
    n = td_train.batch_size[0]
    perm = torch.randperm(n, device="cpu")
    losses = []
    costs = []
    instances_seen = 0

    while instances_seen < n:
        end = min(instances_seen + args.batch_size, n)
        idx = perm[instances_seen:end]
        batch = subset_td(td_train, idx.cpu())
        td = env.reset(td=batch.to(args.device))
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=args.device,
            dtype=args.amp_dtype,
            enabled=(args.device == "cuda"),
        ):
            out = model(td, env, with_greedy=False)
            batch_n = td.batch_size[0]
            reward = out["reward"].view(-1, batch_n)
            log_likelihood = out["log_likelihood"].sum(1).view(-1, batch_n)
            if args.loss == "po":
                loss = compute_po_loss(reward, log_likelihood, args.po_alpha)
            else:
                advantage = reward - reward.mean(dim=0, keepdims=True)
                loss = -(advantage * log_likelihood).mean()
            score_mean = (-reward).max(dim=0).values.mean()

        (scaler.scale(loss) if use_scaler else loss).backward()
        clip_grad_norms(optimizer.param_groups, args.grad_clip)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        losses.append(float(loss.item()))
        costs.append(float(score_mean.item()))
        instances_seen = end
        del td, out, reward, log_likelihood, loss

    print(
        f"Epoch {epoch:03d} train loss={np.mean(losses):.4f} "
        f"best_cost={np.mean(costs):.4f} seen={n}"
    )
    return float(np.mean(losses)), float(np.mean(costs))


def main():
    args = parse_args()
    if args.path_id is None:
        args.path_id = default_path_id(args.n_size)
    if args.batch_size is None:
        args.batch_size = default_batch_size(args.n_size)

    load_yaml_config(args)
    setup_device(args)
    seed_all(args.seed, args.device)
    args.log = print

    if args.device == "cuda":
        cap = torch.cuda.get_device_capability()
        args.amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        use_scaler = cap[0] < 8
    else:
        args.amp_dtype = torch.float16
        use_scaler = False

    csv_path = resolve_cluster_csv(args.n_size, args.cluster_csv)
    ckpt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "result",
        args.path_id,
        f"checkpoint-{args.epoch}.pt",
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        magic = f.read(32)
    if magic.startswith(b"version https://git-lfs.github.com"):
        raise RuntimeError(
            f"{ckpt_path} is a Git LFS pointer, not a real checkpoint. "
            "Run this on the cluster after git lfs pull."
        )

    result_dir = os.path.join(
        "result",
        f"cluster-ft-n{args.n_size}-c{args.cluster}-"
        + time.strftime("%Y-%m%d-%H%M", time.localtime()),
    )
    os.makedirs(result_dir, exist_ok=True)
    print("=" * 70)
    print("POLAR cluster fine-tune (scheme B, no PyVRP)")
    print(f"csv        : {csv_path}")
    print(f"checkpoint : {ckpt_path}")
    print(f"cluster    : {args.cluster}")
    print(f"n_size     : {args.n_size}")
    print(f"result_dir : {result_dir}")
    print("=" * 70)

    rows = load_cluster_rows(csv_path, args.n_size, args.cluster)
    train_idx, hold_idx = stratified_split(rows, args.holdout_frac, args.seed)
    print(
        f"cluster size={len(rows)} types={sorted({r['type'] for r in rows})} "
        f"train={len(train_idx)} holdout={len(hold_idx)}"
    )

    env = MTVRPEnv(**args.env)
    env.set_loss_mode(args.loss)
    td_all = load_cluster_tensordict(env, rows, args.n_size)
    td_train = subset_td(td_all, train_idx)
    td_hold = subset_td(td_all, hold_idx)

    split_payload = {
        "n_size": args.n_size,
        "cluster": args.cluster,
        "csv": csv_path,
        "holdout_frac": args.holdout_frac,
        "seed": args.seed,
        "types": sorted({r["type"] for r in rows}),
        "train_ids": [rows[i]["problem_id"] for i in train_idx],
        "holdout_ids": [rows[i]["problem_id"] for i in hold_idx],
    }
    save_json(os.path.join(result_dir, "split.json"), split_payload)

    model = VRPModel(args).to(args.device)
    model.set_loss_mode(args.loss)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"Loaded pretrained weights from {ckpt_path}")
    del ckpt

    augmentation = StateAugmentation()
    history = []

    def run_eval(tag, epoch):
        metrics = evaluate(
            model, env, td_hold, args, augmentation, args.eval_batch_size
        )
        log_metrics(f"[{tag} epoch {epoch} holdout]", metrics)
        record = {"tag": tag, "epoch": epoch, **metrics}
        history.append(record)
        save_json(os.path.join(result_dir, "metrics.json"), history)
        return metrics

    if not args.skip_zero_shot:
        run_eval("zero_shot", 0)

    if args.eval_only:
        print("eval_only: done")
        return

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch > 0 else max(args.epochs - 2, 1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[decay_epoch], gamma=args.lr_gamma
    )
    print(
        f"optim lr={args.lr} wd={args.weight_decay} "
        f"decay_epoch={decay_epoch} gamma={args.lr_gamma} "
        f"grad_clip={args.grad_clip} loss={args.loss} po_alpha={args.po_alpha}"
    )

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model, env, optimizer, scaler, td_train, args, epoch, use_scaler
        )
        scheduler.step()
        metrics = run_eval("finetune", epoch)
        torch.save(
            {
                "epoch": epoch,
                "cluster": args.cluster,
                "n_size": args.n_size,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "holdout_metrics": metrics,
            },
            os.path.join(result_dir, f"tuned-cluster{args.cluster}-{epoch}.pt"),
        )

    print("Fine-tuning complete")
    print(f"Metrics saved to {result_dir}/metrics.json")


if __name__ == "__main__":
    main()
