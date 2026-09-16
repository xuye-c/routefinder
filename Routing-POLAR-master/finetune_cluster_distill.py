#!/usr/bin/env python3
"""Cluster adaptation by imitating a frozen POLAR teacher (no on-policy PO).

Does not modify finetune_cluster.py. Shared data/eval helpers are imported from it.

Targets (in order of --target):
  teacher  frozen checkpoint-300 greedy/POMO tours (default)
  pyvrp    integer tours from *_sol_pyvrp.npz if present; else fall back to teacher

Student: freeze encoder + PromptNet by default; train decoder with NLL on teacher
tours, plus optional L2-SP pull toward the pretrained decoder weights.
"""

from __future__ import annotations

import argparse
import copy
import os
import time

import numpy as np
import torch

import finetune_cluster as fc
from envs.mtvrp.env import MTVRPEnv
from envs.transformer import StateAugmentation
from models.model import VRPModel
from utils.functions import clip_grad_norms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster distill POLAR from frozen teacher"
    )
    parser.add_argument("--n_size", type=int, required=True, choices=[50, 100])
    parser.add_argument("--cluster", type=int, required=True)
    parser.add_argument("--cluster_csv", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--path_id", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--lr_decay_epoch", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--target",
        type=str,
        default="teacher",
        choices=["teacher", "pyvrp"],
        help="Whose tours to imitate",
    )
    parser.add_argument(
        "--teacher_pomo",
        type=int,
        default=8,
        help="POMO starts used by the frozen teacher to pick a target tour. 0 = all customers",
    )
    parser.add_argument(
        "--l2sp",
        type=float,
        default=1e-3,
        help="Coefficient of ||theta - theta_pretrained||^2 on trainable params",
    )
    parser.add_argument(
        "--freeze_encoder",
        dest="freeze_encoder",
        action="store_true",
        default=True,
        help="Freeze encoder + PromptNet (default)",
    )
    parser.add_argument(
        "--no_freeze_encoder",
        dest="freeze_encoder",
        action="store_false",
        help="Train the full student",
    )
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_zero_shot", action="store_true")
    return parser.parse_args()


def load_checkpoint_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"]
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return ckpt


def freeze_encoder(model):
    frozen, trained = 0, 0
    for name, param in model.named_parameters():
        if name.startswith("decoder."):
            param.requires_grad = True
            trained += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Freeze encoder/prompt: frozen={frozen} trainable_decoder={trained}")


def snapshot_trainable(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def l2sp_penalty(model, anchor):
    loss = None
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in anchor:
            continue
        term = (param - anchor[name]).pow(2).sum()
        loss = term if loss is None else loss + term
    if loss is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return loss


def pick_best_tours(out, batch_size):
    reward = out["reward"].view(-1, batch_size)
    seq = out["tours"].size(-1)
    tours = out["tours"].view(-1, batch_size, seq)
    best = reward.argmax(dim=0)
    bidx = torch.arange(batch_size, device=best.device)
    return tours[best, bidx].contiguous()


def try_load_pyvrp_tours(data_dir, n_size, rows):
    by_type = {}
    for i, row in enumerate(rows):
        by_type.setdefault(row["type"], []).append((row["index"], i))

    ordered = [None] * len(rows)
    used_key = None
    for ptype, items in by_type.items():
        path = os.path.join(data_dir, ptype, "test", f"{n_size}_sol_pyvrp.npz")
        if not os.path.isfile(path):
            print(f"No sol file for {ptype}: {path}")
            return None
        sol = np.load(path)
        keys = list(sol.files)
        tour_key = None
        for cand in ("tours", "actions", "routes", "solutions"):
            if cand in sol:
                tour_key = cand
                break
        if tour_key is None:
            print(f"sol keys for {ptype} have no tours: {keys}")
            return None
        used_key = tour_key
        arr = sol[tour_key]
        if arr.ndim != 2:
            print(f"{path}['{tour_key}'] has shape {arr.shape}, expected 2D")
            return None
        for local_idx, row_i in items:
            if local_idx >= arr.shape[0]:
                print(f"{ptype} tour index {local_idx} out of range")
                return None
            ordered[row_i] = torch.as_tensor(arr[local_idx], dtype=torch.long)

    max_len = max(t.numel() for t in ordered)
    padded = torch.zeros(len(ordered), max_len, dtype=torch.long)
    for i, t in enumerate(ordered):
        padded[i, : t.numel()] = t
    print(f"Loaded PyVRP tours from key='{used_key}', shape={tuple(padded.shape)}")
    return padded


def configure_teacher(teacher, teacher_pomo):
    teacher.args = copy.copy(teacher.args)
    teacher.args.trainer_params = dict(teacher.args.trainer_params)
    teacher.args.trainer_params["po_B"] = None if teacher_pomo <= 0 else int(teacher_pomo)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False


@torch.no_grad()
def teacher_tours_for_batch(teacher, env, td, args):
    td_t = env.reset(td=td.clone(recurse=True).to(args.device))
    with torch.amp.autocast(
        device_type=args.device,
        dtype=args.amp_dtype,
        enabled=(args.device == "cuda"),
    ):
        out = teacher(td_t, env)
    # inference/no_grad tensors cannot be used as indices in autograd
    return pick_best_tours(out, td.batch_size[0]).detach().clone()


def nll_on_tours(student, env, td, tours, args):
    td_s = env.reset(td=td.clone(recurse=True).to(args.device))
    tours = tours.detach().to(device=args.device, dtype=torch.long).clone()
    lengths = torch.full(
        (tours.size(0),), tours.size(1), dtype=torch.long, device=args.device
    )
    with torch.amp.autocast(
        device_type=args.device,
        dtype=args.amp_dtype,
        enabled=(args.device == "cuda"),
    ):
        out = student.route_forward(td_s, env, tours, lengths, num_starts=1)
        nll = -out["log_likelihood"].sum(dim=1).mean()
        cost = (-out["reward"]).mean()
    return nll, cost


def train_one_epoch(
    student,
    teacher,
    env,
    optimizer,
    scaler,
    td_train,
    pyvrp_tours,
    train_indices,
    args,
    epoch,
    use_scaler,
    anchor,
):
    student.train()
    if args.freeze_encoder:
        student.encoder.eval()
        if getattr(student, "prompt_net", None) is not None:
            student.prompt_net.eval()
    if teacher is not None:
        teacher.eval()

    n = td_train.batch_size[0]
    perm = torch.randperm(n, device="cpu")
    losses, nlls, costs = [], [], []
    instances_seen = 0

    while instances_seen < n:
        end = min(instances_seen + args.batch_size, n)
        local = perm[instances_seen:end]
        batch = fc.subset_td(td_train, local.cpu())
        if pyvrp_tours is not None:
            src_idx = torch.as_tensor(train_indices, dtype=torch.long)[local.cpu()]
            tours = pyvrp_tours[src_idx]
        else:
            tours = teacher_tours_for_batch(teacher, env, batch, args)

        optimizer.zero_grad(set_to_none=True)
        nll, cost = nll_on_tours(student, env, batch, tours, args)
        penalty = l2sp_penalty(student, anchor)
        loss = nll + args.l2sp * penalty

        (scaler.scale(loss) if use_scaler else loss).backward()
        clip_grad_norms(optimizer.param_groups, args.grad_clip)
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        losses.append(float(loss.item()))
        nlls.append(float(nll.item()))
        costs.append(float(cost.item()))
        instances_seen = end

    print(
        f"Epoch {epoch:03d} distill loss={np.mean(losses):.4f} "
        f"nll={np.mean(nlls):.4f} imitate_cost={np.mean(costs):.4f} "
        f"l2sp={args.l2sp:g} seen={n}"
    )
    return float(np.mean(losses))


def main():
    args = parse_args()
    if args.path_id is None:
        args.path_id = fc.default_path_id(args.n_size)
    if args.batch_size is None:
        args.batch_size = 32 if args.n_size == 50 else 16

    fc.load_yaml_config(args)
    args.trainer_params["po_B"] = None
    fc.setup_device(args)
    fc.seed_all(args.seed, args.device)
    args.log = print

    if args.device == "cuda":
        cap = torch.cuda.get_device_capability()
        args.amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        use_scaler = cap[0] < 8
    else:
        args.amp_dtype = torch.float16
        use_scaler = False

    csv_path = fc.resolve_cluster_csv(args.n_size, args.cluster_csv)
    ckpt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "result",
        args.path_id,
        f"checkpoint-{args.epoch}.pt",
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        if f.read(32).startswith(b"version https://git-lfs.github.com"):
            raise RuntimeError(f"{ckpt_path} is a Git LFS pointer")

    result_dir = os.path.join(
        "result",
        f"cluster-distill-n{args.n_size}-c{args.cluster}-"
        + time.strftime("%Y-%m%d-%H%M", time.localtime()),
    )
    os.makedirs(result_dir, exist_ok=True)
    print("=" * 70)
    print("POLAR cluster distill (frozen teacher, no on-policy PO)")
    print(f"csv        : {csv_path}")
    print(f"checkpoint : {ckpt_path}")
    print(f"cluster    : {args.cluster}")
    print(f"target     : {args.target}")
    print(f"freeze_enc : {args.freeze_encoder}")
    print(f"l2sp       : {args.l2sp}")
    print(f"teacher_pomo: {args.teacher_pomo}")
    print(f"result_dir : {result_dir}")
    print("=" * 70)

    rows = fc.load_cluster_rows(csv_path, args.n_size, args.cluster)
    train_idx, hold_idx = fc.stratified_split(rows, args.holdout_frac, args.seed)
    print(
        f"cluster size={len(rows)} types={sorted({r['type'] for r in rows})} "
        f"train={len(train_idx)} holdout={len(hold_idx)}"
    )

    env = MTVRPEnv(**args.env)
    env.set_loss_mode("po")
    td_all = fc.load_cluster_tensordict(env, rows, args.n_size)
    td_train = fc.subset_td(td_all, train_idx)
    td_hold = fc.subset_td(td_all, hold_idx)
    fc.save_json(
        os.path.join(result_dir, "split.json"),
        {
            "n_size": args.n_size,
            "cluster": args.cluster,
            "mode": "distill",
            "target": args.target,
            "csv": csv_path,
            "holdout_frac": args.holdout_frac,
            "seed": args.seed,
            "types": sorted({r["type"] for r in rows}),
            "train_ids": [rows[i]["problem_id"] for i in train_idx],
            "holdout_ids": [rows[i]["problem_id"] for i in hold_idx],
        },
    )

    student = VRPModel(args).to(args.device)
    student.set_loss_mode("po")
    load_checkpoint_weights(student, ckpt_path, args.device)
    print(f"Loaded student from {ckpt_path}")

    teacher = None
    pyvrp_tours = None
    if args.target == "pyvrp":
        pyvrp_tours = try_load_pyvrp_tours(args.data_dir, args.n_size, rows)
        if pyvrp_tours is None:
            print("PyVRP tours unavailable; falling back to frozen teacher")
            args.target = "teacher"

    if args.target == "teacher":
        teacher = VRPModel(args).to(args.device)
        teacher.set_loss_mode("po")
        load_checkpoint_weights(teacher, ckpt_path, args.device)
        configure_teacher(teacher, args.teacher_pomo)
        print(
            f"Frozen teacher ready (po_B={teacher.args.trainer_params.get('po_B')})"
        )

    if args.freeze_encoder:
        freeze_encoder(student)
    else:
        print("Training full student (encoder not frozen)")

    trainable = [p for p in student.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters")
    anchor = snapshot_trainable(student)

    augmentation = StateAugmentation()
    history = []

    def run_eval(tag, epoch):
        metrics = fc.evaluate(
            student, env, td_hold, args, augmentation, args.eval_batch_size
        )
        fc.log_metrics(f"[{tag} epoch {epoch} holdout]", metrics)
        record = {
            "tag": tag,
            "epoch": epoch,
            "mode": "distill",
            "target": args.target,
            **metrics,
        }
        history.append(record)
        fc.save_json(os.path.join(result_dir, "metrics.json"), history)
        return metrics

    if not args.skip_zero_shot:
        run_eval("zero_shot", 0)

    if args.eval_only:
        print("eval_only: done")
        return

    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    decay_epoch = (
        args.lr_decay_epoch if args.lr_decay_epoch > 0 else max(args.epochs - 2, 1)
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[decay_epoch], gamma=args.lr_gamma
    )
    print(
        f"optim lr={args.lr} wd={args.weight_decay} decay_epoch={decay_epoch} "
        f"gamma={args.lr_gamma} grad_clip={args.grad_clip} l2sp={args.l2sp} "
        f"freeze_encoder={args.freeze_encoder} target={args.target}"
    )

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            student,
            teacher,
            env,
            optimizer,
            scaler,
            td_train,
            pyvrp_tours,
            train_idx,
            args,
            epoch,
            use_scaler,
            anchor,
        )
        scheduler.step()
        metrics = run_eval("distill", epoch)
        torch.save(
            {
                "epoch": epoch,
                "cluster": args.cluster,
                "n_size": args.n_size,
                "mode": "distill",
                "target": args.target,
                "model_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "holdout_metrics": metrics,
            },
            os.path.join(result_dir, f"distill-cluster{args.cluster}-{epoch}.pt"),
        )

    print("Distill complete")
    print(f"Metrics saved to {result_dir}/metrics.json")


if __name__ == "__main__":
    main()
