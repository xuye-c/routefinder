#!/usr/bin/env python3
"""Distill 6 family-CADA teachers (epoch 1) into one pretrained CADA student.

Teachers: frozen tuned-family{k}-1.pt from a family-ft run (default the
2026-0921-1150 1e-5 run that saved epoch-1 checkpoints).
Student: original CADA checkpoint-300. Freeze encoder + PromptNet; train
decoder with NLL on teacher greedy tours (best of --teacher_pomo starts).

Default: 1 distill epoch, 10-fold OOF so every test instance is scored once.

Run from routefinder/:

    python -u distill_cada_family.py --n_size 50 --epochs 1 --folds 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import finetune_cada_family as fc

FAMILY_NAME = fc.FAMILY_NAME


def parse_args():
    p = argparse.ArgumentParser(description="Distill family-CADA e1 into one CADA")
    p.add_argument("--n_size", type=int, required=True, choices=[50, 100])
    p.add_argument("--clusters", type=str, default="0,1,2,3,4,5")
    p.add_argument("--cluster_csv", type=str, default=None)
    p.add_argument("--epoch", type=int, default=300, help="Student pretrained epoch")
    p.add_argument("--path_id", type=str, default=None)
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--teacher_root", type=str, default=None)
    p.add_argument("--teacher_epoch", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--holdout_frac", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--l2sp", type=float, default=1e-3)
    p.add_argument("--teacher_pomo", type=int, default=8)
    p.add_argument("--no_freeze_encoder", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--skip_zero_shot", action="store_true")
    p.add_argument("--result_root", type=str, default=None)
    return p.parse_args()


def resolve_teacher_root(n_size, explicit):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"teacher_root not found: {path}")
        return path
    root = fc.routefinder_root()
    named = root / "CADA" / str(n_size) / "result" / "family-ft-n50-2026-0921-1150"
    if n_size == 50 and named.is_dir():
        return named
    result = root / "CADA" / str(n_size) / "result"
    hits = sorted(result.glob("family-ft-n*")) if result.is_dir() else []
    for cand in reversed(hits):
        probe = cand / "family0-open_TW" / "tuned-family0-1.pt"
        if probe.is_file():
            return cand
    raise FileNotFoundError(
        "No family-CADA epoch-1 checkpoints. Pass --teacher_root. "
        "Expected e.g. CADA/50/result/family-ft-n50-2026-0921-1150"
    )


def teacher_ckpt(teacher_root: Path, cluster: int, teacher_epoch: int) -> Path:
    name = FAMILY_NAME[cluster]
    fam = teacher_root / f"family{cluster}-{name}"
    candidates = [
        fam / f"tuned-family{cluster}-{teacher_epoch}.pt",
        fam / f"tuned-family{cluster}-fold0-e{teacher_epoch}.pt",
    ]
    if fam.is_dir():
        candidates.extend(
            sorted(fam.glob(f"tuned-family{cluster}-*-e{teacher_epoch}.pt"))
        )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"missing teacher epoch {teacher_epoch} under {fam} "
        f"(need tuned-family{cluster}-{teacher_epoch}.pt from the 80/20 1150 run)"
    )


def freeze_encoder_prompt(model, quiet=False):
    frozen, trained = 0, 0
    for name, param in model.named_parameters():
        if name.startswith("decoder."):
            param.requires_grad = True
            trained += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    if not quiet:
        fc.log(f"Freeze encoder/prompt: frozen={frozen} trainable_decoder={trained}")


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


def _cache_from_embed(model, node_embed):
    from model import PrecomputedCache, reshape_by_heads

    heads = model.args.model_params["head_num"]
    decoder_k = reshape_by_heads(model.decoder.Wk(node_embed), head_num=heads)
    decoder_v = reshape_by_heads(model.decoder.Wv(node_embed), head_num=heads)
    decoder_single_head_k = node_embed.transpose(1, 2)
    return PrecomputedCache(node_embed, decoder_k, decoder_v, decoder_single_head_k)


@torch.inference_mode()
def teacher_tours_batch(model, env, td_in, device, max_starts):
    from model import VRPModel
    from utils.functions import batchify, gather_by_index

    p_s_tag = td_in["p_s_tag"].clone()
    td = env.reset(td=td_in.clone(recurse=True).to(device))
    td["p_s_tag"] = p_s_tag.to(device)
    batch = td.batch_size[0]
    prompt = model.prompt_net(td)["prompt"]
    node_embed = model.encoder(td, prompt)
    num_starts, action = env.select_start_nodes(td)
    if max_starts > 0 and max_starts < num_starts:
        action = action.view(num_starts, batch)[:max_starts].reshape(-1)
        num_starts = max_starts
    td = batchify(td, num_starts)
    actions_list = [action]
    td.set("action", action)
    td = env.step(td)["next"]
    cache = _cache_from_embed(model, node_embed)
    while not td["done"].all():
        logprobs, mask = model.decoder(td, cache, num_starts)
        select = VRPModel.greedy(logprobs, mask)
        td.set("action", select)
        actions_list.append(select)
        td = env.step(td)["next"]
    actions = torch.stack(actions_list, 1)
    reward = env.get_reward(td, actions).view(num_starts, batch)
    tours = actions.view(num_starts, batch, -1)
    best = reward.argmax(dim=0)
    bidx = torch.arange(batch, device=device)
    return tours[best, bidx].contiguous().cpu()


def nll_on_tours(model, env, td_in, tours, device):
    from utils.functions import batchify, gather_by_index

    p_s_tag = td_in["p_s_tag"].clone()
    td = env.reset(td=td_in.clone(recurse=True).to(device))
    td["p_s_tag"] = p_s_tag.to(device)
    tours = tours.detach().to(device=device, dtype=torch.long)
    batch = td.batch_size[0]
    prompt = model.prompt_net(td)["prompt"]
    node_embed = model.encoder(td, prompt)
    # CADA decoder gather_by_index squeezes when the start dim is 1, then
    # torch.cat(cur_node_embedding, state_embedding) is 2D vs 3D. Keep two
    # identical starts so the start dim stays, matching POMO training.
    num_starts = 2
    tours = tours.repeat(num_starts, 1)
    td = batchify(td, num_starts)
    nll = torch.zeros(batch * num_starts, device=device)
    td.set("action", tours[:, 0])
    td = env.step(td)["next"]
    cache = _cache_from_embed(model, node_embed)
    step = 1
    while not td["done"].all() and step < tours.size(1):
        logprobs, mask = model.decoder(td, cache, num_starts)
        select = tours[:, step]
        token = gather_by_index(logprobs, select, dim=1)
        active = (~td["done"]).reshape(batch * num_starts).float()
        nll = nll - token * active
        td.set("action", select)
        td = env.step(td)["next"]
        step += 1
    reward = env.get_reward(td, tours[:, :step]).view(num_starts, batch)[0]
    nll = nll.view(num_starts, batch)[0]
    return nll.mean(), (-reward).mean()


def precompute_tours(env, td_all, rows, clusters, teacher_root, teacher_epoch, device, max_starts, batch_size, model_ctor):
    n = td_all.batch_size[0]
    pieces = [None] * n
    fc.log(f"Precomputing teacher tours n={n} pomo={max_starts} (one family at a time)")
    for c in clusters:
        idx = [i for i, r in enumerate(rows) if int(r["cluster"]) == c]
        teacher = model_ctor().to(device)
        ck = teacher_ckpt(teacher_root, c, teacher_epoch)
        fc.load_pretrained(teacher, ck, device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        fc.log(f"teacher family{c} {FAMILY_NAME[c]} n={len(idx)} <- {ck}")
        for start in range(0, len(idx), batch_size):
            end = min(start + batch_size, len(idx))
            sub_idx = idx[start:end]
            sub = fc.subset_td(td_all, sub_idx)
            tours = teacher_tours_batch(teacher, env, sub, device, max_starts)
            for j, row in zip(sub_idx, tours):
                pieces[j] = row.cpu()
            if start == 0 or end == len(idx):
                fc.log(f"  family{c} tours {end}/{len(idx)}")
        del teacher
        if device == "cuda":
            torch.cuda.empty_cache()
    if any(t is None for t in pieces):
        missing = sum(t is None for t in pieces)
        raise RuntimeError(f"teacher tours missing for {missing} instances")
    width = max(t.numel() for t in pieces)
    out = torch.zeros(n, width, dtype=torch.long)
    for i, t in enumerate(pieces):
        out[i, : t.numel()] = t
    return out


def main():
    args = parse_args()
    clusters = fc.parse_clusters(args.clusters)
    if args.path_id is None:
        args.path_id = fc.DEFAULT_CKPT_DIR[args.n_size]
    if args.batch_size is None:
        args.batch_size = 32 if args.n_size == 50 else 16

    size_dir = fc.add_cada_to_path(args.n_size)
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
    args.env["data_dir"] = str(fc.resolve_data_dir(args.data_dir))
    args.data_dir = args.env["data_dir"]
    args.env["test_size"] = [args.n_size]
    args.env["test_problem"] = list(fc.ALL_TEST_PROBLEMS)
    args.env["test_distribution"] = ["uniform"]
    args.model_params = dict(args.model_params)
    args.model_params.setdefault("p_num", 5 if args.n_size == 50 else 1)
    args.model_params["sqrt_embedding_dim"] = args.model_params["embedding_dim"] ** 0.5
    args.log = fc.log
    args.mute = False
    args.ddp = False
    args.rank = 0

    if torch.cuda.is_available():
        args.device = "cuda"
        torch.cuda.set_device(0)
        fc.log(f"Using device: cuda ({torch.cuda.get_device_name()})")
    else:
        args.device = "cpu"
        fc.log("WARNING: CUDA not available, running on CPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    csv_path = fc.resolve_cluster_csv(args.n_size, args.cluster_csv)
    args.cluster_csv = str(csv_path) if csv_path else None
    student_ckpt = str(size_dir / "result" / args.path_id / f"checkpoint-{args.epoch}.pt")
    if not os.path.isfile(student_ckpt):
        raise FileNotFoundError(student_ckpt)
    teacher_root = resolve_teacher_root(args.n_size, args.teacher_root)

    stamp = time.strftime("%Y-%m%d-%H%M", time.localtime())
    if args.result_root:
        args.result_root = os.path.abspath(os.path.expanduser(args.result_root))
    else:
        args.result_root = str(
            size_dir / "result" / f"family-distill-n{args.n_size}-e{args.teacher_epoch}-{args.folds}fold-{stamp}"
        )
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

    fc.log("=" * 78)
    fc.log("CADA family distill (teachers = family e1, student = one CADA)")
    fc.log(f"student     : {student_ckpt}")
    fc.log(f"teacher_root: {teacher_root}")
    fc.log(f"teacher_ep  : {args.teacher_epoch}")
    fc.log(f"folds={args.folds} epochs={args.epochs} lr={args.lr} l2sp={args.l2sp}")
    fc.log(f"teacher_pomo={args.teacher_pomo} freeze_encoder={not args.no_freeze_encoder}")
    fc.log(f"result_root : {args.result_root}")
    fc.log("=" * 78)

    rows = []
    for c in clusters:
        rows.extend(fc.load_cluster_rows(args.cluster_csv, args.n_size, c))
    all_ids = [r["problem_id"] for r in rows]
    fc.log(f"instances={len(rows)} families={sorted({r['cluster'] for r in rows})}")

    env = MTVRPEnv(**args.env)
    td_all = fc.load_family_tensordict(
        rows,
        args.n_size,
        Path(args.data_dir),
        fill_missing_vrp_fields,
        load_npz_to_tensordict,
    )

    tours = precompute_tours(
        env,
        td_all,
        rows,
        clusters,
        teacher_root,
        args.teacher_epoch,
        args.device,
        args.teacher_pomo,
        args.batch_size,
        lambda: VRPModel(args),
    )
    torch.save({"tours": tours, "problem_id": all_ids}, os.path.join(args.result_root, "teacher_tours.pt"))
    fc.log(f"saved teacher tours {tuple(tours.shape)}")

    augmentation = StateAugmentation()
    splits = (
        fc.stratified_kfold(rows, args.folds, args.seed)
        if args.folds > 1
        else [fc.stratified_split(rows, args.holdout_frac, args.seed)]
    )

    oof_rows = []
    zs_metrics, zs_rows = None, None
    student = VRPModel(args).to(args.device)
    fc.load_pretrained(student, student_ckpt, args.device)
    if not args.skip_zero_shot:
        zs_metrics, zs_rows = fc.evaluate(
            student, env, td_all, args.device, augmentation, args.eval_batch_size, all_ids
        )
        id_to_row = {r["problem_id"]: r for r in rows}
        for rec in zs_rows:
            src = id_to_row[rec["problem_id"]]
            rec["tag"] = "zero_shot"
            rec["epoch"] = 0
            rec["fold"] = -1
            rec["cluster"] = src["cluster"]
            rec["family"] = FAMILY_NAME[int(src["cluster"])]
            rec["type"] = src["type"]
        fc.write_csv(os.path.join(args.result_root, "oof_zeroshot.csv"), zs_rows)
        fc.log_metrics("[zero_shot full student]", zs_metrics)

    if args.eval_only:
        fc.log("eval_only: done")
        return

    freeze = not args.no_freeze_encoder
    for fold, (train_idx, val_idx) in enumerate(splits):
        fc.log(f"----- distill fold {fold}/{len(splits)-1} train={len(train_idx)} val={len(val_idx)} -----")
        fc.load_pretrained(student, student_ckpt, args.device, quiet=(fold > 0))
        if freeze:
            freeze_encoder_prompt(student, quiet=(fold > 0))
        trainable = [p for p in student.parameters() if p.requires_grad]
        anchor = snapshot_trainable(student)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        td_train = fc.subset_td(td_all, train_idx)
        tours_train = tours[torch.as_tensor(train_idx, dtype=torch.long)]

        for epoch in range(1, args.epochs + 1):
            student.train()
            if freeze:
                student.encoder.eval()
                student.prompt_net.eval()
            n = td_train.batch_size[0]
            perm = torch.randperm(n, device="cpu")
            losses, nlls = [], []
            seen = 0
            while seen < n:
                end = min(seen + args.batch_size, n)
                local = perm[seen:end]
                batch = fc.subset_td(td_train, local.cpu())
                batch_tours = tours_train[local.cpu()]
                optimizer.zero_grad(set_to_none=True)
                nll, _cost = nll_on_tours(student, env, batch, batch_tours, args.device)
                penalty = l2sp_penalty(student, anchor)
                loss = nll + args.l2sp * penalty
                loss.backward()
                clip_grad_norms(optimizer.param_groups, args.grad_clip)
                optimizer.step()
                losses.append(float(loss.item()))
                nlls.append(float(nll.item()))
                seen = end
            fc.log(
                f"fold {fold} epoch {epoch:03d} distill loss={np.mean(losses):.4f} "
                f"nll={np.mean(nlls):.4f} seen={n}"
            )

        td_hold = fc.subset_td(td_all, val_idx)
        hold_ids = [rows[i]["problem_id"] for i in val_idx]
        metrics, per_row = fc.evaluate(
            student, env, td_hold, args.device, augmentation, args.eval_batch_size, hold_ids
        )
        fc.log_metrics(f"[distill e{args.epochs} fold {fold} holdout]", metrics)
        for rec, idx in zip(per_row, val_idx):
            rec["tag"] = "distill"
            rec["epoch"] = args.epochs
            rec["fold"] = fold
            rec["cluster"] = rows[idx]["cluster"]
            rec["family"] = FAMILY_NAME[int(rows[idx]["cluster"])]
            rec["type"] = rows[idx]["type"]
        oof_rows.extend(per_row)
        if args.device == "cuda":
            torch.cuda.empty_cache()

    fc.write_csv(os.path.join(args.result_root, "oof_distill.csv"), oof_rows)
    oof_met = fc.oof_metrics(oof_rows)
    fc.log_metrics("[OOF distill ALL]", oof_met)
    if zs_metrics:
        fc.log_metrics("[zero_shot ALL]", zs_metrics)
    for c in clusters:
        sub = [r for r in oof_rows if int(r["cluster"]) == c]
        fc.log_metrics(f"[OOF distill family{c} {FAMILY_NAME[c]}]", fc.oof_metrics(sub))
        if zs_rows:
            zsub = [r for r in zs_rows if int(r["cluster"]) == c]
            fc.log_metrics(f"[zero_shot family{c} {FAMILY_NAME[c]}]", fc.oof_metrics(zsub))
    summary = {
        "n": oof_met.get("n"),
        "zs_aug_cost": None if zs_metrics is None else zs_metrics["aug_cost"],
        "zs_aug_gap": None if zs_metrics is None else zs_metrics["aug_gap"],
        "distill_aug_cost": oof_met.get("aug_cost"),
        "distill_aug_gap": oof_met.get("aug_gap"),
        "teacher_epoch": args.teacher_epoch,
        "distill_epochs": args.epochs,
        "folds": args.folds,
        "teacher_root": str(teacher_root),
    }
    with open(os.path.join(args.result_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    fc.write_csv(os.path.join(args.result_root, "summary.csv"), [summary])
    fc.log(f"Wrote {args.result_root}/oof_distill.csv")
    fc.log("Distill done")


if __name__ == "__main__":
    main()
