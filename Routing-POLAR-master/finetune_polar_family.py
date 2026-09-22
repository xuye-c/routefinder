#!/usr/bin/env python3
"""Specialize Polar into 6 constraint-family Prompt/FiLM heads.

Does NOT use official --tune (that is unseen mb/md, 10k generated, full-net 3e-4).

Train: online generator, only this family's 2–4 variants. Freeze encoder/decoder
except PromptNet + FiLM. Never trains on the labeled test npz.

Eval: all labeled test instances of that family (no leakage). Report aug_gap.

Run from Routing-POLAR-master/:

    python -u finetune_polar_family.py --n_size 50
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch

import finetune_cluster as fc
from envs.mtvrp.env import MTVRPEnv
from envs.transformer import StateAugmentation
from models.model import VRPModel
from utils.functions import clip_grad_norms

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


def parse_args():
    p = argparse.ArgumentParser(description="Polar 6-family prompt specialization")
    p.add_argument("--n_size", type=int, required=True, choices=[50, 100])
    p.add_argument("--families", type=str, default="0,1,2,3,4,5")
    p.add_argument("--cluster_csv", type=str, default=None)
    p.add_argument("--epoch", type=int, default=300)
    p.add_argument("--path_id", type=str, default=None)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--eval_batch_size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--episodes", type=int, default=4096, help="Generated instances per epoch")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--loss", type=str, default="po", choices=["po", "rl"])
    p.add_argument("--po_alpha", type=float, default=0.05)
    p.add_argument(
        "--train_pomo",
        type=int,
        default=8,
        help="POMO starts while training (0 = all customers)",
    )
    p.add_argument(
        "--train_modules",
        type=str,
        default="prompt",
        choices=["prompt", "prompt_film", "all"],
        help="prompt = PromptNet; prompt_film = PromptNet+FiLM; all = full net (not recommended)",
    )
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--skip_zero_shot", action="store_true")
    p.add_argument("--result_root", type=str, default=None)
    return p.parse_args()


def parse_families(s: str):
    ids = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    for c in ids:
        if c not in FAMILY_NAME:
            raise ValueError(f"unknown family {c}")
    return ids


def resolve_family_csv(n_size, explicit):
    if explicit:
        path = os.path.abspath(os.path.expanduser(explicit))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    name = f"constraint_family_cluster_6way_{n_size}.csv"
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, name),
        os.path.join(here, "..", name),
        os.path.join(os.path.expanduser("~"), name),
        os.path.join(os.path.expanduser("~"), "routefinder", name),
        os.path.abspath(name),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(f"need {name}; pass --cluster_csv")


def reset_keep_prompt(env, td):
    p_s_tag = td["p_s_tag"].clone() if "p_s_tag" in td.keys() else None
    out = env.reset(td=td)
    if p_s_tag is not None:
        out["p_s_tag"] = p_s_tag.to(out.device)
    return out


def sample_family_batch(env, types, batch_size, device):
    ptype = random.choice(list(types))
    env.generator.reset_variant_preset(ptype)
    env.generator.use_combinations = False
    env.generator.subsample = True
    raw = env.generator(batch_size=batch_size).to(device)
    return reset_keep_prompt(env, raw), ptype


def freeze_except(model, mode: str):
    for param in model.parameters():
        param.requires_grad = False
    trained = 0
    if mode == "all":
        for param in model.parameters():
            param.requires_grad = True
            trained += param.numel()
        print(f"Train ALL params={trained}")
        return
    for name, param in model.named_parameters():
        keep = name.startswith("prompt_net.")
        if mode == "prompt_film" and "film_generator" in name:
            keep = True
        if keep:
            param.requires_grad = True
            trained += param.numel()
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Train {mode}: trainable={trained} frozen={frozen}")
    if trained == 0:
        raise RuntimeError(f"no params matched train_modules={mode}")


def load_family_eval_td(env, csv_path, n_size, cluster):
    rows = fc.load_cluster_rows(csv_path, n_size, cluster)
    td = fc.load_cluster_tensordict(env, rows, n_size)
    return rows, td


def main():
    args = parse_args()
    families = parse_families(args.families)
    if args.path_id is None:
        args.path_id = fc.default_path_id(args.n_size)
    if args.batch_size is None:
        args.batch_size = 64 if args.n_size == 50 else 32

    cli_lr = args.lr
    cli_epochs = args.epochs
    cli_loss = args.loss
    cli_po_alpha = args.po_alpha
    cli_data = args.data_dir

    fc.load_yaml_config(args)
    args.lr = cli_lr
    args.epochs = cli_epochs
    args.loss = cli_loss
    args.po_alpha = cli_po_alpha
    args.data_dir = cli_data
    args.env["data_dir"] = cli_data
    args.env["generator_params"]["num_loc"] = args.n_size
    args.env["generator_params"]["variant_preset"] = FAMILY_TYPES[families[0]][0]
    args.trainer_params["po_B"] = None if args.train_pomo <= 0 else int(args.train_pomo)
    args.trainer_params["use_ls"] = False

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

    csv_path = resolve_family_csv(args.n_size, args.cluster_csv)
    ckpt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "result",
        args.path_id,
        f"checkpoint-{args.epoch}.pt",
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    with open(ckpt_path, "rb") as f:
        if f.read(32).startswith(b"version https://git-lfs.github.com"):
            raise RuntimeError(f"{ckpt_path} is a Git LFS pointer")

    stamp = time.strftime("%Y-%m%d-%H%M", time.localtime())
    if args.result_root:
        result_root = os.path.abspath(os.path.expanduser(args.result_root))
    else:
        result_root = os.path.join(
            "result", f"family-polar-n{args.n_size}-{stamp}"
        )
    os.makedirs(result_root, exist_ok=True)

    print("=" * 72)
    print("POLAR family specialization (generated data, Prompt/FiLM, not --tune)")
    print(f"ckpt         : {ckpt_path}")
    print(f"csv          : {csv_path}")
    print(f"families     : {families}")
    print(f"train_modules: {args.train_modules} lr={args.lr} epochs={args.epochs}")
    print(f"episodes/ep  : {args.episodes} batch={args.batch_size} train_pomo={args.train_pomo}")
    print(f"result_root  : {result_root}")
    print("=" * 72)

    env = MTVRPEnv(**args.env)
    env.set_loss_mode(args.loss)
    augmentation = StateAugmentation()
    summary = []

    for cluster in families:
        name = FAMILY_NAME[cluster]
        types = FAMILY_TYPES[cluster]
        fam_dir = os.path.join(result_root, f"family{cluster}-{name}")
        os.makedirs(fam_dir, exist_ok=True)
        print()
        print("=" * 72)
        print(f"FAMILY {cluster}/5  {name}  types={list(types)}")
        print("=" * 72)

        rows, td_eval = load_family_eval_td(env, csv_path, args.n_size, cluster)
        print(f"eval instances={len(rows)}")

        model = VRPModel(args).to(args.device)
        model.set_loss_mode(args.loss)
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        state = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        del ckpt
        freeze_except(model, args.train_modules)

        history = []

        def run_eval(tag, epoch):
            metrics = fc.evaluate(
                model, env, td_eval, args, augmentation, args.eval_batch_size
            )
            fc.log_metrics(f"[{tag} family{cluster} {name} epoch {epoch}]", metrics)
            rec = {"tag": tag, "epoch": epoch, "cluster": cluster, "family": name, **metrics}
            history.append(rec)
            fc.save_json(os.path.join(fam_dir, "metrics.json"), history)
            return metrics

        zs = None
        if not args.skip_zero_shot:
            zs = run_eval("zero_shot", 0)

        if args.eval_only:
            summary.append({"cluster": cluster, "family": name, "zero_shot": zs})
            continue

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        best = dict(zs) if zs else None
        best_epoch = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            if args.train_modules != "all":
                model.encoder.eval()
                model.decoder.eval()
                if model.prompt_net is not None:
                    model.prompt_net.train()
            losses, costs = [], []
            seen = 0
            while seen < args.episodes:
                bsz = min(args.batch_size, args.episodes - seen)
                td, ptype = sample_family_batch(env, types, bsz, args.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=args.device,
                    dtype=args.amp_dtype,
                    enabled=(args.device == "cuda"),
                ):
                    out = model(td, env, with_greedy=False)
                    batch_n = bsz
                    reward = out["reward"].view(-1, batch_n)
                    log_likelihood = out["log_likelihood"].sum(1).view(-1, batch_n)
                    if args.loss == "po":
                        loss = fc.compute_po_loss(reward, log_likelihood, args.po_alpha)
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
                seen += bsz
                del td, out, reward, log_likelihood, loss
            print(
                f"family{cluster} epoch {epoch:03d} loss={np.mean(losses):.4f} "
                f"best_cost={np.mean(costs):.4f} seen={seen}"
            )
            metrics = run_eval("finetune", epoch)
            ckpt_out = os.path.join(fam_dir, f"tuned-family{cluster}-{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "cluster": cluster,
                    "family": name,
                    "n_size": args.n_size,
                    "model_state_dict": model.state_dict(),
                    "holdout_metrics": metrics,
                },
                ckpt_out,
            )
            if best is None or metrics["aug_gap"] < best["aug_gap"]:
                best = dict(metrics)
                best_epoch = epoch
            elif zs is not None and metrics["aug_gap"] > zs["aug_gap"] + 0.05:
                print(
                    f"STOP family{cluster}: aug_gap {metrics['aug_gap']:.3f}% "
                    f"> zs {zs['aug_gap']:.3f}% + 0.05"
                )
                break

        rec = {
            "cluster": cluster,
            "family": name,
            "n": len(rows),
            "zs_aug_gap": None if zs is None else zs["aug_gap"],
            "zs_aug_cost": None if zs is None else zs["aug_cost"],
            "best_epoch": best_epoch,
            "best_aug_gap": None if best is None else best["aug_gap"],
            "best_aug_cost": None if best is None else best["aug_cost"],
        }
        summary.append(rec)
        print(
            f"family{cluster} {name}: zs aug_gap={rec['zs_aug_gap']} -> "
            f"best e{best_epoch} {rec['best_aug_gap']}"
        )
        if args.device == "cuda":
            torch.cuda.empty_cache()

    fc.save_json(os.path.join(result_root, "summary.json"), summary)
    print("Wrote", os.path.join(result_root, "summary.json"))
    print("Polar family specialization done")


if __name__ == "__main__":
    main()
