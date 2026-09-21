#!/usr/bin/env python3
"""Build two CADA-family selections from one family-ft result dir.

  epoch1  — every family uses finetune epoch 1
  best    — per family, the checkpoint with lowest holdout aug_gap
            (zero-shot epoch 0 allowed, so a family can keep the pretrained model)

This is NOT 10-fold. Training used one stratified 80/20 split per family
(holdout_frac=0.2, seed=7).

Example on the cluster:

    python pack_cada_family_selection.py \\
        --result_root CADA/50/result/family-ft-n50-2026-0921-1150

Writes:
    <result_root>/selection/epoch1/{selection.csv,holdout.csv}
    <result_root>/selection/best/{selection.csv,holdout.csv}
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def load_metrics(fam_dir: Path):
    path = fam_dir / "metrics.json"
    with open(path) as f:
        rows = json.load(f)
    by_key = {}
    for rec in rows:
        epoch = int(rec["epoch"])
        by_key[epoch] = rec
    return by_key


def family_dirs(result_root: Path):
    dirs = sorted(
        p for p in result_root.iterdir() if p.is_dir() and p.name.startswith("family")
    )
    if not dirs:
        raise FileNotFoundError(f"no family* dirs in {result_root}")
    return dirs


def ckpt_path(fam_dir: Path, cluster: int, epoch: int) -> Path | None:
    if epoch <= 0:
        return None
    p = fam_dir / f"tuned-family{cluster}-{epoch}.pt"
    return p if p.is_file() else None


def holdout_csv(fam_dir: Path, rec: dict) -> Path | None:
    tag = rec.get("tag", "finetune")
    epoch = int(rec["epoch"])
    p = fam_dir / f"holdout_{tag}_e{epoch}.csv"
    return p if p.is_file() else None


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def concat_holdout(out_path: Path, parts: list[tuple[dict, Path]]):
    rows = []
    for rec, csv_path in parts:
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                row["cluster"] = rec["cluster"]
                row["family"] = rec["family"]
                row["selected_epoch"] = rec["epoch"]
                row["selected_tag"] = rec["tag"]
                rows.append(row)
    write_csv(out_path, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_root", type=str, required=True)
    p.add_argument(
        "--copy_ckpts",
        action="store_true",
        help="Also copy selected .pt into selection/{epoch1,best}/ckpts/",
    )
    args = p.parse_args()
    root = Path(args.result_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    epoch1_sel = []
    best_sel = []
    epoch1_hold = []
    best_hold = []

    for fam_dir in family_dirs(root):
        by_ep = load_metrics(fam_dir)
        sample = next(iter(by_ep.values()))
        cluster = int(sample["cluster"])
        family = sample["family"]

        if 1 not in by_ep:
            raise KeyError(f"{fam_dir} has no epoch 1 in metrics.json")
        e1 = dict(by_ep[1])
        e1["ckpt"] = str(ckpt_path(fam_dir, cluster, 1) or "")
        e1["holdout_csv"] = str(holdout_csv(fam_dir, e1) or "")
        e1["rule"] = "epoch1"
        epoch1_sel.append(e1)
        hp = holdout_csv(fam_dir, e1)
        if hp:
            epoch1_hold.append((e1, hp))

        best = min(by_ep.values(), key=lambda r: (r["aug_gap"] is None, r["aug_gap"]))
        best = dict(best)
        best["ckpt"] = str(ckpt_path(fam_dir, cluster, int(best["epoch"])) or "")
        if int(best["epoch"]) == 0:
            best["ckpt"] = "PRETRAINED (epoch 0 / zero-shot, no tuned pt)"
        best["holdout_csv"] = str(holdout_csv(fam_dir, best) or "")
        best["rule"] = "best_aug_gap"
        best_sel.append(best)
        hp = holdout_csv(fam_dir, best)
        if hp:
            best_hold.append((best, hp))

    cols = [
        "rule",
        "cluster",
        "family",
        "tag",
        "epoch",
        "n",
        "no_aug_cost",
        "aug_cost",
        "no_aug_gap",
        "aug_gap",
        "ckpt",
        "holdout_csv",
    ]

    def slim(rows):
        out = []
        for r in rows:
            out.append({k: r.get(k) for k in cols})
        return out

    out_e1 = root / "selection" / "epoch1"
    out_best = root / "selection" / "best"
    write_csv(out_e1 / "selection.csv", slim(epoch1_sel))
    write_csv(out_best / "selection.csv", slim(best_sel))
    if epoch1_hold:
        concat_holdout(out_e1 / "holdout.csv", epoch1_hold)
    if best_hold:
        concat_holdout(out_best / "holdout.csv", best_hold)

    if args.copy_ckpts:
        for rec, dest in ((epoch1_sel, out_e1 / "ckpts"), (best_sel, out_best / "ckpts")):
            dest.mkdir(parents=True, exist_ok=True)
            for r in rec:
                src = Path(r["ckpt"]) if r["ckpt"] and not r["ckpt"].startswith("PRETRAINED") else None
                if src and src.is_file():
                    shutil.copy2(src, dest / src.name)

    print("epoch1:")
    for r in epoch1_sel:
        print(f"  family{r['cluster']} {r['family']:22s} e{r['epoch']} aug_gap={r['aug_gap']}")
    print("best aug_gap:")
    for r in best_sel:
        print(f"  family{r['cluster']} {r['family']:22s} e{r['epoch']} ({r['tag']}) aug_gap={r['aug_gap']}")
    print(f"wrote {out_e1}")
    print(f"wrote {out_best}")


if __name__ == "__main__":
    main()
