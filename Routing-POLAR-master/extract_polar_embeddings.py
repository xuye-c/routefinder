#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
import run
from trainer import VRPTrainer

# ============================================================
# 1. Parse extraction-specific arguments
# ============================================================

def parse_extraction_args():
    parser = argparse.ArgumentParser(
        description="Extract POLAR encoder embeddings"
    )

    parser.add_argument(
        "--n_size",
        type=int,
        required=True,
        choices=[50, 100],
        help="VRP instance size"
    )

    parser.add_argument(
        "--path_id",
        type=str,
        required=True,
        help="POLAR checkpoint path ID, e.g. n=100/2026-0729-1221"
    )

    parser.add_argument(
        "--polar_root",
        type=str,
        default="~/routefinder/Routing-POLAR-master",
        help="Path to the POLAR project root"
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=300,
        help="Checkpoint epoch"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: <polar_root>/polar_embeddings"
    )

    return parser.parse_args()


# ============================================================
# 2. Main
# ============================================================

def main():

    extraction_args = parse_extraction_args()

    N_SIZE = extraction_args.n_size
    EPOCH = extraction_args.epoch
    PATH_ID = extraction_args.path_id

    POLAR_ROOT = os.path.abspath(
        os.path.expanduser(extraction_args.polar_root)
    )

    if extraction_args.output_dir is None:
        OUTPUT_DIR = os.path.join(
            POLAR_ROOT,
            "polar_embeddings"
        )
    else:
        OUTPUT_DIR = os.path.abspath(
            os.path.expanduser(extraction_args.output_dir)
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Add POLAR root to Python path
    # --------------------------------------------------------

    if POLAR_ROOT not in sys.path:
        sys.path.insert(0, POLAR_ROOT)

    # --------------------------------------------------------
    # Problem order
    # --------------------------------------------------------

    PROBLEM_TYPES = [
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
    ]

    INSTANCES_PER_PROBLEM = 1000

    # ========================================================
    # Helper
    # ========================================================

    def build_problem_ids():
        problem_ids = []

        for problem_type in PROBLEM_TYPES:
            for i in range(INSTANCES_PER_PROBLEM):
                problem_ids.append(
                    f"{problem_type}_{N_SIZE}_{i:04d}"
                )

        return np.asarray(problem_ids)

    # ========================================================
    # Print configuration
    # ========================================================

    print("=" * 70)
    print("POLAR Encoder Embedding Extraction")
    print("=" * 70)

    print(f"POLAR root       : {POLAR_ROOT}")
    print(f"n_size           : {N_SIZE}")
    print(f"epoch            : {EPOCH}")
    print(f"path_id          : {PATH_ID}")
    print(f"output directory : {OUTPUT_DIR}")
    print()

    # ========================================================
    # 3. POLAR initialization
    # ========================================================

    args = run.parse_arguments()

    args.n_size = N_SIZE
    args.test_only = True
    args.resume = True
    args.epoch = EPOCH
    args.path_id = PATH_ID

    args.start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run.load_config(args)
    if args.variant is not None:
        args.tuner_params["variant_present"] = args.variant

    run.setup_device(args)

    run.setup_distributed_training(args)

    run.configure_training_settings(args)

    run.set_test_params(args)

    run.setup_logging_and_wandb(args)

    if args.ddp:
        raise RuntimeError(
            "DDP is not supported by this extraction script. "
            "Please run it on a single GPU."
        )

    run.setup_cuda_optimizations(args)

    run.init_seeds(
        args.seed,
        device=args.device
    )

    print(f"Device: {args.device}")
    print()

    # ========================================================
    # 4. Construct official POLAR Trainer
    # ========================================================

    trainer = VRPTrainer(args)

    model = trainer.model
    model.eval()

    test_dataloader = trainer.test_dataloader

    print("Trainer initialized.")
    print(f"Test dataloader: {test_dataloader}")
    print()

    # ========================================================
    # 5. Build problem IDs
    # ========================================================

    problem_ids = build_problem_ids()

    expected_total = (
        len(PROBLEM_TYPES)
        * INSTANCES_PER_PROBLEM
    )

    print(f"Expected instances: {expected_total}")
    print()

    # ========================================================
    # 6. Extract embeddings
    # ========================================================

    all_embeddings = []

    global_count = 0

    print("Starting encoder extraction...")
    print()

    with torch.inference_mode():

        for problem_type in PROBLEM_TYPES:

            dataloader_name = f"{N_SIZE}_{problem_type}_uniform"

            if dataloader_name not in test_dataloader:
                raise RuntimeError(
                    f"Expected dataloader '{dataloader_name}' not found. "
                    f"Available dataloaders:\n"
                    f"{list(test_dataloader.keys())}"
                )

            dataloader = test_dataloader[dataloader_name]

            problem_embeddings = []

            problem_count = 0

            print(f"Processing {problem_type}...")

            for batch_idx, batch in enumerate(dataloader):
                td = batch.to(args.device)

                # PromptNet + PLE Encoder
                encoder_output = model._encode(td)

                if encoder_output.ndim != 3:
                    raise RuntimeError(
                        f"Unexpected encoder output shape for "
                        f"{problem_type}: {encoder_output.shape}"
                    )

                batch_size, num_nodes, embedding_dim = (
                    encoder_output.shape
                )

                # Expected:
                # n_size=50  -> [B, 51, 128]
                # n_size=100 -> [B, 101, 128]

                expected_nodes = N_SIZE + 1

                if num_nodes != expected_nodes:
                    raise RuntimeError(
                        f"Expected {expected_nodes} nodes for "
                        f"{problem_type}, got {num_nodes}"
                    )

                if embedding_dim != 128:
                    raise RuntimeError(
                        f"Expected embedding dimension 128 for "
                        f"{problem_type}, got {embedding_dim}"
                    )

                # Mean pooling over ALL encoder tokens,
                # including the depot token.
                embedding = encoder_output.mean(dim=1)

                if embedding.shape != (
                    batch_size,
                    128
                ):
                    raise RuntimeError(
                        f"Unexpected pooled embedding shape for "
                        f"{problem_type}: {embedding.shape}"
                    )

                embedding = (
                    embedding
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                problem_embeddings.append(embedding)

                problem_count += batch_size
                global_count += batch_size

                if (
                    batch_idx == 0
                    or (batch_idx + 1) % 10 == 0
                    or problem_count == INSTANCES_PER_PROBLEM
                ):
                    print(
                        f"  Batch {batch_idx + 1:4d} | "
                        f"instances: "
                        f"{problem_count:4d}/{INSTANCES_PER_PROBLEM}"
                    )

            problem_embeddings = np.concatenate(
                problem_embeddings,
                axis=0
            )

            if len(problem_embeddings) != INSTANCES_PER_PROBLEM:
                raise RuntimeError(
                    f"{problem_type}: expected "
                    f"{INSTANCES_PER_PROBLEM} embeddings, "
                    f"got {len(problem_embeddings)}"
                )

            all_embeddings.append(problem_embeddings)

            print(
                f"  Finished {problem_type}: "
                f"{problem_embeddings.shape}"
            )
            print()
    # ========================================================
    # 7. Concatenate
    # ========================================================

    embeddings = np.concatenate(
        all_embeddings,
        axis=0
    )

    print()
    print("Extraction finished.")
    print(f"Embedding shape: {embeddings.shape}")

    # ========================================================
    # 8. Sanity checks
    # ========================================================

    if len(embeddings) != expected_total:
        raise RuntimeError(
            f"Number of embeddings ({len(embeddings)}) "
            f"does not match expected number "
            f"({expected_total})."
        )

    if len(problem_ids) != len(embeddings):
        raise RuntimeError(
            f"problem_id count ({len(problem_ids)}) "
            f"does not match embedding count "
            f"({len(embeddings)})."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Embedding contains NaN or Inf values."
        )

    # ========================================================
    # 9. Save aggregate file
    # ========================================================

    aggregate_path = os.path.join(
        OUTPUT_DIR,
        f"all_{N_SIZE}_mean_polar_embeddings.npz"
    )

    np.savez_compressed(
        aggregate_path,
        problem_id=problem_ids,
        embedding=embeddings,
    )

    print()
    print("Saved aggregate embeddings:")
    print(f"  {aggregate_path}")

    # ========================================================
    # 10. Save per-problem files
    # ========================================================

    for problem_idx, problem_type in enumerate(PROBLEM_TYPES):

        start = (
            problem_idx
            * INSTANCES_PER_PROBLEM
        )

        end = start + INSTANCES_PER_PROBLEM

        problem_ids_i = problem_ids[start:end]
        embeddings_i = embeddings[start:end]

        output_path = os.path.join(
            OUTPUT_DIR,
            f"{problem_type}_{N_SIZE}_mean_polar_embeddings.npz"
        )

        np.savez_compressed(
            output_path,
            problem_id=problem_ids_i,
            embedding=embeddings_i,
        )

        print(
            f"Saved {problem_type:10s}: "
            f"{embeddings_i.shape} -> "
            f"{output_path}"
        )

    # ========================================================
    # 11. Final summary
    # ========================================================

    print()
    print("=" * 70)
    print("Extraction completed successfully.")
    print("=" * 70)
    print(f"Total instances      : {len(embeddings)}")
    print(f"Embedding dimension  : {embeddings.shape[1]}")
    print(f"Problems             : {len(PROBLEM_TYPES)}")
    print(f"Instances / problem  : {INSTANCES_PER_PROBLEM}")
    print("Pooling              : mean over all encoder tokens")
    print(f"Output               : {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()