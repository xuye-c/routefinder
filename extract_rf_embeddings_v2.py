import argparse
import os
import numpy as np
import torch
import functools

from routefinder.envs import MTVRPEnv, MTVRPGenerator
from routefinder.models import RouteFinderBase, RouteFinderMoE
from routefinder.models.baselines.mtpomo import MTPOMO
from routefinder.models.baselines.mvmoe import MVMoE


# ============================================================
# Compatibility for newer PyTorch
# ============================================================

_original_torch_load = torch.load
@functools.wraps(_original_torch_load)
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

torch.set_float32_matmul_precision("medium")


# ============================================================
# Same variant order as author's notebook
# ============================================================

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


# ============================================================
# Checkpoint -> Lightning module
# ============================================================

def get_base_lit_module(checkpoint_path):

    if "mvmoe" in checkpoint_path:
        return MVMoE
    elif "mtpomo" in checkpoint_path:
        return MTPOMO
    elif "moe" in checkpoint_path:
        return RouteFinderMoE
    else:
        return RouteFinderBase


# ============================================================
# Extract final encoder embedding
#
# This follows the author's notebook:
#
# h = encoder.init_embedding(td)
# for layer in layers:
#     h = layer(h)
#
# Then mean over nodes.
# ============================================================

def get_final_encoder_embedding(
    policy,
    td_data,
    env,
    device,
):

    encoder = policy.encoder.eval()

    # Author's code:
    #
    # if getattr(encoder, 'net', None) is None:
    #     layers = encoder.layers
    # else:
    #     layers = encoder.net.layers

    if getattr(encoder, "net", None) is None:
        layers = encoder.layers
    else:
        layers = encoder.net.layers

    with torch.inference_mode():

        # Same as author's:
        td_test = env.reset(td_data.clone())

        # Initial embedding
        h = encoder.init_embedding(
            td_test.to(device)
        )

        # Pass through every encoder layer
        for layer in layers:
            h = layer(h)

        # h:
        # [batch, num_nodes, embed_dim]
        #
        # Author:
        # mean_encodings = encodings.mean(axis=1)

        mean_embedding = h.mean(dim=1)

    return mean_embedding.detach().cpu().float().numpy()


# ============================================================
# Random generation mode
# ============================================================

def extract_random_embeddings(
    checkpoint_path,
    size=100,
    samples_per_variant=100,
    device="cuda",
    out_dir="encoder_embeddings",
):

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if "cuda" in device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print("=" * 70)
    print("Loading checkpoint:")
    print(checkpoint_path)
    print("=" * 70)

    BaseLitModule = get_base_lit_module(checkpoint_path)

    model = BaseLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        strict=False,
    )

    policy = model.policy.to(device).eval()

    print("Model loaded.")
    print("Device:", device)

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_embeddings = []
    all_labels = []
    all_problem_ids = []

    # --------------------------------------------------------
    # EXACTLY follow author's variant loop
    # --------------------------------------------------------

    for variant in PROBLEM_TYPES:

        print()
        print("-" * 70)
        print(f"Generating {samples_per_variant} random instances:")
        print(f"  variant = {variant}")
        print(f"  size    = {size}")
        print("-" * 70)

        # EXACTLY author's code
        generator = MTVRPGenerator(
            num_loc=size,
            variant_preset=variant,
        )

        env = MTVRPEnv(
            generator,
            check_solution=False,
        )

        # EXACTLY author's code
        td_data = env.generator(samples_per_variant)

        print("Generated:", len(td_data))

        # ----------------------------------------------------
        # Extract final encoder representation
        # ----------------------------------------------------

        embeddings = get_final_encoder_embedding(
            policy=policy,
            td_data=td_data,
            env=env,
            device=device,
        )

        print("Embedding shape:", embeddings.shape)

        # Should be:
        #
        # (100, 128)
        #

        all_embeddings.append(embeddings)

        all_labels.extend(
            [variant] * embeddings.shape[0]
        )

        all_problem_ids.extend(
            [
                f"{variant}_{size}_{i:04d}"
                for i in range(embeddings.shape[0])
            ]
        )

    # ========================================================
    # Combine
    # ========================================================

    all_embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    all_labels = np.array(all_labels)

    all_problem_ids = np.array(all_problem_ids)

    print()
    print("=" * 70)
    print("[DONE]")
    print("=" * 70)

    print("Total instances:", len(all_embeddings))
    print("Embedding shape:", all_embeddings.shape)
    print("Labels shape:", all_labels.shape)

    # Should be:
    #
    # Total instances: 1600
    # Embedding shape: (1600, 128)
    #

    # ========================================================
    # Save
    # ========================================================

    checkpoint_name = os.path.basename(
        checkpoint_path
    ).replace(".ckpt", "")

    out_path = os.path.join(
        out_dir,
        f"random_{size}_{samples_per_variant}_{checkpoint_name}_embeddings.npz",
    )

    np.savez_compressed(
        out_path,
        problem_id=all_problem_ids,
        type=all_labels,
        embedding=all_embeddings,
    )

    print()
    print("Saved:")
    print(out_path)

    return out_path


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--size",
        type=int,
        required=True,
        choices=[50, 100],
    )

    parser.add_argument(
        "--samples_per_variant",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="encoder_embeddings",
    )

    args = parser.parse_args()

    extract_random_embeddings(
        checkpoint_path=args.checkpoint,
        size=args.size,
        samples_per_variant=args.samples_per_variant,
        device=args.device,
        out_dir=args.out_dir,
    )