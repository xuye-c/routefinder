import argparse
import os
import numpy as np
import torch
from tqdm.auto import tqdm
import functools

from routefinder.data.utils import get_dataloader
from routefinder.envs import MTVRPEnv
from routefinder.models.baselines.mvmoe import MVMoE


# ============================================================
# PyTorch checkpoint compatibility
# ============================================================

_original_torch_load = torch.load


@functools.wraps(_original_torch_load)
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

torch.set_float32_matmul_precision("medium")


# ============================================================
# Problem types
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
# Dataset utilities
# ============================================================

def get_dataset_paths(problem, size):

    if problem != "all":
        return [f"data/{problem}/test/{size}.npz"]

    data_paths = []

    for problem_type in PROBLEM_TYPES:

        fp = f"data/{problem_type}/test/{size}.npz"

        if os.path.exists(fp):
            data_paths.append(fp)
        else:
            print(f"[WARN] missing: {fp}")

    if len(data_paths) == 0:
        raise FileNotFoundError("No datasets found.")

    print(f"Using {len(data_paths)} datasets:")

    for p in data_paths:
        print(" ", p)

    return data_paths


def dataset_to_type_size(dataset_path):

    parts = dataset_path.replace("\\", "/").split("/")

    problem_type = parts[-3]
    size = parts[-1].replace(".npz", "")

    return problem_type, size


# ============================================================
# Mean pooling
# ============================================================

def mean_pool_embeddings(x):
    """
    x:
        [B, N, H]

    return:
        [B, H]

    Exactly the same pooling used for the RF encoder
    comparison / author's visualization.
    """

    return x.mean(dim=1)


# ============================================================
# MVMoE encoder extraction
# ============================================================

def extract_mvmoe_encoder(td, encoder):
    """
    Explicitly reproduce MVMoEEncoder.forward():

        h = self.init_embedding(td)
        h = self.net(h)
        return h

    We manually iterate through encoder.net.layers so that
    the final encoder representation is explicitly obtained.

    Returns
    -------
    final_h:
        [B, N, 128]

    init_h:
        [B, N, 128]
    """

    # --------------------------------------------------------
    # Initial embedding
    # --------------------------------------------------------

    init_h = encoder.init_embedding(td)

    h = init_h

    # --------------------------------------------------------
    # Encoder layers
    # --------------------------------------------------------

    for layer in encoder.net.layers:
        h = layer(h)

    # h = final encoder layer
    return h, init_h


# ============================================================
# Main extraction
# ============================================================

def extract_embeddings(
    checkpoint_path,
    problem="all",
    size=50,
    batch_size=1000,
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

    print("========================================")
    print("Loading MVMoE checkpoint")
    print("========================================")
    print(checkpoint_path)

    # --------------------------------------------------------
    # Load MVMoE Lightning module
    # --------------------------------------------------------

    model = MVMoE.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        strict=False,
    )

    model = model.to(device)
    model.eval()

    # --------------------------------------------------------
    # Get the actual MVMoE policy encoder
    # --------------------------------------------------------

    policy = model.policy
    encoder = policy.encoder

    encoder.eval()

    print("\nModel:", type(model))
    print("Policy:", type(policy))
    print("Encoder:", type(encoder))
    print("Init embedding:", type(encoder.init_embedding))
    print("Encoder network:", type(encoder.net))
    print("Number of encoder layers:", len(encoder.net.layers))

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    env = MTVRPEnv()

    data_paths = get_dataset_paths(
        problem,
        size,
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    all_embeddings = []
    all_init_embeddings = []
    all_problem_ids = []

    # ========================================================
    # Extraction
    # ========================================================

    with torch.inference_mode():

        for dataset in tqdm(data_paths):

            problem_type, size_str = dataset_to_type_size(dataset)

            print(f"\nLoading {dataset}")

            td_test = env.load_data(dataset)

            dataloader = get_dataloader(
                td_test,
                batch_size=batch_size,
            )

            offset = 0

            dataset_embeddings = []
            dataset_init_embeddings = []
            dataset_problem_ids = []

            # ------------------------------------------------
            # Process batches
            # ------------------------------------------------

            for batch in dataloader:

                td = env.reset(batch).to(device)

                # ====================================================
                # MVMoE encoder
                #
                # final_h : [B, N, 128]
                # init_h  : [B, N, 128]
                # ====================================================

                final_h, init_h = extract_mvmoe_encoder(
                    td,
                    encoder,
                )

                # ====================================================
                # Mean pooling
                #
                # [B, N, 128]
                #      ↓
                # [B, 128]
                # ====================================================

                emb = mean_pool_embeddings(final_h)

                init_emb = mean_pool_embeddings(init_h)

                # ------------------------------------------------
                # CPU / numpy
                # ------------------------------------------------

                emb_np = (
                    emb
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )

                init_emb_np = (
                    init_emb
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )

                bsz = emb_np.shape[0]

                # ------------------------------------------------
                # Problem IDs
                # ------------------------------------------------

                problem_ids = [
                    f"{problem_type}_{size_str}_{i:04d}"
                    for i in range(
                        offset,
                        offset + bsz,
                    )
                ]

                offset += bsz

                dataset_embeddings.append(
                    emb_np
                )

                dataset_init_embeddings.append(
                    init_emb_np
                )

                dataset_problem_ids.extend(
                    problem_ids
                )

            # ====================================================
            # Combine current dataset
            # ====================================================

            dataset_embeddings = np.concatenate(
                dataset_embeddings,
                axis=0,
            )

            dataset_init_embeddings = np.concatenate(
                dataset_init_embeddings,
                axis=0,
            )

            print(
                problem_type,
                size_str,
                "embedding:",
                dataset_embeddings.shape,
            )

            print(
                problem_type,
                size_str,
                "init embedding:",
                dataset_init_embeddings.shape,
            )

            # ====================================================
            # Save individual problem type
            # ====================================================

            out_path = os.path.join(
                out_dir,
                f"{problem_type}_{size_str}_mean_mvmoe_embeddings.npz",
            )

            np.savez_compressed(
                out_path,
                problem_id=np.array(
                    dataset_problem_ids
                ),
                embedding=dataset_embeddings,
                init_embedding=dataset_init_embeddings,
            )

            print("Saved:", out_path)

            # ------------------------------------------------
            # Add to global collection
            # ------------------------------------------------

            all_embeddings.append(
                dataset_embeddings
            )

            all_init_embeddings.append(
                dataset_init_embeddings
            )

            all_problem_ids.extend(
                dataset_problem_ids
            )

    # ========================================================
    # Combine ALL problem types
    # ========================================================

    all_embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    all_init_embeddings = np.concatenate(
        all_init_embeddings,
        axis=0,
    )

    # ========================================================
    # Save global file
    # ========================================================

    all_out_path = os.path.join(
        out_dir,
        f"all_{size}_mean_mvmoe_embeddings.npz",
    )

    np.savez_compressed(
        all_out_path,
        problem_id=np.array(
            all_problem_ids
        ),
        embedding=all_embeddings,
        init_embedding=all_init_embeddings,
    )

    # ========================================================
    # Summary
    # ========================================================

    print("\n========================================")
    print("[DONE]")
    print("========================================")

    print(
        "Saved all embeddings:",
        all_out_path,
    )

    print(
        "embedding shape:",
        all_embeddings.shape,
    )

    print(
        "init_embedding shape:",
        all_init_embeddings.shape,
    )

    return all_out_path


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/100/mvmoe.ckpt",
    )

    parser.add_argument(
        "--problem",
        type=str,
        default="all",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
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

    extract_embeddings(
        checkpoint_path=args.checkpoint,
        problem=args.problem,
        size=args.size,
       

        device=args.device,
        out_dir=args.out_dir,
    )