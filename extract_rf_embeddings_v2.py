import argparse
import os
import numpy as np
import torch
from tqdm.auto import tqdm

from routefinder.data.utils import get_dataloader
from routefinder.envs import MTVRPEnv
from routefinder.models import RouteFinderBase, RouteFinderMoE
from routefinder.models.baselines.mtpomo import MTPOMO
from routefinder.models.baselines.mvmoe import MVMoE


torch.set_float32_matmul_precision("medium")


PROBLEM_TYPES = [
    "cvrp", "ovrp", "ovrpb", "ovrpbl", "ovrpbltw", "ovrpbtw",
    "ovrpl", "ovrpltw", "ovrptw",
    "vrpb", "vrpbl", "vrpbltw", "vrpbtw",
    "vrpl", "vrpltw", "vrptw",
]


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


def get_base_lit_module(checkpoint_path):

    if "mvmoe" in checkpoint_path:
        return MVMoE
    elif "mtpomo" in checkpoint_path:
        return MTPOMO
    elif "moe" in checkpoint_path:
        return RouteFinderMoE
    else:
        return RouteFinderBase


def dataset_to_type_size(dataset_path):

    parts = dataset_path.replace("\\", "/").split("/")

    problem_type = parts[-3]
    size = parts[-1].replace(".npz", "")

    return problem_type, size


def extract_embeddings(
    checkpoint_path,
    problem="all",
    size=100,
    batch_size=1000,
    device="cuda",
    out_dir="encoder_embeddings",
):

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    if "cuda" in device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))


    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading checkpoint:", checkpoint_path)

    BaseLitModule = get_base_lit_module(checkpoint_path)

    model = BaseLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        strict=False,
    )

    policy = model.policy.to(device).eval()

    # --------------------------------------------------
    # Encoder
    # --------------------------------------------------

    encoder = policy.encoder.eval()

    # Exactly the same logic as author's notebook
    if getattr(encoder, "net", None) is None:
        layers = encoder.layers
    else:
        layers = encoder.net.layers

    print("Number of encoder layers:", len(layers))

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    env = MTVRPEnv()

    data_paths = get_dataset_paths(problem, size)

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------

    all_embeddings = []
    all_problem_ids = []


    # --------------------------------------------------
    # Extraction
    # --------------------------------------------------

    with torch.inference_mode():

        for dataset in tqdm(data_paths):

            problem_type, size_str = dataset_to_type_size(dataset)

            print(f"\nLoading {dataset}")

            td_test = env.load_data(dataset)

            dataloader = get_dataloader(
                td_test,
                batch_size=batch_size,
            )

            dataset_embeddings = []
            dataset_problem_ids = []

            offset = 0


            for batch in dataloader:

                # Same idea as author's:
                # td_test = env.reset(td_data.clone())

                td = env.reset(batch.clone()).to(device)

                # --------------------------------------------------
                # Initial embedding
                # --------------------------------------------------

                h = encoder.init_embedding(td)

                # --------------------------------------------------
                # Run through ALL layers
                # BUT only keep the final layer
                # --------------------------------------------------

                for layer in layers:
                    h = layer(h)

                # h:
                # [B, N, H]
                #
                # This is exactly the final encoder representation
                # used by the author's notebook.

                # --------------------------------------------------
                # Mean pooling across nodes
                # --------------------------------------------------

                mean_embeddings = h.mean(dim=1)

                # [B, H]

                mean_embeddings = (
                    mean_embeddings
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )

                dataset_embeddings.append(
                    mean_embeddings
                )


                # --------------------------------------------------
                # IDs
                # --------------------------------------------------

                bsz = mean_embeddings.shape[0]

                problem_ids = [
                    f"{problem_type}_{size_str}_{i:04d}"
                    for i in range(
                        offset,
                        offset + bsz
                    )
                ]

                offset += bsz

                dataset_problem_ids.extend(problem_ids)


            # --------------------------------------------------
            # Dataset result
            # --------------------------------------------------

            dataset_embeddings = np.concatenate(
                dataset_embeddings,
                axis=0,
            )

            dataset_problem_ids = np.array(
                dataset_problem_ids
            )

            print(
                f"{problem_type} {size_str}: "
                f"{dataset_embeddings.shape}"
            )

            # --------------------------------------------------
            # Save individual dataset
            # --------------------------------------------------

            os.makedirs(
                out_dir,
                exist_ok=True,
            )

            out_path = os.path.join(
                out_dir,
                f"{problem_type}_{size_str}"
                f"_final_rf_transformer_embeddings.npz",
            )

            np.savez_compressed(
                out_path,
                problem_id=dataset_problem_ids,
                embedding=dataset_embeddings,
            )

            print("Saved:", out_path)


            # --------------------------------------------------
            # Global
            # --------------------------------------------------

            all_embeddings.append(
                dataset_embeddings
            )

            all_problem_ids.extend(
                dataset_problem_ids.tolist()
            )


    # --------------------------------------------------
    # Combine all datasets
    # --------------------------------------------------

    all_embeddings = np.concatenate(
        all_embeddings,
        axis=0,
    )

    all_problem_ids = np.array(
        all_problem_ids
    )


    # --------------------------------------------------
    # Save global embedding
    # --------------------------------------------------

    all_out_path = os.path.join(
        out_dir,
        f"all_{size}_final_rf_transformer_embeddings.npz",
    )

    np.savez_compressed(
        all_out_path,
        problem_id=all_problem_ids,
        embedding=all_embeddings,
    )


    print("\n======================================")
    print("[DONE]")
    print("Saved:", all_out_path)
    print("Embedding shape:", all_embeddings.shape)
    print("Problem IDs:", all_problem_ids.shape)
    print("======================================")


    return all_out_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/100/rf-transformer.ckpt",
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
        batch_size=args.batch_size,
        device=args.device,
        out_dir=args.out_dir,
        problem=args.problem,
        size=args.size,
        batch_size=args.batch_size,
        device=args.device,
        out_dir=args.out_dir,
    )