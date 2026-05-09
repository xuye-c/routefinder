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
    "vrpb", "vrpbl", "vrpbltw", "vrpbtw", "vrpl", "vrpltw", "vrptw",
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
    # data/cvrp/test/50.npz -> cvrp, 50
    parts = dataset_path.replace("\\", "/").split("/")
    problem_type = parts[-3]
    size = parts[-1].replace(".npz", "")
    return problem_type, size


def extract_embeddings(
    checkpoint_path,
    problem="all",
    size=50,
    batch_size=1000,
    device="cuda",
    out_dir="encoder_embeddings",
    pooling="mean",
):
    if "cuda" in device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    os.makedirs(out_dir, exist_ok=True)

    print("Loading checkpoint:", checkpoint_path)

    BaseLitModule = get_base_lit_module(checkpoint_path)

    model = BaseLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        strict=False,
    )

    env = MTVRPEnv()
    policy = model.policy.to(device).eval()

    if not hasattr(policy, "encoder"):
        raise AttributeError("policy has no encoder attribute.")

    data_paths = get_dataset_paths(problem, size)

    print("Found datasets:")
    for p in data_paths:
        print(" ", p)

    all_embeddings = []
    all_init_embeddings = []
    all_problem_ids = []

    with torch.inference_mode():
        for dataset in tqdm(data_paths):
            problem_type, size_str = dataset_to_type_size(dataset)

            print(f"\nLoading {dataset}")
            td_test = env.load_data(dataset)
            dataloader = get_dataloader(td_test, batch_size=batch_size)

            offset = 0

            dataset_embeddings = []
            dataset_init_embeddings = []
            dataset_problem_ids = []

            for batch in dataloader:
                td = env.reset(batch).to(device)

                # h: final transformer node embedding [B, N, H]
                # init_h: initial node embedding [B, N, H]
                h, init_h = policy.encoder(td)

                if pooling == "mean":
                    emb = h.mean(dim=1)
                    init_emb = init_h.mean(dim=1)

                elif pooling == "depot":
                    emb = h[:, 0, :]
                    init_emb = init_h[:, 0, :]

                elif pooling == "mean_max":
                    emb = torch.cat(
                        [
                            h.mean(dim=1),
                            h.max(dim=1).values,
                        ],
                        dim=-1,
                    )
                    init_emb = torch.cat(
                        [
                            init_h.mean(dim=1),
                            init_h.max(dim=1).values,
                        ],
                        dim=-1,
                    )

                else:
                    raise ValueError(f"Unknown pooling: {pooling}")

                emb_np = emb.detach().cpu().float().numpy()
                init_emb_np = init_emb.detach().cpu().float().numpy()

                bsz = emb_np.shape[0]

                problem_ids = [
                    f"{problem_type}_{size_str}_{i:04d}"
                    for i in range(offset, offset + bsz)
                ]

                offset += bsz

                dataset_embeddings.append(emb_np)
                dataset_init_embeddings.append(init_emb_np)
                dataset_problem_ids.extend(problem_ids)

            dataset_embeddings = np.concatenate(dataset_embeddings, axis=0)
            dataset_init_embeddings = np.concatenate(dataset_init_embeddings, axis=0)

            print(problem_type, size_str, dataset_embeddings.shape)

            out_path = os.path.join(
                out_dir,
                f"{problem_type}_{size_str}_{pooling}_rf_transformer_embeddings.npz",
            )

            np.savez_compressed(
                out_path,
                problem_id=np.array(dataset_problem_ids),
                embedding=dataset_embeddings,
                init_embedding=dataset_init_embeddings,
            )

            print("Saved:", out_path)

            all_embeddings.append(dataset_embeddings)
            all_init_embeddings.append(dataset_init_embeddings)
            all_problem_ids.extend(dataset_problem_ids)

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_init_embeddings = np.concatenate(all_init_embeddings, axis=0)

    all_out_path = os.path.join(
        out_dir,
        f"all_{size}_{pooling}_rf_transformer_embeddings.npz",
    )

    np.savez_compressed(
        all_out_path,
        problem_id=np.array(all_problem_ids),
        embedding=all_embeddings,
        init_embedding=all_init_embeddings,
    )

    print("\n[DONE] saved all embeddings:", all_out_path)
    print("embedding shape:", all_embeddings.shape)
    print("init_embedding shape:", all_init_embeddings.shape)

    return all_out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/50/rf-transformer.ckpt",
    )
    parser.add_argument("--problem", type=str, default="all")
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "depot", "mean_max"],
    )
    parser.add_argument("--out_dir", type=str, default="encoder_embeddings")

    args = parser.parse_args()

    extract_embeddings(
        checkpoint_path=args.checkpoint,
        problem=args.problem,
        size=args.size,
        batch_size=args.batch_size,
        device=args.device,
        out_dir=args.out_dir,
        pooling=args.pooling,
    )