import argparse
import os
import pickle
import time
import warnings

import torch
from rl4co.data.utils import save_tensordict_to_npz
from tensordict import TensorDict

from rl4co.data.transforms import StateAugmentation
from rl4co.utils.ops import gather_by_index, unbatchify
from tqdm.auto import tqdm

from routefinder.data.utils import get_dataloader
from routefinder.envs import MTVRPEnv
from routefinder.models import RouteFinderBase, RouteFinderMoE
from routefinder.models.baselines.mtpomo import MTPOMO
from routefinder.models.baselines.mvmoe import MVMoE

import numpy as np
try:
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
except AttributeError:
    pass

torch.set_float32_matmul_precision("medium")


# =====================================================
# Runtime test
# =====================================================

def test(policy, td, env, device="cuda"):

    per_instance_runtimes = []

    with torch.inference_mode():

        for i in range(td.batch_size[0]):

            single_td = td[i:i+1]

            if "cuda" in str(device):
                torch.cuda.synchronize()

            start_time = time.time()

            _ = policy(single_td, env, phase="test")

            if "cuda" in str(device):
                torch.cuda.synchronize()

            end_time = time.time()

            per_instance_runtimes.append(end_time - start_time)

    return {
        "per_instance_runtime": torch.tensor(per_instance_runtimes)
    }


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="Problem size (50 / 100)",
    )

    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional dataset list",
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
        "--remove-mixed-backhaul",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    opts = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*weights_only.*")

    # =====================================================
    # Device
    # =====================================================

    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # =====================================================
    # Dataset discovery
    # =====================================================

    if opts.datasets is not None:

        data_paths = opts.datasets.split(",")

    else:

        data_paths = []

        for root, _, files in os.walk("data"):

            for file in files:

                if "test" not in root:
                    continue

                if not file.endswith(".npz"):
                    continue

                if opts.remove_mixed_backhaul and "m" in root:
                    continue

                if str(opts.size) in file:

                    if file == "50.npz" or file == "100.npz":

                        data_paths.append(os.path.join(root, file))

        assert len(data_paths) > 0, "No datasets found."

        data_paths = sorted(data_paths)

    print("Found datasets:")
    for p in data_paths:
        print(p)

    # =====================================================
    # Load model
    # =====================================================

    print("Loading checkpoint:", opts.checkpoint)

    if "mvmoe" in opts.checkpoint:
        BaseLitModule = MVMoE
    elif "mtpomo" in opts.checkpoint:
        BaseLitModule = MTPOMO
    elif "moe" in opts.checkpoint:
        BaseLitModule = RouteFinderMoE
    else:
        BaseLitModule = RouteFinderBase

    model = BaseLitModule.load_from_checkpoint(
        opts.checkpoint,
        map_location="cpu",
        strict=False
    )

    env = MTVRPEnv()

    policy = model.policy.to(device).eval()

    # =====================================================
    # Save directory
    # =====================================================

    checkpoint_name = os.path.basename(opts.checkpoint).split(".")[0]

    save_dir = os.path.join("instance_time", str(opts.size))

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, checkpoint_name,f"{checkpoint_name}.npz")##这里这里

    # =====================================================
    # Inference
    # =====================================================

    all_times = []
    for dataset in tqdm(data_paths):

        print("Loading", dataset)

        problem_name = dataset.split("/")[1]

        td_test = env.load_data(dataset)

        dataloader = get_dataloader(td_test, batch_size=opts.batch_size)

        dataset_times = []

        for batch in dataloader:

            td = env.reset(batch).to(device)

            o = test(policy, td, env, device=device)

            dataset_times.append(o["per_instance_runtime"])

        dataset_times = torch.cat(dataset_times)

        save_dir = os.path.join("instance_time1", str(opts.size), checkpoint_name)

        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{problem_name}.npz")

        np.savez(
            save_path,
            time=dataset_times.cpu().numpy()
        )

        print(
            f"{problem_name} | "
            f"Mean {dataset_times.mean().item():.6f}s | "
            f"Saved -> {save_path}"
        )   