import os
import sys

# TODO: this is a trick to avoid infinite warnings
# but should be removed in the future
sys.stderr = open(os.devnull, "w")

# ruff: noqa: E402
import time

from rl4co.data.utils import load_npz_to_tensordict, save_tensordict_to_npz
from tensordict import TensorDict
from tqdm.auto import tqdm

from routefinder.baselines.solve import solve
from routefinder.envs.mtdvrp import MTVRPEnv
import numpy as np
import torch

# Size to solving time as in paper (seconds)
size_to_time = {
    50: 10,
    100: 20,
}
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="pyvrp")
    parser.add_argument("--size", type=int, default=100)
    args = parser.parse_args()

    solver = args.solver
    size = args.size

    env = MTVRPEnv(check_solution=False)

    data_files = []
    for root, dirs, files in os.walk("data"):
        for file in files:
            if file == f"{size}.npz":
                data_files.append(os.path.join(root, file))

    for file in tqdm(data_files, desc="Collecting instance runtime with " + solver):

        td_test = load_npz_to_tensordict(file)
        num_problems, _ = td_test["demand_linehaul"].shape
        max_runtime = size_to_time[size]

        td_test = env.reset(td_test)

        dataset_times = []

        for i in range(num_problems):

            inst = td_test[i:i+1]

            start = time.time()

            solve(
                inst,
                max_runtime=max_runtime,
                num_procs=1,
                solver=solver
            )

            runtime = time.time() - start
            dataset_times.append(runtime)

        dataset_times = torch.tensor(dataset_times)

        # ====== 构造保存路径 ======

        problem_name = os.path.basename(os.path.dirname(file))
        checkpoint_name = solver

        save_dir = os.path.join("instance_time", str(size), checkpoint_name)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{problem_name}.npz")

        np.savez(
            save_path,
            time=dataset_times.cpu().numpy()
        )

        print("Saved:", save_path)