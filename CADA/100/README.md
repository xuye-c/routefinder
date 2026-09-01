# Train

```shell
cd ./100
CUDA_VISIBLE_DEVICES=1,2 torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 --nproc_per_node=2 run.py --n_size 100 --test
```

# Test

```shell
cd ./100
CUDA_VISIBLE_DEVICES=0 python run.py --resume --epoch 300 --path_id 2024-1121-1355 --n_size 100 --test --test_only
```

Loads RouteFinder `data/{problem}/test/100.npz` for all 16 types.
Results: `result/<timestamp>/cada_results_100.csv` with `cost` / `aug_cost`.
