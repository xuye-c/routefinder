import torch
from tensordict.tensordict import TensorDict


def pad_depot_feature(x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Align a per-node feature with locs.

    RouteFinder npz stores customer-only demands of length N.
    CADA (and locs) use depot + customers, length N+1, with depot first.
    Already-padded tensors (length == num_nodes) are returned unchanged.
    """
    last = x.shape[-1]
    if last == num_nodes:
        return x
    if last == num_nodes - 1:
        depot = torch.zeros(
            (*x.shape[:-1], 1),
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([depot, x], dim=-1)
    raise ValueError(
        f"Expected last dim {num_nodes} (with depot) or {num_nodes - 1} "
        f"(customers only), got {tuple(x.shape)}"
    )


def fill_missing_vrp_fields(td: TensorDict) -> TensorDict:
    """
    Fill optional fields omitted by RouteFinder datasets.

    RouteFinder does not store fields for constraints that are not present
    in a problem variant. CADA expects these fields to exist, so we fill
    them with semantically neutral defaults.

    Defaults:
        no backhaul       -> demand_backhaul = 0
        no open route     -> open_route = False
        no time windows   -> time_windows = [0, inf]
        no service time   -> service_time = 0
        no duration limit -> distance_limit = inf
    """

    device = td.device
    batch_size = td.batch_size
    num_nodes = td["locs"].shape[-2]

    # RouteFinder: demand is [B, N] (customers). CADA: [B, N+1] (depot + customers).
    td["demand_linehaul"] = pad_depot_feature(td["demand_linehaul"], num_nodes)
    if "demand_backhaul" in td.keys():
        td["demand_backhaul"] = pad_depot_feature(td["demand_backhaul"], num_nodes)
    else:
        td["demand_backhaul"] = torch.zeros(
            td["demand_linehaul"].shape,
            dtype=td["demand_linehaul"].dtype,
            device=device,
        )

    if "speed" not in td.keys():
        td["speed"] = torch.ones(
            (*batch_size, 1),
            dtype=torch.float32,
            device=device,
        )
    # Closed route by default
    if "open_route" not in td.keys():
        td["open_route"] = torch.zeros(
            (*batch_size, 1),
            dtype=torch.bool,
            device=device,
        )

    # No time-window constraint
    if "time_windows" not in td.keys():
        td["time_windows"] = torch.zeros(
            (*batch_size, num_nodes, 2),
            dtype=torch.float32,
            device=device,
        )
        td["time_windows"][..., 1] = float("inf")

    # No service time
    if "service_time" not in td.keys():
        td["service_time"] = torch.zeros(
            (*batch_size, num_nodes),
            dtype=torch.float32,
            device=device,
        )

    # No duration limit
    if "distance_limit" not in td.keys():
        td["distance_limit"] = torch.full(
            (*batch_size, 1),
            float("inf"),
            dtype=torch.float32,
            device=device,
        )

    return td
