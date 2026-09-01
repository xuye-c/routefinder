import torch
from tensordict.tensordict import TensorDict


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

    # No backhaul
    if td["demand_linehaul"].shape[-1] == num_nodes - 1:
        depot_demand = torch.zeros(
            (*batch_size, 1),
            dtype=td["demand_linehaul"].dtype,
            device=device,
        )

        td["demand_linehaul"] = torch.cat(
            [depot_demand, td["demand_linehaul"]],
            dim=-1,
        )
    if "demand_backhaul" not in td.keys():
        td["demand_backhaul"] = torch.zeros(
            td["demand_linehaul"].shape,
            dtype=td["demand_linehaul"].dtype,
            device=td.device,
        )
    if "speed" not in td.keys(): 
        batch_size = td.batch_size 
        td["speed"] = torch.ones( 
            (*batch_size, 1), 
            #td["demand_linehaul"].shape,
            dtype=torch.float32, 
            device=td.device, )
    # Closed route by default
    if "open_route" not in td.keys():
        td["open_route"] = torch.zeros(
            (*batch_size, 1),
            #td["demand_linehaul"].shape,
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