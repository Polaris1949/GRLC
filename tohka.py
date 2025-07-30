import random
import warnings

import torch


def zero_ratio(tensor: torch.Tensor) -> float:
    """
    计算张量中0值的数量占总数的比例
    
    Args:
        tensor (torch.Tensor): 输入张量
        
    Returns:
        float: 0值占比（0到1之间的浮点数）
    """
    total_elements = tensor.numel()  # 获取张量总元素数
    if total_elements == 0:
        return 0.0
    zero_count = (tensor == 0).sum().item()  # 计算0值的数量
    return zero_count / total_elements


def get_dataset_yamai(args, dataset_kwargs):
    from train import get_dataset_or_loader
    train_d, val_d, test_d = get_dataset_or_loader(
        args.dataset_class, args.dataset_name, args.data_root,
        batch_size=args.batch_size, seed=args.seed, num_splits=args.data_num_splits,
        **dataset_kwargs,
    )
    return train_d


def analyze_dataset(train_dataloader):
    from data_unit.data_av2 import (MIKU_DIST_MAX, MIKU_DIST_MIN,
                                    MIKU_SPEED_MAX, MIKU_SPEED_MIN, Yamai)
    print(f'{MIKU_DIST_MAX=}')
    print(f'{MIKU_DIST_MIN=}')
    print(f'{MIKU_SPEED_MAX=}')
    print(f'{MIKU_SPEED_MIN=}')
    ZERO_RATIO_MAX = 0.0
    ZERO_RATIO_MIN = 1.0
    for data_idx, data in enumerate(train_dataloader):
        yamai = Yamai.from_batch(data)
        for graph_idx, graph in enumerate(yamai.graphs):
            zr = zero_ratio(graph.x)
            ZERO_RATIO_MAX = max(ZERO_RATIO_MAX, zr)
            ZERO_RATIO_MIN = min(ZERO_RATIO_MIN, zr)
    print(f'{ZERO_RATIO_MAX=}')
    print(f'{ZERO_RATIO_MIN=}')


def check_float(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} nan")
    if torch.isinf(tensor).any():
        print(f"{name} inf")


def random_non_adjacent_permutation(x, edge_index):
    # x: (num_nodes, feature_dim) tensor, vertex features
    # edge_index: (2, num_edges) tensor, edge list
    # Returns: new feature tensor based on randomly selected vertices, preferring non-adjacent ones
    
    num_nodes = x.shape[0]
    device = x.device
    
    # Create adjacency list
    adj_list = [set() for _ in range(num_nodes)]
    for src, dst in edge_index.t():
        adj_list[src.item()].add(dst.item())
        adj_list[dst.item()].add(src.item())
    
    # Find non-adjacent vertices for each vertex
    non_adj_list = []
    for i in range(num_nodes):
        non_adj = set(range(num_nodes)) - adj_list[i] - {i}  # Exclude self and adjacent vertices
        non_adj_list.append(list(non_adj))
    
    # Initialize permutation
    permutation = [-1] * num_nodes
    available = set(range(num_nodes))  # Available vertices to be assigned as j
    
    # Randomly assign vertices, preferring non-adjacent ones
    vertices = list(range(num_nodes))
    random.shuffle(vertices)  # Shuffle to ensure randomness
    
    for i in vertices:
        non_adj = non_adj_list[i]
        # Filter available non-adjacent vertices
        valid_choices = [j for j in non_adj if j in available]
        
        if not valid_choices:
            # No non-adjacent vertices available, warn and use any available vertex
            warnings.warn(f"No non-adjacent vertex available for vertex {i}, selecting from remaining vertices")
            if not available:
                warnings.warn(f"No vertices available for vertex {i}, permutation may be incomplete")
                continue
            valid_choices = list(available)
        
        # Randomly select one vertex
        j = random.choice(valid_choices)
        permutation[i] = j
        available.remove(j)
    
    # Check if permutation is complete
    if -1 in permutation:
        warnings.warn("Incomplete permutation due to insufficient available vertices")
        # Fill remaining with -1 or handle differently if needed
        permutation = [j if j != -1 else random.choice(list(range(num_nodes))) for j in permutation]
    
    # Create new feature tensor based on permutation
    new_x = x[permutation]
    
    return new_x


def random_non_adjacent_selection(x, edge_index):
    # x: (num_nodes, feature_dim) tensor, vertex features
    # edge_index: (2, num_edges) tensor, edge list
    # Returns: new feature tensor based on randomly selected non-adjacent vertices
    
    num_nodes = x.shape[0]
    device = x.device
    
    # Create adjacency list
    adj_list = [set() for _ in range(num_nodes)]
    for src, dst in edge_index.t():
        adj_list[src.item()].add(dst.item())
        adj_list[dst.item()].add(src.item())
    
    # Find non-adjacent vertices for each vertex
    non_adj_list = []
    for i in range(num_nodes):
        non_adj = set(range(num_nodes)) - adj_list[i] - {i}  # Exclude self and adjacent vertices
        non_adj_list.append(list(non_adj))
    
    # Initialize selection
    selected = []
    
    # Randomly select non-adjacent vertices for each vertex
    for i in range(num_nodes):
        non_adj = non_adj_list[i]
        
        if not non_adj:
            # No non-adjacent vertices available, warn and select any vertex
            warnings.warn(f"No non-adjacent vertex available for vertex {i}, selecting random vertex")
            selected.append(random.choice(range(num_nodes)))
        else:
            # Randomly select one non-adjacent vertex
            j = random.choice(non_adj)
            selected.append(j)
    
    # Create new feature tensor based on selected vertices
    new_x = x[selected]
    
    return new_x
