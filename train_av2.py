import random
import time
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
from colorama import init, Fore, Back, Style
import numpy as np
from tqdm import tqdm
from data_all import getattr_d, get_dataset_or_loader
from data_unit.utils import blind_other_gpus
from models import LogReg, GRLC_GCN_test
from munkres import Munkres
from sklearn import metrics
from tensorboardX import SummaryWriter
import os
import argparse
from ruamel.yaml import YAML
from termcolor import cprint

from train import get_args, pprint_args, normalize_graph
from torch.utils.data import DataLoader as EasyDataLoader
from data_unit.data_av2 import Yamai, YamaiGraph
from torch_geometric.data import Data as TGData

GRLC_DEVICE = 'cuda'
GRLC_NUM_NODES = 135
GRLC_NUM_FEATURES = 4

def get_dataset_yamai(args, dataset_kwargs):
    train_d, val_d, test_d = get_dataset_or_loader(
        args.dataset_class, args.dataset_name, args.data_root,
        batch_size=args.batch_size, seed=args.seed, num_splits=args.data_num_splits,
        **dataset_kwargs,
    )
    return train_d

def get_data_yamai(graph: YamaiGraph):
    x=F.pad(graph.x, (0, 0, 0, GRLC_NUM_NODES-graph.x.size(0))).float().to(GRLC_DEVICE)
    #print(f'{x.shape=}')
    train_d = TGData(x=x, edge_index=graph.edge_index)
    eps = 2.2204e-16
    norm = train_d.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
    train_d.x = train_d.x.div(norm.expand_as(train_d.x))

    i = train_d.edge_index.long()
    v = torch.FloatTensor(torch.ones([train_d.num_edges]))

    A_sp = torch.sparse.FloatTensor(i, v, torch.Size([train_d.num_nodes, train_d.num_nodes]))
    A = A_sp.to_dense()
    I = torch.eye(A.shape[1]).to(A.device)
    A_I = A + I
    A_I_nomal = normalize_graph(A_I)
    return train_d, [A_I, A_I_nomal, A_sp], [train_d.x]


def check_nan(tensor, message="Tensor contains NaN values"):
    """
    检查张量中是否存在 NaN 值，若存在则打印消息。
    
    Args:
        tensor (torch.Tensor): 输入张量
        message (str): 发现 NaN 值时打印的消息，默认为 "Tensor contains NaN values"
        
    Returns:
        bool: True 如果张量包含 NaN，False 否则
    """
    has_nan = torch.isnan(tensor).any()
    if has_nan:
        print(message)
    return has_nan


def run_GCN_yamai(args, gpu_id=None, exp_name=None, number=0):
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    writename = "-" + exp_name[4:] + "_" + str(number)
    logname = os.path.join(exp_name, str(number) + "_" + str(args.seed) + ".txt")
    logfile = open(logname, 'a')
    writer = SummaryWriter(comment=writename)
    final_acc = 0
    best_acc = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = "cpu" if gpu_id is None \
        else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    dataset_kwargs = {}

    train_dataloader: EasyDataLoader[Yamai] = get_dataset_yamai(args, dataset_kwargs)

    useA = True  # TODO
    # TODO: nb_feature=4 or 128?
    model = GRLC_GCN_test(GRLC_NUM_NODES, GRLC_NUM_FEATURES, args.dim,
                          dim_x=args.dim_x, useact=args.usingact, liner=args.UsingLiner,
                          dropout=args.dropout, useA=useA)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    model.to(running_device)
    my_margin = args.margin1
    my_margin_2 = my_margin + args.margin2
    margin_loss = torch.nn.MarginRankingLoss(margin=my_margin, reduce=False)
    num_neg = args.NN
    print(num_neg)
    print(f'{args.w_loss1=}, {args.w_loss2=}')

    for current_iter, epoch in enumerate(tqdm(range(args.start_epoch, args.start_epoch + args.epochs + 1))):
        # print(f"!0, {epoch=}")
        for data_idx, data in enumerate(train_dataloader):
            # TODO: Multibatch
            yamai = Yamai.from_batch(data)
            # print(f"!1, {yamai}")
            for graph_idx, graph in enumerate(yamai.a2a_graphs):
                train_d, adj_list, x_list = get_data_yamai(graph)
                
                A_I = adj_list[0]
                A_I_nomal = adj_list[1]

                nb_edges = train_d.num_edges
                nb_nodes = train_d.num_nodes
                #print(f'{nb_nodes=}') #135
                nb_feature = train_d.num_features

                I = (torch.eye(nb_nodes).to(running_device) == 1)
                zero_vec = 0.0 * torch.ones_like(A_I_nomal)

                A_I_nomal_dense = A_I_nomal
                I_input = torch.eye(A_I_nomal.shape[1])  # .to(A_I_nomal.device)

                A_I_nomal = A_I_nomal.to(running_device)
                I_input = I_input.to(A_I_nomal.device)
                mask_I = I.to(running_device)
                zero_vec = zero_vec.to(running_device)

                model.train()
                optimiser.zero_grad()
                idx = np.random.permutation(nb_nodes)
                feature_X = x_list[0].to(running_device)
                lbl_z = torch.tensor([0.]).to(running_device)
                feature_a = feature_X
                feature_p = feature_X
                feature_n = []
                idx_list = []
                # idx_lable = []
                for i in range(num_neg):
                    idx_0 = np.random.permutation(nb_nodes)
                    idx_list.append(idx_0)
                    # idx_lable.append(lable[idx_0])
                    feature_temp = feature_X[idx_0]
                    feature_n.append(feature_temp)
                h_a, h_p, h_n_lsit, h_a_0, h_p_0, h_n_0_list = model(feature_a, feature_p, feature_n, A_I_nomal, I=I_input)
                check_nan(h_a, 'h_a nan')
                check_nan(h_p, 'h_p nan')
                s_p = F.pairwise_distance(h_a, h_p)
                check_nan(s_p, 's_p nan')
                cos_0_list = []
                for h_n_0 in h_n_0_list:
                    cos_0 = F.pairwise_distance(h_a_0, h_n_0)
                    cos_0_list.append(cos_0)
                cos_0_stack = torch.stack(cos_0_list).detach()
                check_nan(cos_0_stack, 'cos_0_stack nan')
                cos_0_min = cos_0_stack.min(dim=0)[0]
                cos_0_max = cos_0_stack.max(dim=0)[0]
                gap = cos_0_max - cos_0_min
                weight_list = []
                for i in range(cos_0_stack.size()[0]):
                    weight_list.append((cos_0_stack[i] - cos_0_min) / gap)
                check_nan(torch.stack(weight_list), 'weight_list nan')
                s_n_list = []
                s_n_cosin_list = []
                for h_n in h_n_lsit:
                    s_n = F.pairwise_distance(h_a, h_n)
                    s_n_list.append(s_n)
                margin_label = -1 * torch.ones_like(s_p)
                loss_mar = 0
                mask_margin_N = 0
                i = 0
                for s_n in s_n_list:
                    loss_mar += (margin_loss(s_p, s_n, margin_label) * weight_list[i]).mean()
                    mask_margin_N += torch.max((s_n - s_p.detach() - my_margin_2), lbl_z).sum()
                    i += 1
                mask_margin_N = mask_margin_N / num_neg
                string_1 = " loss_1: {:.3f}||loss_2: {:.3f}||".format(loss_mar.item(), mask_margin_N.item())
                loss = loss_mar * args.w_loss1 + mask_margin_N * args.w_loss2 / nb_nodes
                loss.backward()
                optimiser.step()
                tqdm.write(string_1)  # TODO


if __name__ == '__main__':
    num_total_runs = 10
    main_args = get_args(
        model_name="GRLC",
        dataset_class="Planetoid",
        dataset_name="Cora",
        custom_key="classification",
    )
    pprint_args(main_args)
    filePath = "log"
    exp_ID = 0
    for filename in os.listdir(filePath):
        file_info = filename.split("_")
        file_dataname = file_info[0]
        if file_dataname == main_args.dataset_name:
            exp_ID = max(int(file_info[1]), exp_ID)
    exp_name = main_args.dataset_name + "_" + str(exp_ID + 1)
    exp_name = os.path.join(filePath, exp_name)
    os.makedirs(exp_name)
    if len(main_args.black_list) == main_args.num_gpus_total:
        alloc_gpu = [None]
        cprint("Use CPU", "yellow")
    else:
        alloc_gpu = blind_other_gpus(num_gpus_total=main_args.num_gpus_total,
                                     num_gpus_to_use=main_args.num_gpus_to_use,
                                     black_list=main_args.black_list)
        if not alloc_gpu:
            alloc_gpu = [int(np.random.choice([g for g in range(main_args.num_gpus_total)
                                               if g not in main_args.black_list], 1))]
        cprint("Use GPU the ID of which is {}".format(alloc_gpu), "yellow")

    run_GCN_yamai(main_args, gpu_id=alloc_gpu[0], exp_name=exp_name, number=0)
