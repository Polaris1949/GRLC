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

CKPT_PATH = "runs"
GRLC_DEVICE = 'cuda'
MAX_NUM_AGENTS = 135
TIME_SPAN = 10
GRLC_NUM_NODES = MAX_NUM_AGENTS * TIME_SPAN
GRLC_NUM_FEATURES = 4 + 3 * GRLC_NUM_NODES

def get_dataset_yamai(args, dataset_kwargs):
    train_d, val_d, test_d = get_dataset_or_loader(
        args.dataset_class, args.dataset_name, args.data_root,
        batch_size=args.batch_size, seed=args.seed, num_splits=args.data_num_splits,
        **dataset_kwargs,
    )
    return train_d

def get_data_yamai(graph: YamaiGraph):
    num_pad_nodes = GRLC_NUM_NODES - graph.num_nodes
    x = F.pad(graph.x, (0, 3 * num_pad_nodes, 0, num_pad_nodes)).float().to(GRLC_DEVICE)
    #print(f'{x.shape=}')
    train_d = TGData(x=x, edge_index=graph.edge_index)
    eps = 2.2204e-16
    norm = train_d.x.norm(p=1, dim=1, keepdim=True).clamp(min=0.) + eps
    train_d.x = train_d.x.div(norm.expand_as(train_d.x))

    i = train_d.edge_index.long().to(GRLC_DEVICE)
    v = torch.ones([train_d.num_edges]).float().to(GRLC_DEVICE)

    A_sp = torch.sparse_coo_tensor(i, v, torch.Size([train_d.num_nodes, train_d.num_nodes]))
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


def run_GCN_yamai(args, gpu_id=None, exp_id=None):
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    final_acc = 0
    best_acc = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    running_device = "cpu" if gpu_id is None \
        else torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    dataset_kwargs = {}

    train_dataloader: EasyDataLoader[Yamai] = get_dataset_yamai(args, dataset_kwargs)
    logfile = open(f"{CKPT_PATH}/{exp_id}/train.log", "w", encoding="utf-8", newline="\n")

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
    num_neg = 10  # TODO
    print(num_neg)
    #print(f'{args.w_loss1=}, {args.w_loss2=}')

    for current_iter, epoch in enumerate(tqdm(range(args.epochs))):
        # print(f"!0, {epoch=}")
        if epoch % 50 == 0:
            torch.save(model, f"{CKPT_PATH}/{exp_id}/grlc_{epoch}.pth")
            logfile.flush()
        for data_idx, data in enumerate(train_dataloader):
            # TODO: Multibatch
            yamai = Yamai.from_batch(data)
            # print(f"!1, {yamai}")
            for graph_idx, graph in enumerate(yamai.graphs):
                #if graph_idx >= 2:
                #    exit()  # TODO
                #print()
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
                feature_X = x_list[0].to(running_device)
                lbl_z = torch.tensor([0.]).to(running_device)
                feature_a = feature_X
                feature_p = feature_X
                feature_n = []
                for i in range(num_neg):
                    idx_0 = np.random.permutation(nb_nodes)
                    feature_temp = feature_X[idx_0]
                    feature_n.append(feature_temp)
                #print(f'{feature_n=}')
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
                #print(f'{cos_0_stack=}')
                check_nan(cos_0_stack, 'cos_0_stack nan')
                cos_0_min = cos_0_stack.min(dim=0)[0]
                cos_0_max = cos_0_stack.max(dim=0)[0]
                #print(f'{cos_0_min=}, {cos_0_max=}')
                gap = cos_0_max - cos_0_min  # FIXME: This contains zero.
                #print(f'{gap=}')
                weight_list = []
                for i in range(cos_0_stack.size()[0]):
                    weight = (cos_0_stack[i] - cos_0_min) / gap
                    if torch.isnan(weight).any():
                        # FIXME: Every weight contains NaN.
                        #print('!!!', i, cos_0_stack[i], cos_0_min, gap)
                        weight = torch.nan_to_num(weight, nan=0.0)
                        #print('@@@', torch.isnan(weight).any())
                    weight_list.append(weight)
                weight_list_stk = torch.stack(weight_list)
                #print(f'{weight_list_stk=}')
                check_nan(weight_list_stk, 'weight_list_stk nan')
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
                logfile.write(string_1+'\n')
    torch.save(model, f"{CKPT_PATH}/{exp_id}/grlc_{args.epochs}.pth")
    logfile.close()

if __name__ == '__main__':
    num_total_runs = 10
    main_args = get_args(
        model_name="GRLC",
        dataset_class="Planetoid",
        dataset_name="Cora",
        custom_key="classification",
    )
    pprint_args(main_args)
    
    if not os.path.exists(CKPT_PATH):
        os.makedirs(CKPT_PATH)

    filePath = CKPT_PATH
    exp_ID = 0
    for filename in os.listdir(filePath):
        file_info = filename.split("_")
        file_dataname = file_info[0]
        if file_dataname == main_args.dataset_name:
            exp_ID = max(int(file_info[1]), exp_ID)
    exp_dir = os.path.join(CKPT_PATH, str(exp_ID))
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

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

    run_GCN_yamai(main_args, gpu_id=alloc_gpu[0], exp_id=exp_ID)
