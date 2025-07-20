# GRLC Study
The original repository is [LarryUESTC/GRLC](https://github.com/LarryUESTC/GRLC).

## 运行原仓库代码
以下修改过程基于原仓库，仅供参考。本仓库已合并代码相关改动。

1. 执行`conda create -n grlc python=3.9`
2. 执行`conda activate grlc`
3. 进入项目工作目录`GRLC`下
4. 删除`requirements.txt`中所有`torch`开头的包
5. 删除`requirements.txt`中所有指定版本的`==X.Y.Z`字符串
6. 向`requirements.txt`末尾追加
  ```
  tensorboardx
  ruamel.yaml
  ```
7. 执行`pip install -r requirements.txt`
8. 手动安装`torch`相关包，网页链接参考
  ```
  https://download.pytorch.org/whl/torch_stable.html
  https://pytorch-geometric.com/whl/torch-2.3.1%2Bcu118.html
  ```
  文件包括
  ```
  pyg_lib-0.4.0+pt23cu118-cp39-cp39-win_amd64.whl
  torch-2.3.1+cu118-cp39-cp39-win_amd64.whl
  torchaudio-2.3.1+cu118-cp39-cp39-win_amd64.whl
  torchvision-0.18.1+cu118-cp39-cp39-win_amd64.whl
  torch_cluster-1.6.3+pt23cu118-cp39-cp39-win_amd64.whl
  torch_scatter-2.1.2+pt23cu118-cp39-cp39-win_amd64.whl
  torch_sparse-0.6.18+pt23cu118-cp39-cp39-win_amd64.whl
  torch_spline_conv-1.2.2+pt23cu118-cp39-cp39-win_amd64.whl
  ```
9. 执行`pip install torch-geometric`
10. 将`data_unit/data_sampler.py`中第14行修改为
  ```
  from torch_geometric.sampler import NeighborSampler
  ```
11. 调整`args.yaml`中GPU相关参数，注意这些参数可能多次出现，数据集特定参数会替换默认参数，包括
  ```
  black_list: []
  num_gpus_total: 1
  num_gpus_to_use: 1
  ```
12. 新建`log`文件夹
13. 执行`python train.py --dataset-class Planetoid --dataset-name Cora --custom-key classification`

Below is the original README.

## GRLC: Graph Representation Learning with Constraints
This repository contains the reference code for the paper Graph Representation Learning with Constraints (TNNLS submission).

## Contents

0. [Installation](#installation)
0. [Preparation](#Preparation)
0. [Basics](#Basics)
0. [GPU Setting](#GPU Setting)

## Installation
pip install -r requirements.txt 

## Preparation


Dataset (`--dataset-class`, `--dataset-name`,`--Custom-key`)

| Dataset class   | Dataset name        |Custom key    |
|-----------------|---------------------|--------------|
| Planetoid              | Cora         |classification|
| Planetoid              | CiteSeer     |classification|
| Planetoid              | PubMed       |classification|
| WikiCS                 | WikiCS       |classification|
| MyAmazon               | Photo        |classification|
| MyCitationFull         | CoraFull     |classification|
| MyCitationFull         | DBLP         |classification|
| Crocodile              | Crocodile    |classification|
| PygNodePropPredDataset | ogbn-arxiv   |classification|
| PygNodePropPredDataset | ogbn-mag     |classification|
| PygNodePropPredDataset | ogbn-products|classification|

Important args:
* `--usepretraining` Test checkpoints
* `--dataset-class` Planetoid, MyAmazon, WikiCS, MyCitationFull, Crocodile, PygNodePropPredDataset
* `--dataset-name` Cora, CiteSeer, PubMed, Photo, WikiCS, CoraFull, DBLP, Crocodile, ogbn-arxiv, ogbn-mag, ogbn-products
* `--custom_key` classification, link, clu

## Basics
- The main train/test code is in `Code_GRLC/train.py`.
- If you want to see the UGRL layer in PyTorch Geometric `MessagePassing` grammar, refer to `Code_GRLC/layers`.


## GPU Setting

There are three arguments for GPU settings (`--num-gpus-total`, `--num-gpus-to-use`, `--black-list`).
Default values are from the author's machine, so we recommend you modify these values from `GRLC/args.yaml` or by the command line.
- `--num-gpus-total` (default 4): The total number of GPUs in your machine.
- `--num-gpus-to-use` (default 1): The number of GPUs you want to use.
- `--black-list` (default: [1, 2, 3]): The ids of GPUs you want to not use.

