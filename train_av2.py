from typing import Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch import Tensor
from torch.optim import Optimizer

from data_unit.data_av2 import YamaiDataModule, YamaiGraph
from models import GRLC

MAX_NUM_AGENTS = 135
TIME_SPAN = 10
GRLC_NUM_NODES = MAX_NUM_AGENTS * TIME_SPAN
GRLC_NUM_FEATURES = 4 + 3 * GRLC_NUM_NODES
X_NORM_EPS = 2.2204e-16

def natsumi_normalize_graph(structure: Tensor) -> Tensor:
    deg_inv_sqrt = (structure.sum(dim=-1).clamp(min=0.0) + X_NORM_EPS).pow(-0.5)
    return deg_inv_sqrt.unsqueeze(-1) * structure * deg_inv_sqrt.unsqueeze(-2)


class Natsumi(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        hidden_mul: int = 2,
        dropout: float = 0.2,
        is_neg_emb_structure: bool = True,
        lr: float = 0.005,
        weight_decay: float = 0.0001,
        num_neg_samples: int = 10,
        w_loss1: float = 2.0,
        w_loss2: float = 0.001,
        margin1: float = 0.8,
        margin2: float = 0.2,
    ) -> None:
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_neg_samples = num_neg_samples
        self.w_loss1 = w_loss1
        self.w_loss2 = w_loss2
        self.grlc = GRLC(num_features, hidden_dim, hidden_mul, dropout, is_neg_emb_structure)
        self.margin_loss = nn.MarginRankingLoss(margin=margin1, reduce=False)
        self.mask_margin = margin1 + margin2
        self.cnt_good_weights = 0

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.grlc.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, graph: YamaiGraph) -> Tuple[Tensor, Tensor, List[Tensor], Tensor, Tensor, List[Tensor]]:
        num_nodes = graph.num_nodes
        num_edges = graph.num_edges
        num_pad_nodes = GRLC_NUM_NODES - num_nodes

        # Only pad feature dimension, since node dimension is dynamic
        features = F.pad(graph.x, (0, 3 * num_pad_nodes, 0, 0))
        device = features.device

        features /= (features.norm(p=1, dim=1, keepdim=True).clamp(min=0.0) + X_NORM_EPS).expand_as(features)
        structure = torch.sparse_coo_tensor(graph.edge_index, torch.ones([num_edges], device=device), [num_nodes, num_nodes]).to_dense()
        identity = torch.eye(num_nodes, device=device)
        structure = natsumi_normalize_graph(structure + identity)

        negative_feature_samples = [features[np.random.permutation(num_nodes)] for _ in range(self.num_neg_samples)]
        return self.grlc(features, negative_feature_samples, structure, identity)

    def training_step(self, batch: Dict[str, Any]) -> Tensor:
        graph = YamaiGraph.from_batch(batch)
        anc_embs, pos_embs, neg_embs_samples, anc_embs_aug, _, neg_embs_aug_samples = self(graph)
        device = anc_embs.device
        pos_sim = F.pairwise_distance(anc_embs, pos_embs)
        neg_aug_sims = [F.pairwise_distance(anc_embs_aug, neg_embs_aug) for neg_embs_aug in neg_embs_aug_samples]

        neg_aug_sims = torch.stack(neg_aug_sims).detach()
        neg_aug_sim_min = neg_aug_sims.min(dim=0).values
        neg_aug_sim_max = neg_aug_sims.max(dim=0).values
        neg_aug_sim_gap = neg_aug_sim_max - neg_aug_sim_min  # FIXME: This contains zero.

        neg_weights = []
        pass_corr_negw_upd = True
        for i in range(self.num_neg_samples):
            neg_weight = (neg_aug_sims[i] - neg_aug_sim_min) / neg_aug_sim_gap
            if torch.isnan(neg_weight).any():
                pass_corr_negw_upd = False
                # FIXME: Every weight contains NaN.
                neg_weight = torch.nan_to_num(neg_weight, nan=0.0)
            neg_weights.append(neg_weight)
        if pass_corr_negw_upd is True:
            self.cnt_good_weights += 1

        neg_sims = [F.pairwise_distance(anc_embs, neg_embs) for neg_embs in neg_embs_samples]

        margin_label = torch.full_like(pos_sim, -1)
        loss1 = torch.tensor(0.0, device=device)
        loss2 = torch.tensor(0.0, device=device)
        zero_label = torch.tensor([0.0], device=device)
        for neg_sim, neg_weight in zip(neg_sims, neg_weights):
            loss1 += (self.margin_loss(pos_sim, neg_sim, margin_label) * neg_weight).mean()
            loss2 += torch.max((neg_sim - pos_sim.detach() - self.mask_margin), zero_label).sum()
        loss2 /= self.num_neg_samples
        loss = loss1 * self.w_loss1 + loss2 * self.w_loss2

        self.log('loss', loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('loss1', loss1, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('loss2', loss2, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        return loss


if __name__ == '__main__':
    pl.seed_everything(2025, workers=True)
    model = Natsumi(num_features=GRLC_NUM_FEATURES, hidden_dim=128, hidden_mul=2)
    datamodule = YamaiDataModule('./data_av2', 1, 1, 1)
    model_checkpoint = ModelCheckpoint(monitor='loss', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    strategy = DDPStrategy(process_group_backend='gloo', find_unused_parameters=False, gradient_as_bucket_view=True)
    trainer = pl.Trainer(
        accelerator='auto',
        devices=[0],
        strategy=strategy,
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=1000,
    )
    trainer.fit(model, datamodule=datamodule) # type: ignore
