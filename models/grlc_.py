from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class XavierLinear(nn.Linear):
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0.0)


class GCN(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc = XavierLinear(in_features, out_features, bias=True)
        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, features: Tensor, structure: Tensor) -> Tensor:
        return self.relu(torch.mm(structure, self.fc(features)) + self.bias)


class GRLC(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        hidden_mul: int = 2,
        dropout: float = 0.2,
        is_neg_emb_structure: bool = True,
    ) -> None:
        super().__init__()
        self.gcn_0 = GCN(num_features, hidden_dim * hidden_mul)
        self.gcn_1 = GCN(hidden_dim * hidden_mul, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.is_neg_emb_structure = is_neg_emb_structure

    def forward(
        self,
        features: Tensor,
        negative_features_samples: List[Tensor],
        structure: Tensor,
        identity: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor], Tensor, Tensor, List[Tensor]]:
        positive_features = self.dropout(features)
        negative_features_samples = [self.dropout(negative_features) for negative_features in negative_features_samples]

        anchor_embeds_aug = self.gcn_0(features, identity)
        anchor_embeds = self.gcn_1(anchor_embeds_aug, identity)
        positive_embeds_aug = self.gcn_0(positive_features, identity)
        positive_embeds = self.gcn_1(self.dropout(positive_embeds_aug), structure)

        negative_embeds_samples = []
        negative_embeds_aug_samples = []
        for negative_features in negative_features_samples:
            negative_embeds_aug = self.gcn_0(negative_features, identity)
            negative_embeds_aug_samples.append(negative_embeds_aug)
            negative_embeds = self.gcn_1(
                self.dropout(negative_embeds_aug),
                structure if self.is_neg_emb_structure else identity
            )
            negative_embeds_samples.append(negative_embeds)

        return anchor_embeds, positive_embeds, negative_embeds_samples, anchor_embeds_aug, positive_embeds_aug, negative_embeds_aug_samples

    def embed(self, features: Tensor, structure: Tensor, identity: Tensor) -> Tuple[Tensor, Tensor]:
        anchor_embeds = self.gcn_1(self.gcn_0(features, identity), identity)
        positive_embeds = self.gcn_1(self.gcn_0(features, identity), structure)

        return anchor_embeds.detach(), positive_embeds.detach()
