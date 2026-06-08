import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


class ResidualBlock(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class AutoEncoder(nn.Module):

    def __init__(
            self,
            input_dim: int,
            feature_dim: int,
            dims: List[int],
            dynamic_ib: bool = False
    ):
        super().__init__()
        self.dynamic_ib = dynamic_ib
        self.feature_dim = feature_dim

        layers = []
        prev = input_dim
        for h in dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(0.2),
                # ResidualBlock(h),
            ]
            prev = h

        if not dynamic_ib:
            layers += [nn.Linear(prev, feature_dim), nn.ReLU()]
            self.encoder = nn.Sequential(*layers)
        else:
            self.encoder = nn.Sequential(*layers)
            self.mu_layer = nn.Linear(prev, feature_dim)
            self.logvar_layer = nn.Linear(prev, feature_dim)

    def forward(self, x: Tensor):
        x = self.encoder(x)
        if self.dynamic_ib:
            mu = self.mu_layer(x)
            logvar = self.logvar_layer(x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z, mu, logvar
        else:
            return x


class AutoDecoder(nn.Module):

    def __init__(self, input_dim: int, feature_dim: int, dims: List[int]):
        super().__init__()
        self.decoder = nn.Sequential()
        rev = list(reversed(dims))
        prev = feature_dim
        for i, h in enumerate(rev):
            self.decoder.add_module(f"Linear{i}", nn.Linear(prev, h))
            self.decoder.add_module(f"ReLU{i}", nn.ReLU())
            self.decoder.add_module(f"BN{i}", nn.BatchNorm1d(h))
            self.decoder.add_module(f"Drop{i}", nn.Dropout(0.5))
            self.decoder.add_module(f"Res{i}", ResidualBlock(h))
            prev = h
        self.decoder.add_module("Linear_out", nn.Linear(prev, input_dim))
        self.decoder.add_module("ReLU_out", nn.ReLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)


class DS_Fusion(nn.Module):
    def __init__(self, num_clusters: int):
        super(DS_Fusion, self).__init__()
        self.num_clusters = num_clusters

    def ds_combine(self, alpha1: Tensor, alpha2: Tensor) -> Tensor:
        K = self.num_clusters
        e1, e2 = alpha1 - 1.0, alpha2 - 1.0
        S1, S2 = torch.sum(e1, dim=1, keepdim=True) + K, torch.sum(e2, dim=1, keepdim=True) + K

        b1, u1 = e1 / S1, K / S1
        b2, u2 = e2 / S2, K / S2

        C = (1 - u1) * (1 - u2) - torch.sum(b1 * b2, dim=1, keepdim=True)

        denominator = 1 - C + 1e-8
        b_fused = (b1 * b2 + b1 * u2 + b2 * u1) / denominator
        u_fused = (u1 * u2) / denominator
        S_fused = K / u_fused
        e_fused = b_fused * S_fused
        alpha_fused = e_fused + 1.0

        return alpha_fused

    def forward(self, alpha_list: List[Tensor]) -> Tensor:
        alpha_fused = alpha_list[0]
        for v in range(1, len(alpha_list)):
            alpha_fused = self.ds_combine(alpha_fused, alpha_list[v])
        return alpha_fused


class AD_MVC(nn.Module):
    def __init__(
            self,
            num_views: int,
            input_sizes: List[int],
            dims: List[int],
            dim_high_feature: int,
            dim_low_feature: int,
            num_clusters: int,
            teacher_index: int = 0
    ):
        super().__init__()
        self.num_views = num_views
        self.teacher_index = teacher_index
        self.num_clusters = num_clusters

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for v in range(num_views):
            dynamic_ib = (v != teacher_index)
            self.encoders.append(
                AutoEncoder(input_sizes[v], dim_high_feature, dims, dynamic_ib)
            )
            self.decoders.append(
                AutoDecoder(input_sizes[v], dim_high_feature, dims)
            )

        self.logit_layer_v = nn.ModuleList([
            nn.Linear(dim_high_feature, num_clusters)
            for _ in range(num_views)
        ])

        self.fusion_layer = DS_Fusion(num_clusters)

    @torch.no_grad()
    def set_teacher(self, new_teacher_idx: int):
        if new_teacher_idx == self.teacher_index:
            return
        self.teacher_index = new_teacher_idx
        device = next(self.parameters()).device

        for v, enc in enumerate(self.encoders):
            if v == new_teacher_idx and enc.dynamic_ib:
                enc.dynamic_ib = False
                del enc.mu_layer
                del enc.logvar_layer
                last_dim = None
                for m in reversed(enc.encoder):
                    if isinstance(m, nn.Linear):
                        last_dim = m.out_features
                        break
                if last_dim != enc.feature_dim:
                    enc.encoder.add_module("to_feat",
                                           nn.Linear(last_dim, enc.feature_dim).to(device))
                    enc.encoder.add_module("relu_feat", nn.ReLU())
            elif v != new_teacher_idx and not enc.dynamic_ib:
                enc.dynamic_ib = True
                while isinstance(enc.encoder[-1], (nn.ReLU, nn.Linear)):
                    enc.encoder = enc.encoder[:-1]
                last_dim = None
                for m in reversed(enc.encoder):
                    if isinstance(m, nn.Linear):
                        last_dim = m.out_features
                        break
                enc.mu_layer = nn.Linear(last_dim, enc.feature_dim).to(device)
                enc.logvar_layer = nn.Linear(last_dim, enc.feature_dim).to(device)

    def forward(self, data_views: List[Tensor]):
        recons, features = [], []
        kl_loss_total = 0.0

        for v, x in enumerate(data_views):
            out = self.encoders[v](x)
            if isinstance(out, tuple):
                z, mu, logvar = out
                kl_loss_total += -0.5 * torch.mean(
                    (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
                )
            else:
                z = out
            recons.append(self.decoders[v](z))
            features.append(z)

        assert len(features) == self.num_views

        evidence_list = [F.softplus(self.logit_layer_v[v](features[v])) for v in range(self.num_views)]

        alpha_list = [e + 1.0 for e in evidence_list]

        alpha_fused = self.fusion_layer(alpha_list)

        fused_prob = alpha_fused / torch.sum(alpha_fused, dim=1, keepdim=True)

        return alpha_list, recons, features, kl_loss_total, alpha_fused, fused_prob