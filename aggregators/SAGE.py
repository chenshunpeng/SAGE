# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from prettytable import PrettyTable

class DPN(nn.Module):
    def __init__(self, num_channels=128, clamp=False, eps=1e-6):
        super(DPN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(num_channels // 2, 1)
        )
        self.act = nn.Sigmoid()
        self.eps = eps
        self.clamp = clamp

    def forward(self, x):
        y = self.avg_pool(x.transpose(-1, -2)).squeeze(-1)
        y = self.projection(y)
        p = self.act(y).unsqueeze(-1)
        _, L, D = x.shape
        if self.clamp:
            return x.clamp(min=self.eps).pow(p.expand(-1, -1, D))
        else:
            sign = torch.sign(x)
            pow = torch.pow(torch.abs(x) + self.eps, p.expand(-1, L, D))
            return sign * pow + x


class SoftWeighting(nn.Module):
    """
    Soft Weighting module for Soft Probing (SoftP):
    Applies a data-driven residual weighting to adaptively emphasize
    informative spatial locations based on their l2 response.
    """
    def __init__(self, hidden_dim=128, alpha=0.5, eps=1e-6):
        super(SoftWeighting, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.predictor = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        descriptors = x.view(B, C, -1).transpose(1, 2)
        l2_response = torch.norm(descriptors, p=2, dim=-1, keepdim=True) + self.eps
        beta = self.predictor(l2_response) * self.alpha
        beta = beta.expand(-1, -1, C)
        modulated_descriptors = descriptors + beta * descriptors
        out = modulated_descriptors.transpose(1, 2).view(B, C, H, W)
        return out

class SoftP(nn.Module):
    """
    Attributes:
        num_channels (int): The number of channels of the inputs (d).
        num_clusters (int): The number of clusters in the model (m).
        cluster_dim (int): The number of channels of the clusters (l).
        token_dim (int): The dimension of the global scene token (g).
        dropout (float): The dropout rate.
        bilinear (bool): True->bilinear, False->singlebranch
        remove_mean (bool):
        constant_norm (str): softmax; sigmoid; none
        post_norm (str): SqrtColL2; NSqrtColL2; SqrtFlattenL2; NSqrtFlattenL2; none
        with_token (bool):
        final_norm (bool):
    """

    def __init__(self,
                 num_channels=768,
                 num_clusters=64,
                 cluster_dim=128,
                 token_dim=256,
                 dropout=0.3,
                 bilinear=True,
                 singlebranch_mid_dim=512,
                 singlebranch_feature_dim=192,
                 singlebranch_split_dim=128,
                 remove_mean=False,
                 constant_norm='softmax',
                 post_norm='SqrtColL2',
                 with_token=False,
                 final_norm=False,
                 use_soft_weighting=True,
                 weighting_hidden_dim=128,
                 weighting_alpha=0.5
                 ) -> None:
        super().__init__()

        self.use_soft_weighting = use_soft_weighting
        if self.use_soft_weighting:
            self.soft_weighting = SoftWeighting(hidden_dim=weighting_hidden_dim, alpha=weighting_alpha)
        else:
            self.soft_weighting = None

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        self.bilinear = bilinear
        self.singlebranch_mid_dim = singlebranch_mid_dim
        self.singlebranch_feature_dim = singlebranch_feature_dim
        self.singlebranch_split_dim = singlebranch_split_dim
        self.remove_mean = remove_mean
        self.constant_norm = constant_norm
        self.post_norm = post_norm
        self.with_token = with_token
        self.final_norm = final_norm

        self.printconfig()

        # MLP for global scene token g
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim)
        )
        # MLP for local features f_i
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1)
        )
        # MLP for score matrix S
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

        self.single_branch = nn.Sequential(
            nn.Conv2d(self.num_channels, self.singlebranch_mid_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.singlebranch_mid_dim, self.singlebranch_feature_dim, 1),
        )

        if 'dpn' == self.post_norm:
            self.gpn = DPN(num_channels=num_clusters)

    def printconfig(self):
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["num_channels", f"{self.num_channels}"])
        table.add_row(["num_clusters", f"{self.num_clusters}"])
        table.add_row(["cluster_dim", f"{self.cluster_dim}"])
        table.add_row(["token_dim", f"{self.token_dim}"])
        table.add_row(["bilinear", f"{self.bilinear}"])
        table.add_row(["singlebranch_mid_dim", f"{self.singlebranch_mid_dim}"])
        table.add_row(["singlebranch_feature_dim", f"{self.singlebranch_feature_dim}"])
        table.add_row(["singlebranch_split_dim", f"{self.singlebranch_split_dim}"])
        table.add_row(["remove_mean", f"{self.remove_mean}"])
        table.add_row(["constant_norm", f"{self.constant_norm}"])
        table.add_row(["post_norm", f"{self.post_norm}"])
        table.add_row(["with_token", f"{self.with_token}"])
        table.add_row(["final_norm", f"{self.final_norm}"])
        print(table.get_string(title="SoftP config"))

    def removemean(self, input):
        mean = nn.functional.adaptive_avg_pool2d(input, (1, 1))
        output = input - mean
        return output

    def bilinearbranch(self, x):
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        if self.remove_mean:
            f = self.removemean(f)
            p = self.removemean(p)

        return f, p

    def singlebranch(self, x):
        f = self.single_branch(x).flatten(2)  # B,D,16,16 -> B,D,256
        if self.remove_mean:
            f = self.removemean(f)

        assert (self.singlebranch_split_dim < self.singlebranch_feature_dim)

        f1 = f[:, :self.singlebranch_split_dim, :]
        f2 = f[:, self.singlebranch_split_dim:, :]

        return f1, f2

    def branch(self, x):
        if self.bilinear:
            return self.bilinearbranch(x)

        # else: single branch
        return self.singlebranch(x)

    def constantnorm(self, p):
        if self.constant_norm == 'softmax':
            p_constantnorm = nn.functional.softmax(p, dim=2)
        elif self.constant_norm == 'sigmoid':
            p_constantnorm = nn.functional.sigmoid(p)
            p_constantnorm = nn.functional.normalize(p_constantnorm, p=1, dim=2)
        elif self.constant_norm == 'none':
            p_constantnorm = p
        else:
            raise ValueError(f"Invalid constant norm: {self.constant_norm}")

        return p_constantnorm

    def bmm(self, f, p):
        if self.bilinear:
            fp = torch.bmm(f, p.transpose(1, 2))  # B,f_d,p_d
            return fp

        # else: single branch
        fp_cat = torch.cat([f, p], dim=1)

        return fp_cat.bmm(fp_cat.transpose(1, 2))  # B, fp_d fp_d

    def postnorm(self, fp, N):
        if self.post_norm == 'none':
            fp_postnorm = fp.flatten(1)
        elif self.post_norm.startswith('gpn') or self.post_norm.startswith('dpn'):  # modified or fail to run
            fp_N = (1. / N) * fp
            fp_gpn = self.gpn(fp_N)
            fp_postnorm = nn.functional.normalize(fp_gpn).flatten(1)
        else:
            raise ValueError(f"Invalid post norm: {self.post_norm}")

        return fp_postnorm

    def withtoken(self, fp, token):
        if self.with_token:
            t = self.token_features(token)
            fp_withtoken = torch.cat([
                nn.functional.normalize(t, p=2, dim=-1),
                fp
            ], dim=-1)
        else:
            fp_withtoken = fp

        return fp_withtoken

    def finalnorm(self, fp):
        # check
        if self.final_norm:
            return nn.functional.normalize(fp, p=2, dim=-1)
        return fp

    def forward(self, x):
        """
        x (tuple): A tuple containing two elements, f and t.
            (torch.Tensor): The feature tensors (t_i) [B, C, H // 14, W // 14].
            (torch.Tensor): The token tensor (t_{n+1}) [B, C].

        Returns:
            f (torch.Tensor): The global descriptor [B, m*l + g]
        """
        x, t = x  # Extract features and token
        if self.use_soft_weighting:
            x = self.soft_weighting(x)

        f, p = self.branch(x)
        _, _, N = p.shape

        p_constantnorm = self.constantnorm(p)
        fp = self.bmm(f, p_constantnorm)
        fp_postnorm = self.postnorm(fp, N)
        fp_withtoken = self.withtoken(fp_postnorm, t)

        return self.finalnorm(fp_withtoken)
