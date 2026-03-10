# -*- coding: UTF-8 -*-
from torch import nn
from backbone.dinov2_sage import DINOv2
from aggregators.SAGE import SoftP

class SAGE(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        backbone_arch = 'dinov2_vitl14'
        # backbone_arch = 'dinov2_vitb14'
        backbone_config = {
            'norm_layer': True,
            'num_recalib_blocks': 4,
            'num_trainable_blocks': 0,
            'recalibration': 'dpn_s1',
            'return_token': True
        }
        self.backbone = DINOv2(model_name=backbone_arch, **backbone_config)
        agg_config = {
            'num_channels': 1024,
            # 'num_channels': 768,
            'num_clusters': 64,
            'cluster_dim': 128,
            'token_dim': 256,
            'bilinear': True,
            'singlebranch_mid_dim': 512,
            'singlebranch_feature_dim': 192,
            'singlebranch_split_dim': 128,
            'remove_mean': True,
            'constant_norm': 'softmax',
            'post_norm': 'dpn',
            'with_token': True,
            'final_norm': True
        }

        self.crossimage_encoder = args.crossimage_encoder
        if self.crossimage_encoder:
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16, dim_feedforward=1024, activation="gelu", dropout=0.1, batch_first=False)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.aggregator = SoftP(**agg_config)

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)

        if self.crossimage_encoder:
            B, D = x.shape[0], 768
            x = self.encoder(x.view(B, -1, D)).view(B, -1)

        return x