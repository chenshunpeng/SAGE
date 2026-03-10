import torch
import torch.nn as nn
# from .PSRP import *
#from .adapter import Adapter
DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

class DPN(nn.Module):
    def __init__(self, num_channels=128, clamp=False, eps=1e-6):
        super(DPN, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(num_channels, 16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.act = nn.Sigmoid()
        self.eps = eps
        self.clamp = clamp
    def forward(self, x):
        y = self.avg_pool(x.transpose(-1,-2)).squeeze(-1)#B, 768,257 -> B, 768, 1 -> B,768
        y = self.projection(y)
        p = self.act(y).unsqueeze(-1)#B,1 
        _, L, D = x.shape
        if self.clamp:
            return x.clamp(min=self.eps).pow(p.expand(-1,-1,D))
        else:
            sign = torch.sign(x)
            pow = torch.pow(torch.abs(x) + self.eps, p.expand(-1,L,D))
            return sign * pow + x

class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=4,
            num_recalib_blocks=4,
            norm_layer=False,
            return_token=False,
            recalibration='none'
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'

        # Recommended: Load directly from GitHub (if offline usage is not strictly required)
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        # If your project requires offline loading, use a generic placeholder instead of an absolute path:
        # self.model = torch.hub.load('/path/to/your/local/facebookresearch_dinov2_main/', model_name, source='local')

        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.num_recalib = num_recalib_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token
        self.recalibration = recalibration
        self.num_no_reccalib = len(self.model.blocks) -  self.num_recalib
        
        self.gpn_norms = nn.ModuleList()
        if self.recalibration.startswith('dpn'):
            for _ in range(self.num_recalib):
                self.gpn_norms.append(DPN(num_channels=self.num_channels))
        else:
            pass

        # Frozen all
        for name, para in self.model.named_parameters():
            para.requires_grad = False
        if self.num_trainable_blocks > 0 and self.num_trainable_blocks <= len(self.model.blocks):
            for blk in self.model.blocks[-self.num_trainable_blocks:]:
                for name, para in blk.named_parameters():
                    para.requires_grad = True
        else:
            pass

        for name, para in self.model.named_parameters():
            if name =="norm.weight" or name == "norm.bias":
                para.requires_grad = True
            # print(name, para.requires_grad)

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)
        for idx, blk in enumerate(self.model.blocks):
            if self.recalibration == 'none' or idx < self.num_no_reccalib:
                x = blk(x)
            elif self.recalibration == 'psrp':
                x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
                weight, bias = self.psrps[idx-self.num_no_reccalib](x)
                x = x + blk.drop_path2(blk.ls2(bias + (weight + 1)*blk.mlp(blk.norm2(x))))
            elif self.recalibration == 'gpnl_s1' or self.recalibration == 'dpn_s1':
                x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
                x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
                x = self.gpn_norms[idx-self.num_no_reccalib](x)
            elif self.recalibration == 'gpnl_s2' or self.recalibration == 'dpn_s2':
                x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
                x = self.gpn_norms[idx-self.num_no_reccalib](x)
                x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            elif self.recalibration == 'gpnl_p1' or self.recalibration == 'dpn_p1':
                x1 = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
                x2 = blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x1))))
                x3 = self.gpn_norms[idx-self.num_no_reccalib](x1)
                x = x2 + x3
            elif self.recalibration == 'gpnl_p2' or self.recalibration == 'dpn_p2':
                x1 = blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
                x2 = self.gpn_norms[idx-self.num_no_reccalib](x)
                x = x1 + x2
                x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            else:
                raise ValueError(f"Invalid recalibration: {self.recalibration}")

        if self.norm_layer:
            x = self.model.norm(x)
        
        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f
