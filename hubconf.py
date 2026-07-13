from types import SimpleNamespace

import torch

import network


dependencies = ["torch", "prettytable"]

_MODEL_URLS = {
    "sage": "https://github.com/chenshunpeng/SAGE/releases/download/v1.0.0/SAGE.pth",
    "sage_vitb": "https://github.com/chenshunpeng/SAGE/releases/download/v1.0.0/SAGE_No-Encoder_Vit-B.pth",
    "sage_vitl": "https://github.com/chenshunpeng/SAGE/releases/download/v1.0.0/SAGE_No-Encoder_Vit-L.pth",
}


def _load_checkpoint(model, url, progress):
    checkpoint = torch.hub.load_state_dict_from_url(
        url,
        map_location="cpu",
        progress=progress,
    )
    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict, strict=True)


def _build_model(name, backbone_arch, crossimage_encoder, pretrained, progress, **kwargs):
    num_trainable_blocks = kwargs.pop("num_trainable_blocks", 0)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected model keyword argument(s): {unexpected}")

    args = SimpleNamespace(crossimage_encoder=crossimage_encoder)
    model = network.SAGE(
        args,
        backbone_arch=backbone_arch,
        num_trainable_blocks=num_trainable_blocks,
    )
    if pretrained:
        _load_checkpoint(model, _MODEL_URLS[name], progress)
    return model.eval()


def sage(pretrained=True, progress=True, **kwargs):
    """Load Full SAGE with DINOv2 ViT-L/14 and the cross-image encoder.

    Args:
        pretrained: Load the released SAGE checkpoint when True.
        progress: Show the checkpoint download progress bar when True.
        **kwargs: Optional non-structural model arguments (num_trainable_blocks).

    Returns:
        The SAGE PyTorch model in evaluation mode.
    """
    return _build_model(
        "sage", "dinov2_vitl14", True, pretrained, progress, **kwargs
    )


def sage_vitb(pretrained=True, progress=True, **kwargs):
    """Load SAGE with DINOv2 ViT-B/14 and no cross-image encoder.

    Args:
        pretrained: Load the released ViT-B checkpoint when True.
        progress: Show the checkpoint download progress bar when True.
        **kwargs: Optional non-structural model arguments (num_trainable_blocks).

    Returns:
        The SAGE PyTorch model in evaluation mode.
    """
    return _build_model(
        "sage_vitb", "dinov2_vitb14", False, pretrained, progress, **kwargs
    )


def sage_vitl(pretrained=True, progress=True, **kwargs):
    """Load SAGE with DINOv2 ViT-L/14 and no cross-image encoder.

    Args:
        pretrained: Load the released ViT-L checkpoint when True.
        progress: Show the checkpoint download progress bar when True.
        **kwargs: Optional non-structural model arguments (num_trainable_blocks).

    Returns:
        The SAGE PyTorch model in evaluation mode.
    """
    return _build_model(
        "sage_vitl", "dinov2_vitl14", False, pretrained, progress, **kwargs
    )
