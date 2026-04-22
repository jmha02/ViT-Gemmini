from types import MethodType
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import vision_transformer
from timm.models.vision_transformer import Attention
from timm.models.swin_transformer import WindowAttention, checkpoint_filter_fn as swin_checkpoint_filter_fn


def attention_forward(self, x, attn_mask=None, **kwargs):
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    # attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = self.matmul1(q, k.transpose(-2, -1)) * self.scale
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    # del q, k

    # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B, N, C)
    # del attn, v
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def window_attention_forward(self, x, mask=None, **kwargs):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

    q = q * self.scale
    # attn = (q @ k.transpose(-2, -1))
    attn = self.matmul1(q, k.transpose(-2,-1))

    relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    attn = attn + relative_position_bias.unsqueeze(0)

    if mask is not None:
        nW = mask.shape[0]
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    else:
        attn = self.softmax(attn)

    attn = self.attn_drop(attn)

    # x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
    x = self.matmul2(attn, v).transpose(1, 2).reshape(B_, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class MatMul(nn.Module):
    def forward(self, A, B):
        return A @ B


def build_model(
    name,
    pretrained=True,
    checkpoint_path=None,
    cache_dir=None,
    exportable=None,
):
    """
    Get a vision transformer model.
    This will replace matrix multiplication operations with matmul modules in the model.
    Currently support almost all models in timm.models.transformers, including:
    - vit_tiny/small/base/large_patch16/patch32_224/384,
    - deit_tiny/small/base(_distilled)_patch16_224,
    - deit_base(_distilled)_patch16_384,
    - swin_tiny/small/base/large_patch4_window7_224,
    - swin_base/large_patch4_window12_384
    These models are finetuned on imagenet-1k and should use ViTImageNetLoaderGenerator
    for calibration and testing.
    """
    manual_swin_checkpoint = checkpoint_path is not None and name.startswith("swin")
    model = timm.create_model(
        name,
        pretrained=pretrained and not manual_swin_checkpoint,
        checkpoint_path=None if manual_swin_checkpoint else checkpoint_path,
        cache_dir=cache_dir,
        exportable=exportable,
    )

    if manual_swin_checkpoint:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        filtered_state_dict = swin_checkpoint_filter_fn(state_dict, model)
        incompatible = model.load_state_dict(filtered_state_dict, strict=False)
        allowed_unexpected = {
            "layers.1.downsample.reduction.bias",
            "layers.2.downsample.reduction.bias",
            "layers.3.downsample.reduction.bias",
        }
        unexpected = [key for key in incompatible.unexpected_keys if key not in allowed_unexpected]
        if unexpected:
            raise RuntimeError(f"Unexpected Swin checkpoint keys: {unexpected}")

    #for module in model.modules():
    for name, module in model.named_modules():
        if isinstance(module, Attention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(attention_forward, module)
        if isinstance(module, WindowAttention):
            setattr(module, "matmul1", MatMul())
            setattr(module, "matmul2", MatMul())
            module.forward = MethodType(window_attention_forward, module)

    return model
