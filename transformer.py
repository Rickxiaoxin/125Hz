import torch
from einops import repeat
from einops.layers.torch import Rearrange
import numpy as np
from torch.nn import functional as F
from torch import nn, einsum
from einops import rearrange


class SepConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
    ):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvAttention(nn.Module):
    def __init__(
        self,
        dim,
        image_size,
        heads=8,
        dim_head=64,
        kernel_size=3,
        q_stride=1,
        k_stride=1,
        v_stride=1,
        dropout=0.0,
        last_stage=False,
    ):

        super().__init__()
        self.last_stage = last_stage
        self.image_size = image_size

        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        pad = (kernel_size - q_stride) // 2
        self.to_q = SepConv2d(dim, dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, dim, kernel_size, v_stride, pad)

        self.to_out = (
            nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            x = x[:, 1:]
            cls_token = rearrange(cls_token.unsqueeze(1), "b n (h d) -> b h n d", h=h)
        x = rearrange(
            x, "b (l w) n -> b n l w", l=self.image_size[0], w=self.image_size[1]
        )
        q = self.to_q(x)
        q = rearrange(q, "b (h d) l w -> b h (l w) d", h=h)

        v = self.to_v(x)
        v = rearrange(v, "b (h d) l w -> b h (l w) d", h=h)

        k = self.to_k(x)
        k = rearrange(k, "b (h d) l w -> b h (l w) d", h=h)

        if self.last_stage:
            q = torch.cat((cls_token, q), dim=2)
            v = torch.cat((cls_token, v), dim=2)
            k = torch.cat((cls_token, k), dim=2)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        image_size,
        dropout=0.0,
        last_stage=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            ConvAttention(
                                dim,
                                image_size,
                                heads=heads,
                                dim_head=dim_head,
                                dropout=dropout,
                                last_stage=last_stage,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_classes=5,
        image_size=(20, 150),
        dim=32,
        kernels=[(1, 5), (2, 10)],
        strides=[(1, 5), (2, 3)],
        heads=[8, 16],
        depth=[1, 1],
        pool="cls",
        dropout=0.1,
    ):
        super().__init__()

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.pool = pool
        self.dim = dim

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernels[0],
                strides[0],
            ),
            Rearrange(
                "b c h w -> b (h w) c",
                h=image_size[0] // strides[0][0],
                w=image_size[1] // strides[0][1],
            ),
            nn.LayerNorm(dim),
        )
        self.stage1_transformer = nn.Sequential(
            transformer(
                dim=dim,
                depth=depth[0],
                heads=heads[0],
                dim_head=self.dim,
                mlp_dim=dim,
                image_size=(
                    image_size[0] // strides[0][0],
                    image_size[1] // strides[0][1],
                ),
                dropout=dropout,
            ),
            Rearrange(
                "b (h w) c -> b c h w",
                h=image_size[0] // strides[0][0],
                w=image_size[1] // strides[0][1],
            ),
        )

        in_channels = dim
        scale = heads[1] // heads[0]
        dim = scale * dim
        image_sacle_h = strides[0][0] * strides[1][0]
        image_sacle_w = strides[0][1] * strides[1][1]
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dim,
                kernels[1],
                strides[1],
            ),
            Rearrange(
                "b c h w -> b (h w) c",
                h=image_size[0] // image_sacle_h,
                w=image_size[1] // image_sacle_w,
            ),
            nn.LayerNorm(dim),
        )
        self.stage2_transformer = nn.Sequential(
            transformer(
                dim=dim,
                depth=depth[1],
                heads=heads[1],
                dim_head=self.dim,
                mlp_dim=dim,
                image_size=(
                    image_size[0] // image_sacle_h,
                    image_size[1] // image_sacle_w,
                ),
                dropout=dropout,
                last_stage=True,
            ),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout_large = nn.Dropout(emb_dropout)
        self.act = F.gelu
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, xs):

        xs = self.stage1_conv_embed(xs)
        xs = self.stage1_transformer(xs)

        xs = self.stage2_conv_embed(xs)
        b, n, _ = xs.shape
        cls_tokens = repeat(self.cls_token, "h n d -> (h b) n d", b=b)
        xs = torch.cat((cls_tokens, xs), dim=1)
        xs = self.stage2_transformer(xs)
        xs = xs[:, 0]
        xs = self.act(xs)
        xs = self.mlp_head(xs)
        return xs


if __name__ == "__main__":
    img0 = torch.ones([2, 2, 20, 150])
    model = Transformer(
        dim=32,
        kernels=[(1, 5), (2, 5)],
        strides=[(1, 5), (2, 5)],
        heads=[8, 16],
        depth=[1, 2],
        dropout=0.1,
    )

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    out = model(img0)

    print("Shape of out :", out.shape)  # [B, num_classes]
