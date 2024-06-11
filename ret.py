from einops import repeat
from einops.layers.torch import Rearrange
from module import ConvAttention, PreNorm, FeedForward
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Residuals(nn.Module):
    def __init__(
        self, input_channels, num_channels, use_one_multiply_one_conv=False, strides=1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_channels, num_channels, kernel_size=3, padding=1, stride=strides
        )
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_one_multiply_one_conv:
            self.conv3 = nn.Conv2d(
                input_channels, num_channels, kernel_size=1, stride=strides
            )
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        X1 = F.relu(self.bn1(self.conv1(x)))
        # print(f"X1.shape:{X1.shape}")
        X1 = self.bn2(self.conv2(X1))
        # print(f"X1_1.shape:{X1.shape}")
        if self.conv3:
            x = self.conv3(x)
        # print(f"X.shape:{X.shape}")
        X1 += x
        # print(f"output.shape:{X1.shape}")
        return F.relu(X1)


def res_block(input_channels, num_channels, num_residuals, strides, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residuals(
                    input_channels,
                    num_channels,
                    use_one_multiply_one_conv=True,
                    strides=strides,
                )
            )
        else:
            blk.append(Residuals(num_channels, num_channels))
    return blk


def simplied_featrues_CNN():
    block1 = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(1, 5), stride=(1, 5)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(
            kernel_size=(1, 3),
            stride=(1, 3),
        ),
    )
    block2 = nn.Sequential(*res_block(32, 64, 2, 2))
    # blk3 = nn.Sequential(*res_block(32, 64, 2, (1, 2)))
    net = nn.Sequential(block1, block2)
    return net


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, last_stage=False
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


class ReT(nn.Module):
    def __init__(
        self,
        in_channels=128,
        num_classes=5,
        dim=256,
        kernels=3,
        strides=2,
        heads=16,
        depth=1,
        pool="cls",
        dropout=0.1,
        scale_dim=2,
    ):
        super().__init__()

        self.conv_1 = simplied_featrues_CNN()
        self.conv_2 = simplied_featrues_CNN()
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.pool = pool
        self.dim = dim

        # in_channels = dim
        # scale = heads[2] // heads[1]
        # dim = scale * dim
        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernels, strides, padding=1),
            Rearrange("b c h w -> b (h w) c"),
            nn.LayerNorm(dim),
        )
        self.transformer = nn.Sequential(
            Transformer(
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=self.dim,
                mlp_dim=dim * scale_dim,
                dropout=dropout,
                last_stage=True,
            ),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, X_ecg, X_eog):
        X_ecg = self.conv_1(X_ecg)
        X_eog = self.conv_2(X_eog)
        x = torch.cat([X_ecg, X_eog], dim=1)
        x = self.dropout0(x)
        x = self.conv_embed(x)
        x = self.dropout1(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "h n d -> (h b) n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.mlp_head(x)
        return x


if __name__ == "__main__":
    img0 = torch.ones([2, 1, 50, 750])
    img1 = torch.ones([2, 1, 50, 750])
    model = ReT()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    out = model(img0, img1)
    print(out)

    print("Shape of out :", out.shape)  # [B, num_classes]
