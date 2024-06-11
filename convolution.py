import torch
from torch import nn
import torch.nn.functional as F


def conv_bn(
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation=1,
    bias=False,
):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        ),
    )
    result.add_module("bn", nn.BatchNorm1d(out_channels))
    return result


class ReparamLargeKernelConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        small_kernel,
    ):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel

        self.lkb_origin = conv_bn(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=1,
            bias=False,
        )
        if small_kernel is not None:
            assert (
                small_kernel <= kernel_size
            ), "The kernel size for re-param cannot be larger than the large kernel!"
            self.small_conv = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=small_kernel,
                stride=stride,
                padding=small_kernel // 2,
                dilation=1,
                bias=False,
            )

    def forward(self, inputs):

        out = self.lkb_origin(inputs)
        out += self.small_conv(inputs)
        return out


class Block(nn.Module):
    def __init__(
        self,
        large_size,
        small_size,
        dmodel,
        dff,
        nvars,
        drop=0.1,
    ):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * dmodel,
            out_channels=nvars * dmodel,
            kernel_size=large_size,
            stride=1,
            small_kernel=small_size,
        )
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(
            in_channels=nvars * dmodel,
            out_channels=nvars * dff,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=nvars,
        )
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(
            in_channels=nvars * dff,
            out_channels=nvars * dmodel,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=nvars,
        )
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(
            in_channels=nvars * dmodel,
            out_channels=nvars * dff,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=dmodel,
        )
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(
            in_channels=nvars * dff,
            out_channels=nvars * dmodel,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=dmodel,
        )
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff // dmodel

    def forward(self, x):

        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x


class Stage(nn.Module):
    def __init__(
        self,
        ffn_ratio,
        num_blocks,
        large_size,
        small_size,
        dmodel,
        nvars=2,
        small_kernel_merged=False,
        drop=0.1,
    ):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for _ in range(num_blocks):
            blk = Block(
                large_size=large_size,
                small_size=small_size,
                dmodel=dmodel,
                dff=d_ffn,
                nvars=nvars,
                drop=drop,
            )
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class Convolution(nn.Module):
    def __init__(
        self,
        dims,
        num_blocks,
        downsample_ratio,
        ffn_ratio,
        larges_kernel,
        small_kernel,
        block_dropout,
        class_dropout,
        patch_size,
        patch_stride,
    ):
        self.downsample_ratio = downsample_ratio
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        super(Convolution, self).__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0]),
        )
        self.downsample_layers.append(stem)
        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsampler = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(
                        dims[i],
                        dims[i + 1],
                        kernel_size=downsample_ratio[i],
                        stride=downsample_ratio[i],
                    ),
                )
                self.downsample_layers.append(downsampler)
        self.stages = nn.ModuleList()
        for i in range(self.num_stage):
            layer = Stage(
                ffn_ratio,
                num_blocks[i],
                larges_kernel[i],
                small_kernel[i],
                dims[i],
                drop=block_dropout,
            )
            self.stages.append(layer)

        self.act_class = F.gelu
        self.class_dropout0 = nn.Dropout(class_dropout)
        self.dense0 = nn.Linear(
            2 * dims[-1] * 3750 // patch_stride // downsample_ratio[0],
            5,
        )
        # self.class_dropout1 = nn.Dropout(class_dropout)
        # self.dense1 = nn.Linear(5072, 5)

    def forward(self, x):
        B, M, L = x.shape

        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio[i - 1] != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        x = self.act_class(x)
        x = self.class_dropout0(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dense0(x)
        # x = F.softmax(x,dim=1)
        # x = self.dense1(self.class_dropout1(x))
        return x


if __name__ == "__main__":
    conv = Convolution(
        dims=[16, 32],
        num_blocks=[1, 1],
        downsample_ratio=[5, 2],
        ffn_ratio=1,
        larges_kernel=[15, 15],
        small_kernel=[5, 5],
        block_dropout=0.1,
        class_dropout=0.5,
        patch_size=15,
        patch_stride=15,
    )
    data = torch.randn([1, 2, 3750])
    result = conv(data)
    print(result)
