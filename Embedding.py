import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from einops import rearrange
import math


def Normalization(data):
    scaler = StandardScaler()
    # data_norm = torch.empty((data.shaself.pe[0], data.shaself.pe[1], data.shaself.pe[2]))
    for i in range(data.shape[0]):
        data[i, 0, :] = scaler.fit_transform(data[i, 0, :].reshape(-1, 1)).reshape(
            1, -1
        )
        data[i, 1, :] = scaler.fit_transform(data[i, 1, :].reshape(-1, 1)).reshape(
            1, -1
        )
    return data


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        self.pe = torch.zeros(max_len, d_model).float()
        self.pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = self.pe.unsqueeze(0).to(device="cuda")

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# class spectrum(nn.Module):
#     def __init__(
#         self,
#     ) -> None:
#         super().__init__()

#         self.window = signal.windows.hann(50, sym=False)
#         f, t,


class Embedding(nn.Module):
    def __init__(
        self,
        step_size=250,
        num_var=2,
        kernel_size=3,
        stride=1,
        d_model=512,
        length_point=7500,
        dropout=0.0,
    ) -> None:
        super().__init__()

        self.step_size = step_size
        self.num_var = num_var
        self.time_step = length_point // self.step_size
        self.embedding = nn.Linear(self.step_size, d_model)
        self.conv0 = nn.Conv1d(
            1, 1, kernel_size, stride=stride, padding=(kernel_size - stride) // 2
        )
        self.conv1 = nn.Conv1d(
            in_channels=self.num_var * self.time_step,
            out_channels=self.num_var * self.time_step,
            kernel_size=3,
            padding=1,
            groups=self.num_var,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.num_var * self.time_step,
            out_channels=self.num_var * self.time_step,
            kernel_size=3,
            padding=1,
            groups=self.time_step,
        )
        self.conv3 = nn.Conv1d(
            in_channels=num_var,
            out_channels=1,
            kernel_size=1,
            # padding=1,
        )

    def forward(self, x):
        x = x.reshape(-1, self.num_var, self.step_size, self.time_step).permute(
            0, 1, 3, 2
        )  # (batch_size, self.num_var, self.time_step, self.step_size)
        x = self.embedding(x)
        x = rearrange(
            x, "b n (l 1) s -> (b n l) 1 s"
        )  # (batch_size*self.num_var*self.time_step, 1, self.step_size)

        # temporal feature
        x = self.conv0(x)  # Temporal feature
        x = rearrange(
            x,
            "(b n l) c s -> b n (l c) s",
            n=self.num_var,
            l=self.time_step,
        )  # batch_size,self.num_var,self.time_step,self.step_size

        # long-time dimension
        x = rearrange(
            x, "b n l s -> b (n l) s"
        )  # batch_size,self.num_var*self.time_step,self.step_size
        x = self.conv1(x)

        # varaite dimension
        x = rearrange(x, "b (n l) s -> b (l n) s", n=self.num_var)
        x = self.conv2(x)
        x = rearrange(x, "b (l n) s -> (b l) n s", n=self.num_var)
        x = self.conv3(x)
        x = rearrange(x, "(b l) n s -> b l (n s)", l=self.time_step)

        return x


if __name__ == "__main__":
    x = torch.randn((1, 2, 7500))
    x = Embedding()(x)
    print(x.shape)
