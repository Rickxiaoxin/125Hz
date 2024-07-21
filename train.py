import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import numpy as np
from Embedding import Normalization
from tqdm import tqdm
import time
from data_loader import DataSets, StratifiedSampler
from convolution import Convolution
from torch.nn import functional as F
from ret import ReT
from earlystop import EarlyStopping


def fix_seed(seed=66):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def train(model, train_iter, valid_iter, lr, num_epochs, device="cuda"):
    """训练模型"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d or type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight)

    early_stop = EarlyStopping(patience=10, verbose=True)

    model.apply(init_weights)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.003)
    loss = nn.CrossEntropyLoss()
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0

        for data, ecg_feature, eog_feature, label in tqdm(train_iter):
            optimizer.zero_grad()
            data = data.to(device)
            ecg_feature = ecg_feature.to(device)
            eog_feature = eog_feature.to(device)
            label = label.type(torch.LongTensor).to(device)
            pred = model(data, ecg_feature, eog_feature)
            l = loss(pred, label)
            l.backward()
            optimizer.step()
            acc = (pred.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_iter)
            epoch_loss += l / len(train_iter)
        epoch_time = time.time() - start_time

        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, ecg_feature, eog_feature, label in tqdm(valid_iter):
                data = data.to(device)
                label = label.type(torch.LongTensor).to(device)
                ecg_feature = ecg_feature.to(device)
                eog_feature = eog_feature.to(device)

                pred = model(data, ecg_feature, eog_feature)
                val_loss = loss(pred, label)

                acc = (pred.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_iter)
                epoch_val_loss += val_loss / len(valid_iter)
        print(
            f"epoch {epoch + 1}: train loss {epoch_loss:.3f}, train acc {epoch_accuracy:.3f}, "
            f"val loss {epoch_val_loss:.3f}, val acc {epoch_val_accuracy:.3f}, time {epoch_time:.1f}s"
        )
        start_time = time.time()
        early_stop(
            epoch_val_accuracy,
            model,
        )
        if early_stop.early_stop:
            print("Early stopping")
            break


class Model(nn.Module):
    def __init__(self, conv, trans):
        super(Model, self).__init__()
        self.conv = conv
        self.trans = trans
        self.act = F.gelu
        self.fc = nn.Linear(8, 4)

    def forward(self, signal, ecg_feature, eog_feature):
        cls0 = self.conv(signal)
        cls1 = self.trans(ecg_feature, eog_feature)
        cls = torch.cat((cls0, cls1), dim=1)
        cls = self.act(cls)
        output = self.fc(cls)
        return output
        # return cls0


if __name__ == "__main__":
    fix_seed()

    labels = np.load("./signal/labels-4.npy")
    # ecg = np.load("./data/ecg.npy")
    # eog = np.load("./data/eog.npy")
    # data = np.stack((ecg, eog), axis=1)
    # del ecg, eog
    # data = Normalization(data).astype(np.float32)
    # np.save("./data/data_norm.npy", data)
    data = np.load("./signal/data_norm.npy")
    ecg_feature = np.load("./signal/ecg.npy")
    eog_feature = np.load("./signal/eog.npy")

    epochs = len(labels)
    data = DataSets(data, ecg_feature, eog_feature, labels)
    del ecg_feature, eog_feature

    stratified_sampler = StratifiedSampler(labels)

    train_set, val_set = stratified_sampler.get_split(data, 0.8)
    del data
    train_iter = DataLoader(
        train_set,
        batch_size=16,
        shuffle=True,
    )
    valid_iter = DataLoader(
        val_set,
        batch_size=32,
    )

    del train_set, val_set
    conv = Convolution(
        dims=[8, 16],
        num_blocks=[1, 1],
        downsample_ratio=[5, 5],
        ffn_ratio=2,
        larges_kernel=[9, 9],
        small_kernel=[5, 5],
        block_dropout=0.5,
        class_dropout=0.5,
        patch_size=8,
        patch_stride=8,
    )
    # trans = Transformer(
    #     dim=32,
    #     kernels=[(1, 5), (2, 5)],
    #     strides=[(1, 5), (2, 5)],
    #     heads=[8, 16],
    #     depth=[1, 2],
    #     dropout=0.7,
    # )
    trans = ReT(
        in_channels=128,
        num_classes=4,
        dim=128,
        kernels=3,
        strides=2,
        heads=16,
        depth=1,
        dropout=0.5,
    )
    model = Model(conv, trans)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(model, train_iter, valid_iter, lr=0.0005, num_epochs=100, device=device)
