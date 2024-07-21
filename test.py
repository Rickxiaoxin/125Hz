import torch
from torch.utils.data import DataLoader
import numpy as np
from Embedding import Normalization
from tqdm import tqdm
from data_loader import DataSets, StratifiedSampler
from convolution import Convolution
from train import Model
from ret import ReT


def predict(model, test_iter, device="cuda"):
    """预测"""
    model.to(device)
    accuracy = 0
    outputs = []
    model.eval()
    for data, ecg_feature, eog_feature, label in tqdm(test_iter):
        data = data.to(device)
        label = label.type(torch.LongTensor).to(device)
        ecg_feature = ecg_feature.to(device)
        eog_feature = eog_feature.to(device)

        pred = model(data, ecg_feature, eog_feature)

        output = pred.argmax(dim=1)
        acc = (output == label).float().mean()
        outputs.extend(output.cpu().detach().numpy())
        accuracy += acc / len(test_iter)
    print(f"acc {accuracy:.3f}")
    np.save("./output.npy", np.array(outputs))


if __name__ == "__main__":
    labels = np.load("./signal/labels-4.npy")
    data = np.load("./signal/data_norm.npy")
    ecg_feature = np.load("./signal/ecg.npy")
    eog_feature = np.load("./signal/eog.npy")
    data = DataSets(data, ecg_feature, eog_feature, labels)
    del ecg_feature, eog_feature

    # stratified_sampler = StratifiedSampler(labels)
    # train_set, val_set = stratified_sampler.get_split(data, train_ratio=0.8)
    valid_iter = DataLoader(
        data,
        batch_size=32,
    )
    conv = Convolution(
        dims=[8, 16],
        num_blocks=[1, 1],
        downsample_ratio=[5, 2],
        ffn_ratio=1,
        larges_kernel=[15, 15],
        small_kernel=[5, 5],
        block_dropout=0.3,
        class_dropout=0.3,
        patch_size=15,
        patch_stride=15,
    )
    trans = ReT(
        in_channels=128,
        num_classes=4,
        dim=256,
        kernels=3,
        strides=2,
        heads=16,
        depth=1,
        dropout=0.3,
    )
    model = Model(conv, trans)
    model.load_state_dict(torch.load("./checkpoint87.3.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predict(model, valid_iter, device=device)
