import numpy as np
from torch.utils.data import Dataset, Subset
from collections import Counter


# class DataSets(Dataset):
#     def __init__(self, signal, feature, label):
#         self.signal = signal
#         self.label = label
#         self.feature = feature

#     def __len__(self):
#         return self.label.shape[0]


#     def __getitem__(self, idx):
#         label = self.label[idx]
#         signal = self.signal[idx]
#         feature = self.feature[idx]
#         return signal, feature, label
class DataSets(Dataset):
    def __init__(self, signal, ecg_feature, eog_feature, label):
        self.signal = signal
        self.label = label
        self.ecg_feature = ecg_feature
        self.eog_feature = eog_feature

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        label = self.label[idx]
        signal = self.signal[idx]
        ecg_feature = self.ecg_feature[idx]
        eog_feature = self.eog_feature[idx]
        return signal, ecg_feature, eog_feature, label


class StratifiedSampler:
    """按标签的比例划分数据集"""

    def __init__(self, labels):
        self.labels = labels
        # self.classes, self.counts = np.unique(labels, return_counts=True)
        # self.class_indices = [np.where(labels == cls)[0] for cls in self.classes]

    def _get_split_indices(self, fold):
        label_counts = Counter(self.labels)
        train_indices = []
        valid_indices = []
        for label in label_counts.keys():
            indices = np.where(self.labels == label)[0]
            start = int(fold * 0.2 * len(indices))
            end = int((fold + 1) * 0.2 * len(indices))
            valid_indices.extend(indices[start:end])
            train_indices.extend(indices[:start])
            train_indices.extend(indices[end:])
        return train_indices, valid_indices

    def get_split(self, dataset, fold):
        train_indices, val_indices = self._get_split_indices(fold)
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        return train_set, val_set


if __name__ == "__main__":
    # 使用示例
    files_path = "./data/"
    with np.load(files_path + "annotation.npz") as f:
        labels = f["stage"][:1000]
    with np.load(files_path + "ecg.npz") as f:
        ecg = f["ECG"][:1000]
    with np.load(files_path + "eog.npz") as f:
        eog = f["EOG"][:1000]
    signal = np.stack((ecg, eog), axis=1)

    signal = DataSets(signal, labels)
    stratified_sampler = StratifiedSampler(labels)
    train_set, val_set = stratified_sampler.get_split(signal, train_ratio=0.8)
    print(len(train_set))
    print(len(val_set))
