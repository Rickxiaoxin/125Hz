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

    def _get_split_indices(self, train_ratio):
        # 确定每个标签的样本数量
        label_counts = Counter(self.labels)

        # 计算每个标签在整个数据集中的占比
        total_samples = len(self.labels)
        label_ratios = {
            label: count / total_samples for label, count in label_counts.items()
        }

        # 确定训练集和测试集的大小
        # train_ratio = 0.6  # 训练集所占比例
        # test_ratio = 0.2  # 测试集所占比例
        # valid_ratio = 0.2  # 验证集所占比例
        train_counts = {
            label: int(train_ratio * count) for label, count in label_counts.items()
        }
        # test_counts = {
        #     label: int(test_ratio * count) for label, count in label_counts.items()
        # }
        valid_counts = {
            label: count - train_count
            for (label, count), train_count, in zip(
                label_counts.items(), train_counts.values()
            )
        }

        # 根据标签数量划分训练集和测试集的样本
        train_indices = []
        # test_indices = []
        valid_indices = []
        for label in label_counts.keys():
            indices = np.where(self.labels == label)[0]
            permuted_indices = np.random.permutation(len(indices))
            train_indices.extend(indices[permuted_indices[: train_counts[label]]])
            # test_indices.extend(
            #     indices[
            #         permuted_indices[
            #             train_counts[label] : train_counts[label] + test_counts[label]
            #         ]
            #     ]
            # )
            valid_indices.extend(indices[permuted_indices[train_counts[label] :]])
        # train_indices = []
        # val_indices = []
        # for cls, indices in zip(self.classes, self.class_indices):
        #     cls_indices = indices.tolist()
        #     n_train = int(train_ratio * len(cls_indices))
        #     train_indices.extend(cls_indices[:n_train])
        #     val_indices.extend(cls_indices[n_train:])
        return train_indices, valid_indices

    def get_split(self, dataset, train_ratio=0.8):
        train_indices, val_indices = self._get_split_indices(train_ratio)
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
