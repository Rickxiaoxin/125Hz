from Embedding import Normalization
import numpy as np

ecg, eog = [], []
for i in range(13):
    ecg_sec = np.load(f"./data/sample{i}/ecg.npy").reshape(-1, 3750)
    ecg.append(ecg_sec)
ecg = np.concatenate(ecg, axis=0)
for i in range(13):
    eog_sec = np.load(f"./data/sample{i}/eog.npy").reshape(-1, 3750)
    eog.append(eog_sec)
eog = np.concatenate(eog, axis=0)
data = np.stack((ecg, eog), axis=1)
del ecg, eog
data = Normalization(data).astype(np.float32)
np.save("./signal/data_norm.npy", data)
# data = np.load("./signal/data_norm.npy")
# print(data.shape)
labels = []
for i in range(13):
    labels_sec = np.load(f"./data/sample{i}/annotation.npy")
    labels.append(labels_sec)
labels = np.concatenate(labels, axis=0)
print(type(labels), labels.shape)
np.save("./signal/labels.npy", labels)
