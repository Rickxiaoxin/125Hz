import numpy as np
from collections import Counter

labels = np.load("./signal/labels.npy")
label_counts = Counter(labels)
# train_indices = []
# valid_indices = []
# for label in label_counts.keys():
#     indices = np.where(labels == label)[0]
#     print(len(indices), type(indices))
#     start = int(0 * 0.2 * len(indices))
#     end = int((0 + 1) * 0.2 * len(indices))
#     valid_indices.extend(indices[start:end])
#     train_indices.extend(indices[:start])
#     train_indices.extend(indices[end:])
#     print(len(train_indices), len(valid_indices))
for i in range(len(labels)):
    if labels[i] > 1:
        labels[i] -= 1

np.save("./signal/labels-4.npy", labels)
lables4 = np.load("./signal/labels-4.npy")
labels4_counts = Counter(lables4)
print(label_counts)
print(labels.shape)
print(labels4_counts)
print(lables4.shape)
