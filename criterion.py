import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, fftfreq, fftshift
from scipy import signal
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score

true_labels = np.load("./signal/labels-4.npy")
pred_labels = np.load("./output.npy")

print(true_labels.size)
stage_indeces = []

for i in range(4):
    label_indices = np.where(
        true_labels == i,
    )[0]
    stage_indeces.append(label_indices)
for i in range(4):
    print(stage_indeces[i].size)

# 计算混淆矩阵
confusion_matrix = np.zeros((4, 4))
for true_label, pred_label in zip(true_labels, pred_labels):
    confusion_matrix[true_label, pred_label] += 1
# confusion_matrix=confusion_matrix(true_labels,pred_labels)
for i, label in enumerate(confusion_matrix):
    amount = label.sum()
    accuracy = label / amount
    confusion_matrix[i, :] = accuracy
print(confusion_matrix)
# 绘制矩阵图
plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix,
    annot=True,
    cmap="Blues",
    xticklabels=["W", "N1/N2", "N3", "REM"],
    yticklabels=["W", "N1/N2", "N3", "REM"],
    fmt=".3f",
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# Micro-F1
print(f1_score(true_labels, pred_labels, average="micro"))
print(
    cohen_kappa_score(
        true_labels,
        pred_labels,
    )
)
