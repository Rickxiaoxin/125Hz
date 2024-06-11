from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def micro_f1(y_true, y_pred):
    """
    计算micro-f1分数

    参数:
    y_true (list): 真实标签列表
    y_pred (list): 预测标签列表

    返回:
    micro_f1 (float): micro-f1分数
    """

    # 计数真实标签和预测标签
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)

    # 计算TP, FP和FN
    tp = sum((y_true_counts & y_pred_counts).values())
    fp = sum((y_pred_counts - y_true_counts).values())
    fn = sum((y_true_counts - y_pred_counts).values())

    # 计算精确率和召回率
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # 计算micro-f1分数
    micro_f1 = 2 * (precision * recall) / (precision + recall)

    return micro_f1


if __name__ == "__main__":
    true_labels = np.load("./test/annotation.npy")
    predict_labels = np.load("./test/test_output.npy")
    confusion_matrix = np.zeros((5, 5))
    for true_label, pred_label in zip(true_labels, predict_labels):
        confusion_matrix[true_label, pred_label] += 1
    # 绘制矩阵图
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(
    #     confusion_matrix,
    #     annot=True,
    #     cmap="Blues",
    #     xticklabels=["W", "N1", "N2", "N3", "REM"],
    #     yticklabels=["W", "N1", "N2", "N3", "REM"],
    # )
    # plt.xlabel("Predicted Labels")
    # plt.ylabel("True Labels")
    # plt.show()
    print(confusion_matrix)
    # f1 = micro_f1(true_labels, predict_labels)
    # print(f1)
