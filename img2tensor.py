import os
from torchvision import transforms
from PIL import Image
import numpy as np
from multiprocessing import Pool


def get_file(filepath):
    """加载图片目录"""
    ecg = os.listdir(os.path.join(filepath, "ecg"))
    eog = os.listdir(os.path.join(filepath, "eog"))
    ecg = sorted(ecg, key=lambda x: int("".join(filter(str.isdigit, x))))
    eog = sorted(eog, key=lambda x: int("".join(filter(str.isdigit, x))))
    # stage = None
    # filenames = np.vstack([ecg, eog])
    # with np.load(os.path.join(filepath + "/annotation.npz")) as f:
    #     stage = f["stage"]
    return ecg, eog  # , stage


# 将图片保存为数组
def Image_to_Tensor(
    image_flag,
):
    """读取图片并保存为可用数据"""
    flag, image_file = image_flag
    image_to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    num, _ = os.path.splitext(image_file)
    if flag:
        image_dir = os.path.join("./photo/ecg", image_file)
    else:
        image_dir = os.path.join("./photo/eog", image_file)
    image_feature = Image.open(image_dir).convert("L")
    image_feature = image_feature.crop((0, 75, 750, 125))
    image_tensor = image_to_tensor(image_feature)
    # image_tensor = image_tensor.unsqueeze(0)
    print(f"processing {image_file}")
    return int(num), image_tensor


# 定义一个函数来判断数组是否按大小顺序排列
def is_sorted(arr):
    n = len(arr)
    # 遍历数组中的元素，逐一比较相邻的元素
    for i in range(1, n):
        # 如果存在逆序，则数组不是按大小顺序排列
        if arr[i] < arr[i - 1]:
            return False
    # 如果数组中的所有元素都满足大小顺序，则返回True
    return True


if __name__ == "__main__":
    file_path = "./photo"
    save_path = "./signal/"
    ecg_files, eog_files = get_file(file_path)
    ecg_flag = [(1, ecg_file) for ecg_file in ecg_files]
    eog_flag = [(0, eog_file) for eog_file in eog_files]

    with Pool() as pool:
        ecg = pool.map(Image_to_Tensor, ecg_flag)
    print("All hvae been converted,then will be sorted")
    sorted_ecg = sorted(ecg, key=lambda x: x[0])
    sort = np.array([i for i, _ in sorted_ecg])
    assert np.all(np.diff(sort) > 0)
    del sort, ecg
    print("All are sorted successfully")
    ECG = np.stack([ecg for _, ecg in sorted_ecg])
    print(ECG.shape)
    print(f"waiting for ecg saving...")
    np.save("./signal/ecg.npy", ECG)
    del ECG

    with Pool() as pool:
        eog = pool.map(Image_to_Tensor, eog_flag)
    sorted_eog = sorted(eog, key=lambda x: x[0])
    sort = np.array([i for i, _ in sorted_eog])
    assert np.all(np.diff(sort) > 0)
    del sort, eog
    EOG = np.stack([eog for _, eog in sorted_eog])
    print(EOG.shape)
    print(f"waiting for eog saving...")
    np.save("./signal/eog.npy", EOG)
