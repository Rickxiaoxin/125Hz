import numpy as np
import os

image_path = "./photo/ecg"
ecg_files = os.listdir(image_path)
ecg = sorted(ecg_files, key=lambda x: int("".join(filter(str.isdigit, x))))
for i in range(12308):
    if ecg[i] != f"{i}.png":
        print(i)


ecg = []
for i in range(13):
    ecg_sec = np.load(f"./data/sample{i}/ecg.npy").reshape(-1, 3750)
    ecg.append(ecg_sec)

print(len(ecg))
ecg = np.concatenate(ecg, axis=0)
print(ecg.shape, type(ecg))
