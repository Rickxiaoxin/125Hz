import pywt
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

wavelist = pywt.wavelist("morl")
print(wavelist)
fs = 125
wavename = "morl"
totalscale = 125
fc = pywt.central_frequency(wavename)
print(fc)
cparam = 2 * fc * totalscale
scales = cparam / np.arange(1, totalscale + 1)
t = np.arange(0, 30, 1 / fs)


def wavelets(param):
    flag, i, signal = param
    if flag:
        filepath = f"./photo/ecg/{i}.png"
    else:
        filepath = f"./photo/eog/{i}.png"
    coeffs, freqs = pywt.cwt(signal, scales, wavename, 1 / fs)
    # print(f"coeffs shape: {coeffs.shape}")
    # print(f"freqs shape: {freqs.shape}")
    fig = plt.figure(figsize=(7.5, 1.25))
    plt.contourf(abs(coeffs))
    # plt.ylabel(u"freq(Hz)")
    # plt.xlabel(u"time(s)")
    # plt.colorbar()
    plt.xticks([])  # 清除x
    plt.yticks([])
    plt.axis("off")  # 清除轴
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # 删除坐标轴刻度
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.show()

    fig.savefig(filepath)
    fig.clear()
    plt.close(fig)
    print(f"{filepath} has been processed")


if __name__ == "__main__":
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # results = []
    # signals = np.load(signal_path)["EOG"]
    # for i, signal in enumerate(signals):
    #     results.append(pool.apply_async(wavelets, (i, signal)))
    # pool.close()
    # pool.join()

    # for result in results:
    #     result.get()
    ecg = []
    for i in range(13):
        ecg_sec = np.load(f"./data/sample{i}/ecg.npy").reshape(-1, 3750)
        ecg.append(ecg_sec)
    ecg = np.concatenate(ecg, axis=0)
    param = [(1, i, signal) for i, signal in enumerate(ecg)]
    with multiprocessing.Pool() as pool:
        pool.map(wavelets, param)

    del ecg

    eog = []
    for i in range(13):
        eog_sec = np.load(f"./data/sample{i}/eog.npy").reshape(-1, 3750)
        eog.append(eog_sec)
    eog = np.concatenate(eog, axis=0)
    param = [(0, i, signal) for i, signal in enumerate(eog)]
    with multiprocessing.Pool() as pool:
        pool.map(wavelets, param)
