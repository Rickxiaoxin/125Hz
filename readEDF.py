import mne
import os
import numpy as np
import xml.etree.ElementTree as ET

np.set_printoptions(threshold=np.inf)

# 读取文件名并取前num个文件
num = 10  # 读取文件数量
signals_path = "E:/shhs/polysomnography/edfs/shhs1"
hypnogram_path = "E:/shhs/polysomnography/annotations-events-nsrr/shhs1"
signals_files = os.listdir(signals_path)
hypnogram_files = os.listdir(hypnogram_path)
signals_files.sort()
hypnogram_files.sort()
signals_files = signals_files[100 : 100 + num]
hypnogram_files = hypnogram_files[100 : 100 + num]

ANNOTATIONS = {
    0: "Sleep stage W",
    1: "Sleep stage 1",
    2: "Sleep stage 2",
    3: "Sleep stage 3",
    4: "Sleep stage R",
}
STAGES_DICT = {
    "Wake|0": 0,
    "Stage 1 sleep|1": 1,
    "Stage 2 sleep|2": 2,
    "Stage 3 sleep|3": 3,
    "Stage 4 sleep|4": 3,
    "REM sleep|5": 5,
}
event_id = {}
for key, value in ANNOTATIONS.items():
    event_id[value] = key


def extract_hypnogram(hypnogram_file):
    """提取划分的睡眠阶段"""
    # 读取原始数据并提取睡眠分期阶段

    with open(os.path.join(hypnogram_path, hypnogram_file)) as f:
        nsrr_annotation = ET.parse(f).findall("ScoredEvents")
        events = []
        for event in nsrr_annotation[0]:
            if event[0].text != "Stages|Stages":
                continue
            stage = STAGES_DICT[event[1].text]
            start = float(event[2].text)
            duration = float(event[3].text)
            events.append([stage, start, duration])
        events = np.array(events)
    Annotations = []
    for stage in events[:, 0]:
        if stage == 0:
            Annotations.append(ANNOTATIONS[0])
        elif stage == 1:
            Annotations.append(ANNOTATIONS[1])
        elif stage == 2:
            Annotations.append(ANNOTATIONS[2])
        elif stage == 3:
            Annotations.append(ANNOTATIONS[3])
        else:
            Annotations.append(ANNOTATIONS[4])
    hypnogram = mne.Annotations(events[:, 1], events[:, 2], Annotations)
    return hypnogram


signals_channels = ["ECG", "EOG(L)", "EOG(R)"]


def read_data(signals_files, hypnogram_files, signals_channels):
    """读取edf文件并使用extract_hypnogram函数添加睡眠阶段"""

    i = 0
    ecg = []
    eog = []
    annotations = []
    for signals_file, hypnogram_file in zip(signals_files, hypnogram_files):
        ignored = False
        raw = mne.io.read_raw_edf(
            os.path.join(signals_path, signals_file),
            include=signals_channels,
            preload=True,
        )
        annotation = extract_hypnogram(hypnogram_file)
        annotation.crop(
            annotation[1]["onset"] - 30 * 60, annotation[-2]["onset"] + 30 * 60
        )
        raw.set_annotations(annotation)
        anno, stage_dict = mne.events_from_annotations(
            raw, event_id=event_id, chunk_duration=30.0
        )
        for key in event_id.keys():
            if key not in stage_dict:
                print(f"{key} not in stage_dict")
                ignored = True
        if ignored:
            continue
        i += 1
        epochs = mne.Epochs(
            raw=raw,
            events=anno,
            event_id=event_id,
            tmin=0,
            tmax=30.0 - 1.0 / raw.info["sfreq"],
            baseline=None,
        )

        data_ecg = epochs.get_data("ECG")
        data_ecg = np.squeeze(data_ecg, axis=1)
        data_eog = epochs.get_data("EOG(R)") - epochs.get_data("EOG(L)")
        data_eog = np.squeeze(data_eog, axis=1)
        stage = epochs.events[:, 2]

        ecg.append(data_ecg)
        eog.append(data_eog)
        annotations.append(stage)
        print(f"{i} files have been processed")

    ecg = np.concatenate(ecg)
    eog = np.concatenate(eog)
    annotations = np.concatenate(annotations)
    print(f"ecg.shape:{ecg.shape}")
    print(f"eog.shape:{eog.shape}")
    print(f"annotations.shape:{annotations.shape}")

    np.save(f"./test/ecg.npy", ecg)
    np.save(f"./test/eog.npy", eog)
    np.save(f"./test/annotation.npy", annotations)


if __name__ == "__main__":
    read_data(signals_files, hypnogram_files, signals_channels)
    labels = np.load("./test/annotation.npy")
    print(labels.shape)
