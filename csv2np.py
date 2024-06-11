import pandas as pd
import numpy as np


def np2csv(ecg_file, eog_file):
    ecg = np.load(ecg_file)
    eog = np.load(eog_file)

    ecg_df = pd.DataFrame(ecg)
    eog_df = pd.DataFrame(eog)
    ecg_df.to_csv("ecg.csv")
