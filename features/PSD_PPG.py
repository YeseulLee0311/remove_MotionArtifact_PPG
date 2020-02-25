from scipy import signal
import numpy as np


def calc_PSD(PPG_list, freq):
    # window size=2sec
    window_size = 2
    overlap_size = 1  # sliding window
    psd_mean = []
    for i in range(0, len(PPG_list), freq * (window_size - overlap_size)):
        temp_f = []
        temp_psd = []
        f, Pxx_den = signal.periodogram(PPG_list[i:i + freq * window_size], freq,
                                        nfft=window_size * freq * 8)
        for j in range(len(f)):
            if 1 <= f[j] <= 2:
                temp_f.append(f[j])
                temp_psd.append(Pxx_den[j])
        psd_mean.append(np.mean(temp_psd))

    return psd_mean
