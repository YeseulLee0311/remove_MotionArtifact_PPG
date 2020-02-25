import numpy as np

def calc_mean(raw_list, freq):
    window_size = 2
    overlap_size = 1  # sliding window
    mean = []
    for i in range(0, len(raw_list), freq * (window_size - overlap_size)):
        mean.append(np.mean(raw_list[i:i + (freq * window_size)]))

    return mean