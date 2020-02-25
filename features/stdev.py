import numpy as np

def calc_stdev(raw_list, freq):
    window_size = 2
    overlap_size = 1  # sliding window
    stdev = []
    for i in range(0, len(raw_list), freq * (window_size - overlap_size)):
        stdev.append(np.std(raw_list[i:i + (freq * window_size)]))

    return stdev