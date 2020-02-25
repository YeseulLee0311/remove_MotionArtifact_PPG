import scipy.stats

def calc_corr(raw_list1, raw_list2, freq):
    window_size = 2
    overlap_size = 1  # sliding window
    corr = []

    for i in range(0, len(raw_list1), freq * (window_size - overlap_size)):
        corr.append(abs(scipy.stats.pearsonr(raw_list1[i:i + (freq * window_size)],
                                             raw_list2[i:i + (freq * window_size)])[0]))

    return corr