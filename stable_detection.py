import pandas as pd
import pickle
from features.stdev import calc_stdev
from features.PSD_PPG import calc_PSD


def stable_detection(raw_PPG, raw_accX, raw_accY, raw_accZ, freq_PPG, freq_ACC):
    if type(raw_PPG) == pd.core.frame.DataFrame or type(raw_accX) == pd.core.frame.DataFrame or type(
            raw_accY) == pd.core.frame.DataFrame or type(raw_accZ) == pd.core.frame.DataFrame:
        raw_PPG = list(map(float, raw_PPG.values))
        raw_accX = list(map(float, raw_accX.values))
        raw_accY = list(map(float, raw_accY.values))
        raw_accZ = list(map(float, raw_accZ.values))
    freq_PPG = int(freq_PPG)
    freq_ACC = int(freq_ACC)

    # load the models
    feature_scaler = pickle.load(open('./models/feature_scaler.sav', 'rb'))
    score_scaler = pickle.load(open('./models/score_scaler.sav', 'rb'))
    IF = pickle.load(open('./models/PPG_stable_detection.sav', 'rb'))

    # extract features
    stdevX = pd.DataFrame(calc_stdev(raw_accX, freq_ACC))
    stdevY = pd.DataFrame(calc_stdev(raw_accY, freq_ACC))
    stdevZ = pd.DataFrame(calc_stdev(raw_accZ, freq_ACC))
    psd = pd.DataFrame(calc_PSD(raw_PPG, freq_PPG))
    # merge dataset
    merged_test = pd.concat([stdevX, stdevY, stdevZ, psd], axis=1)
    merged_test.columns = ['stdevX', 'stdevY', 'stdevZ', 'PSD']
    # normalize datasert
    normalized_test = pd.DataFrame(feature_scaler.transform(merged_test))
    # get score
    score = IF.decision_function(normalized_test)
    # normalize score
    normalized_score = pd.DataFrame(score_scaler.transform(pd.DataFrame(score)))
    # get binary score
    binary = IF.predict(normalized_test)

    return normalized_score, binary
