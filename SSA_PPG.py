from mySSA import mySSA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import healthgate
from scipy import signal
from scipy import stats


def getTime2(timestamp):
    readable = [0 for x in range(int(len(timestamp)))]
    timearray = []
    j = 0
    for i in timestamp:
        readable[j] = datetime.datetime.fromtimestamp(i).strftime('%Y%m%d%H%M')
        j += 1

    for i in range(len(readable)):
        timearray.append(int(readable[i]))

    return timearray


def getTimestamp(initial_time, datalength, fs):
    timelist = []
    flag = 0

    for i in range(0, datalength):
        timelist.append(initial_time)
        flag += 1
        if flag == fs:
            initial_time += 1
            flag = 0

    return timelist


def removeMA_PPG(Raw_PPG, Raw_ACC_X, Raw_ACC_Y, Raw_ACC_Z, freq_PPG, freq_ACC, PPG_initime):
    # bandpass filter raw PPG, raw ACC data
    lowcut = 0.4
    highcut = 5
    Raw_PPG_list = list(map(float, Raw_PPG.values))
    Raw_ACC_X_list = list(map(float, Raw_ACC_X.values))
    Raw_ACC_Y_list = list(map(float, Raw_ACC_Y.values))
    Raw_ACC_Z_list = list(map(float, Raw_ACC_Z.values))

    ppg_filtered = pd.DataFrame(healthgate.butter_bandpassfilter(Raw_PPG, lowcut, highcut, freq_PPG, order=4))
    accX_filtered = pd.DataFrame(healthgate.butter_bandpassfilter(Raw_ACC_X, lowcut, highcut, freq_ACC, order=4))
    accY_filtered = pd.DataFrame(healthgate.butter_bandpassfilter(Raw_ACC_Y, lowcut, highcut, freq_ACC, order=4))
    accZ_filtered = pd.DataFrame(healthgate.butter_bandpassfilter(Raw_ACC_Z, lowcut, highcut, freq_ACC, order=4))

    # time window=8sec
    window_size_ppg = int(8 * freq_PPG)
    window_size_acc = int(8 * freq_ACC)

    # set datetime index for PPG data
    ppg_timestamp = getTimestamp(PPG_initime, len(ppg_filtered), freq_PPG)
    ppg_time = getTime2(ppg_timestamp)
    ppg_datetime = pd.to_datetime(ppg_time, format='%Y%m%d%H%M')
    ppg_filtered.insert(0, "Time", ppg_datetime)
    ppg_filtered = ppg_filtered.set_index(ppg_filtered.columns[0])

    index_acc = 0
    breaker = False

    full = []
    comp_full = []
    embedded_full = []
    for k in range(0, len(Raw_PPG), int(window_size_ppg / 2)):
        # current time window
        window_ppg = ppg_filtered.iloc[k:k + window_size_ppg, ]
        window_accX = accX_filtered.iloc[index_acc:index_acc + window_size_acc, ]
        window_accY = accY_filtered.iloc[index_acc:index_acc + window_size_acc, ]
        window_accZ = accZ_filtered.iloc[index_acc:index_acc + window_size_acc, ]
        index_acc += int(window_size_acc / 2)
        window_accX_list = list(map(float, window_accX.values))
        window_accY_list = list(map(float, window_accY.values))
        window_accZ_list = list(map(float, window_accZ.values))

        if k == 0:
            full.append(window_ppg)
            continue

        ###Singular Spectrum Analysis###
        ssa = mySSA(window_ppg)
        # embedding
        if len(window_ppg) < window_size_ppg:
            embedded_ppg = ssa.embed(embedding_dimension=None, verbose=False, return_df=True)
        else:
            embedded_ppg = ssa.embed(embedding_dimension=200, verbose=False, return_df=True)
        embedded_full.append(embedded_ppg)
        # decompose
        ssa.decompose(verbose=False)
        # get contribution
        contrib = ssa.view_s_contributions(return_df=True)
        # find Facc for each channel of ACC
        fX, fY, fZ = [], [], []
        Pxx_specX, Pxx_specY, Pxx_specZ = [], [], []
        FaccX, FaccY, FaccZ = [], [], []
        # Facc = []

        fX, Pxx_specX = signal.periodogram(window_accX_list, freq_ACC, nfft=4096)
        fY, Pxx_specY = signal.periodogram(window_accY_list, freq_ACC, nfft=4096)
        fZ, Pxx_specZ = signal.periodogram(window_accZ_list, freq_ACC, nfft=4096)

        for i in range(len(Pxx_specX)):
            if Pxx_specX[i] >= (Pxx_specX.max() * 0.5):
                FaccX.append(fX[i])
            if Pxx_specY[i] >= (Pxx_specY.max() * 0.5):
                FaccY.append(fY[i])
            if Pxx_specZ[i] >= (Pxx_specZ.max() * 0.5):
                FaccZ.append(fZ[i])
        Facc = FaccX + FaccY + FaccZ
        # print("Facc : ",Facc)

        #exclude HR freq (dominant and harmonic) from previous window
        Facc_except = []
        Facc_except2 = []
        Facc_except3 = []

        fP, specP = signal.periodogram(list(map(float, full[k // window_size_ppg - 1].values)), freq_PPG, nfft=4096)
        for i in range(0, len(specP)):
            if specP[i] == specP.max():
                for j in range(20):
                    Facc_except.append(fP[i - j])
                    Facc_except.append(fP[i + j])
                for j in range(0, len(fP)):
                    if fP[j] == 2 * fP[i]:
                        for l in range(20):
                            Facc_except2.append(fP[j - l])
                            Facc_except2.append(fP[j + l])

                for j in range(0, len(fP)):
                    if fP[j] == 3 * fP[i]:
                        for l in range(20):
                            Facc_except3.append(fP[j - l])
                            Facc_except3.append(fP[j + l])

        # print(Facc_except)
        # print("Facc_except2: ", Facc_except2)
        # print("Facc_except3: ", Facc_except3)
        for i in range(0, len(Facc)):
            if Facc[i] in Facc_except:
                Facc[i] = -1
        for i in range(0, len(Facc)):
            if Facc[i] in Facc_except2:
                Facc[i] = -2
        for i in range(0, len(Facc)):
            if Facc[i] in Facc_except3:
                Facc[i] = -3

        # print("after exclude Facc : ",Facc)

        # get each decomposed signal
        components = []
        for i in range(0, len(ssa.s_contributions)):
            components.append(
                ssa.view_reconstruction(ssa.Xs[i], names=i, plot=False, symmetric_plots=False, return_df=True))
            components[i] = list(map(float, components[i].values))
        comp_full.append(components)

        # Periodogram for each decomposed signal
        comp_freq = []
        comp_spec = []
        for i in range(0, len(ssa.s_contributions)):
            comp_freq.append(signal.periodogram(components[i], freq_PPG, nfft=4096)[0])
            comp_spec.append(signal.periodogram(components[i], freq_PPG, nfft=4096)[1])

        Facc_comp = []
        for i in range(0, len(ssa.s_contributions)):
            temp = []
            for j in range(len(comp_spec[i])):
                if comp_spec[i][j] >= (comp_spec[i].max() / 2):
                    temp.append(comp_freq[i][j])
            Facc_comp.append(temp)

        ncomp_remove = []
        for i in range(0, len(ssa.s_contributions)):
            for j in range(len(Facc_comp[i])):
                if Facc_comp[i][j] in Facc:
                    ncomp_remove.append(i)
        ncomp_remove = list(set(ncomp_remove))
        print("removed signal: ", ncomp_remove)
        '''
        # temp test: use 95%of decomposed signal (optional)
        sum = 0
        index = []
        for i in range(len(contrib)):
            sum += contrib.iloc[i, 0]
            index.append(i)
            if (sum > 0.95):
                break
        '''

        stream = []
        for i in range(len(ssa.s_contributions)):
            if i in ncomp_remove:
                continue
            # if you want to use all of the signal, remove two lines below (optional)
            # if not (i in index):
            # continue
            else:
                stream.append(i)

        try:
            cleaned_PPG = ssa.view_reconstruction(*[ssa.Xs[i] for i in stream], return_df=True)

            cleaned_PPG_list = list(map(float, cleaned_PPG.values))
            window_ppg_list = list(map(float, window_ppg.values))

            print(k, k + window_size_ppg)

            ##if fundamental&harmonic freq of HR removed use original signal##

            '''
            fs=64
            ppg_peaklist_t = threshold_peakdetection(cleaned_PPG_list, fs)
            ppg_correct_peaklist_t = correct_peaklist(Raw_PPG_list[k:k+window_size_ppg], ppg_peaklist_t, fs)
            #print("cleaned:",ppg_correct_peaklist_t)

            ppg_smooth2 = movingaverage(window_ppg_list, periods=8)
            ppg_peaklist_t2 = threshold_peakdetection(ppg_smooth2, fs)
            ppg_correct_peaklist_t2 = correct_peaklist(Raw_PPG_list[k:k+window_size_ppg], ppg_peaklist_t2, fs)
            #print("original:",ppg_correct_peaklist_t2)

            if len(ppg_correct_peaklist_t)>=15 and len(ppg_correct_peaklist_t)>len(ppg_correct_peaklist_t2) and len(ppg_correct_peaklist_t2)<=10:
                cleaned_PPG = window_ppg
            elif len(ppg_correct_peaklist_t)<=5 and len(ppg_correct_peaklist_t)<len(ppg_correct_peaklist_t2) and len(ppg_correct_peaklist_t2)>=6:
                cleaned_PPG = window_ppg
            '''
            '''
            removed_sum=0
            for i in ncomp_remove:
                removed_sum+=contrib.iloc[i,0]

            if removed_sum>=0.6:
                cleaned_ppg=window_ppg

            '''
        except TypeError:
            cleaned_PPG = window_ppg

        full.append(cleaned_PPG)

    final = []
    for i in range(len(full)):
        for j in range(len(full[i])):
            final.append(full[i].iloc[j, 0])

    return ppg_filtered, final, comp_full