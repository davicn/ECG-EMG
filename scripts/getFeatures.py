import os
import pandas as pd
import numpy as np
import mne
import torch
import sys

PATH = os.getcwd().replace('notebooks', '')
sys.path.insert(0, PATH)

from functions.gpuFeatures import energia, variancia

# %%
com = pd.read_csv(PATH + '/docs/EMG_train_comcrise.csv')
com = com[com['freq'] == 256]
com.index = np.arange(len(com))


sem = pd.read_csv(PATH + '/docs/EMG_train_semcrise.csv')
sem = sem[sem['freq'] == 256]
sem.index = np.arange(len(sem))


# %%
energia_com = 0
variancia_com = 0

for i in range(len(com)):
    file = com.loc[i, 'path'].replace('tse', 'edf')
    raw = mne.io.read_raw_edf(
        PATH + '/data/'+file, preload=False, verbose=False)
    d = raw.to_data_frame().loc[:, 'EEG EKG1-REF'].to_numpy()

    aux1 = energia(d[256*int(com.loc[i, 'start']):256*int(com.loc[i, 'end'])])
    aux2 = variancia(d[256*int(com.loc[i, 'start'])
                     :256*int(com.loc[i, 'end'])])

    energia_com = np.append(energia_com, aux1)
    variancia_com = np.append(variancia_com, aux1)

    print(energia_com.shape)
    print(variancia_com.shape)

np.save('energia_com.npy', energia_com)
np.save('variancia_com.npy', variancia_com)

