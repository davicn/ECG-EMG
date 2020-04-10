import os
import pandas as pd
import numpy as np
import mne
import torch
import sys

PATH = os.getcwd().replace('notebooks', '')
sys.path.insert(0, PATH)

from functions.gpuFeatures import energia, variancia

sem = pd.read_csv(PATH + '/docs/EMG_train_semcrise.csv')
sem = sem[sem['freq'] == 256]
sem.index = np.arange(len(sem))

aux = np.load(PATH+'/data/energia_com.npy')
fs = 256

energia_sem = 0
variancia_sem = 0

for i in range(len(sem)):
    file = sem.loc[i, 'path'].replace('tse', 'edf')
    raw = mne.io.read_raw_edf(
        PATH + '/data/'+file, preload=False, verbose=False)
    d = raw.to_data_frame().loc[:, 'EEG EKG1-REF'].to_numpy()

    aux1 = energia(d)

    energia_sem = np.append(energia_sem, aux1)
    variancia_sem = np.append(variancia_sem, aux1)

    if len(energia_sem)>len(aux) and len(variancia_sem)>len(aux):
        break

print(energia_sem.shape)
print(variancia_sem.shape) 

np.save('energia_sem.npy', energia_sem)
np.save('variancia_sem.npy', variancia_sem)
