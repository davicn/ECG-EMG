import os 
import pandas as pd 
import numpy as np 

PATH = os.getcwd()

url = 'wget -cr --no-parent --http-user="nedc_tuh_eeg" --http-passwd="nedc_tuh_eeg" https://www.isip.piconepress.com/projects/tuh_eeg/downloads/tuh_eeg_seizure/v1.5.0/edf/'

emg_w = pd.read_csv(PATH +"/docs/EMG_train_comcrise.csv")
emg_n = pd.read_csv(PATH +"/docs/EMG_train_semcrise.csv")

for i in emg_n.loc[:,'path'].to_numpy():
    print(url + i.replace('tse','edf'))

