import pandas as pd
import os
import csv
import pandas as pd
import glob
import moviepy.editor as mp
import torch
import pandas as pd
import librosa
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import librosa.display
import matplotlib.pyplot as plt
import tarfile
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



# Helper function to generate mfccs
def extract_mfcc(path):
    audio, sr=librosa.load(path)
    mfccs=librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)
	
FOLDER_NAME ='/backup/hatemm/Dataset/'
# audioPath = FOLDER_NAME + "/AudioFiles/"  
audio_plots_path = os.path.join(FOLDER_NAME, "Audio_plots") 

# Ensure the Audio_plots directory exists
os.makedirs(audio_plots_path, exist_ok=True)



import pickle
with open(FOLDER_NAME+'final_allNewData.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)
    # print(list(allDataAnnotation.values())[0])
    allVidList = list(allDataAnnotation.values())

# train, test split
# train_list, train_label= allDataAnnotation['train']
# val_list, val_label  =  allDataAnnotation['val']
# test_list, test_label  =  allDataAnnotation['test']


# In[4]:


# allVidList = []
# allVidLab = []

# allVidList.extend(train_list)
# allVidList.extend(val_list)
# allVidList.extend(test_list)

# allVidLab.extend(train_label)
# allVidLab.extend(val_label)
# allVidLab.extend(test_label)




allAudioFeatures = {}
failedList = []


import traceback

for i in tqdm(allVidList):
    try:
        aud = extract_mfcc(i)
        # Extract the base name without extension as the key
        video_name = os.path.basename(i)
        # video_name = os.path.splitext(os.path.basename(i))[0]
        allAudioFeatures[video_name.replace(".wav", ".mp4")] = aud
        # aud = extract_mfcc(audioPath+i+".wav")
        # allAudioFeatures[i]=aud
    except Exception as e:
        print(f"Error processing {i}: {e}")
        traceback.print_exc()  # This will print the stack trace
        failedList.append(i)




for i in failedList:
    allAudioFeatures[i] = np.zeros(40)


import pickle
with open(FOLDER_NAME+'MFCCFeaturesNew.p', 'wb') as fp:
    pickle.dump(allAudioFeatures,fp)


#--------------------Audio Spectrum Generation For VGG19---------------------

import os
from os import walk, listdir
import cv2
import shutil
import pandas as pd



from scipy.io.wavfile import read
from tqdm import tqdm
import matplotlib.pyplot as plt





from tqdm import tqdm
for selected_file in tqdm(allVidList):
    try:
        base_name = os.path.basename(selected_file)
        file_name, _ = os.path.splitext(base_name)
        # path1 = os.path.join(audioPath, selected_file)

        sample_rate, data = read(selected_file)

        fig = plt.figure(num=1, clear=True)
        ax = fig.add_subplot(111)
        # ax.plot(read(path1)[1])
        ax.plot(data)

        save_path = os.path.join(audio_plots_path, file_name + '.png')
        # fig.savefig(FOLDER_NAME + "Audio_plots" + selected_file + '.png')
        fig.savefig(save_path)
        fig.clf()
        plt.close(fig)

    except Exception as e:
        print(e)
        continue
