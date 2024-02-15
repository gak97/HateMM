
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
import pickle
from tqdm import tqdm
from sklearn.metrics import *


from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests

FOLDER_NAME = '/backup/hatemm/Dataset/'
VITF_FOLDER = os.path.join(FOLDER_NAME, 'VITF')

print(torch.__version__)

# Create the VITF directory if it does not exist
if not os.path.exists(VITF_FOLDER):
    os.makedirs(VITF_FOLDER)
    print(f"Created directory: {VITF_FOLDER}")
else:
    print(f"Directory already exists: {VITF_FOLDER}")


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")



num_video_features = 1024
num_audio_features = 128
num_features = 256

k = 2
epochs = 1
batch_size = 1
learning_rate = 1e-4
log_interval = 1
minFrames = 100
img_x1, img_y1 = 299, 299
img_x2, img_y2 = 224, 224

begin_frame, end_frame, skip_frame = 0, minFrames, 0


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
#device = torch.device("cpu") 
# print(device)



import pickle
with open(FOLDER_NAME+'final_allImageFrames.p', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)
    allVidList = list(allDataAnnotation.values())

# train, test split
# train_list, train_label= allDataAnnotation['train']
# val_list, val_label  =  allDataAnnotation['val']
# test_list, test_label  =  allDataAnnotation['test']



# allVidList = []
# allVidLab = []

# allVidList.extend(train_list)
# allVidList.extend(val_list)
# allVidList.extend(test_list)

# allVidLab.extend(train_label)
# allVidLab.extend(val_label)
# allVidLab.extend(test_label)




# def read_images(path, selected_folder):
def read_images(frame_paths):
    X = []
    currFrameCount = 0
    videoFrameCount = len(frame_paths)
    # videoFrameCount = len([name for name in os.listdir(os.path.join(path, selected_folder))])
    if videoFrameCount <= minFrames:
        for frame_path in frame_paths:
            image = Image.open(frame_path)    
        # for i in range(videoFrameCount):
        #     image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i)))

            X.append(image)
            currFrameCount += 1
            if(currFrameCount==minFrames):
                break
        paddingImage = Image.fromarray(np.zeros((100,100)), 'RGB')
        while currFrameCount < minFrames:
            X.append(paddingImage)
            currFrameCount+=1
        #X = torch.stack(X, dim=0)
    else:
        step = int(videoFrameCount/minFrames)
        for i in range(0,videoFrameCount,step):
            image = Image.open(frame_paths[i])
            # image = Image.open(os.path.join(path, selected_folder, 'frame_{}.jpg'.format(i)))
            X.append(image)
            currFrameCount += 1
            if(currFrameCount==minFrames):
                break
        paddingImage = Image.fromarray(np.zeros((100,100)), 'RGB')
        while currFrameCount < minFrames:
            X.append(paddingImage)
            currFrameCount+=1
        #X = torch.stack(X, dim=0)
    return X



# set path
data_image_path = "/backup/hatemm/Dataset_Images/"
# data_image_path = FOLDER_NAME + "Dataset_Images/"  





import numpy as np


from tqdm import tqdm
import os

# for folder, label in tqdm(list(zip(allVidList, allVidLab))):
# for file in tqdm(allVidList):
#     if os.path.exists("VITF/"+file+"_vit.p")==True:
#         continue
for video_frames in tqdm(allVidList):
    # Extract a unique identifier for the video, assuming the first frame's path can be used for this
    # This line extracts the video's folder name from the first frame's path
    video_id = os.path.basename(os.path.dirname(video_frames[0]))
    
    # Construct the filename for the pickle file using the video_id
    pickle_filename = os.path.join(VITF_FOLDER, f"{video_id}_vit.p") 
    # pickle_filename = os.path.join("VITF", f"{video_id}_vit.p")
    
    if os.path.exists(pickle_filename):
        continue
    try:
        video = read_images(video_frames)
        # video = read_images(data_image_path, video_frames)
        inputs = feature_extractor(images=video, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        video_features =[(last_hidden_states[i][0].detach().numpy()) for i in range(0,100)]

        with open(pickle_filename, 'wb') as fp:
            pickle.dump(video_features, fp)
        # with open("VITF/"+file+"_vit.p", 'wb') as fp:
        #     pickle.dump(video_features,fp)
        # del video
        # del inputs
        # del last_hidden_states
    except Exception as e:
        print(e)
        continue

