import os
import pickle

def add_transcripts_to_pickle(directory, pickle_file):
    transcripts = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                transcripts[filename.replace(".txt", ".mp4")] = file.read()
    
    if os.path.getsize(pickle_file) > 0:        
        with open(pickle_file, 'rb') as fp:
            existing_data = pickle.load(fp)
    else:
        existing_data = {}

    existing_data.update(transcripts)
    # print(existing_data)
    
    with open(pickle_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

# add_transcripts_to_pickle('/backup/hatemm/Dataset/non_hate_videos/', '/backup/hatemm/Dataset/all__video_vosk_audioMap.p')


def add_audio_paths_to_pickle(directory, pickle_file):
    audio_paths = {}
    for filename in os.listdir(directory):
        if filename.endswith(".mp4"):
            audio_paths[filename.replace(".mp4", "")] = os.path.join(directory, filename)
    
    if os.path.getsize(pickle_file) > 0:        
        with open(pickle_file, 'rb') as fp:
            existing_data = pickle.load(fp)
    else:
        existing_data = {}

    existing_data.update(audio_paths)
    
    with open(pickle_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

# add_audio_paths_to_pickle('/backup/hatemm/Dataset/non_hate_videos/', '/backup/hatemm/Dataset/final_allVideos.p')


def add_video_frames_paths_to_pickle(directory, pickle_file):
    video_frames_paths = {}
    for video_folder in os.listdir(directory):
        video_folder_path = os.path.join(directory, video_folder)
        if os.path.isdir(video_folder_path):
            frame_paths = []
            for frame_file in sorted(os.listdir(video_folder_path)):
                if frame_file.endswith((".jpg", ".png")):  # Assuming frames are in jpg or png format
                    frame_path = os.path.join(video_folder_path, frame_file)
                    frame_paths.append(frame_path)
            video_frames_paths[video_folder] = frame_paths
    
    # Check if the pickle file already exists and has content
    if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
        with open(pickle_file, 'rb') as fp:
            existing_data = pickle.load(fp)
    else:
        existing_data = {}
    
    # Update the existing data with new video frames paths
    existing_data.update(video_frames_paths)
    
    # Write the updated data to the pickle file
    with open(pickle_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

# add_video_frames_paths_to_pickle('/backup/hatemm/Dataset_Images/', '/backup/hatemm/Dataset/final_allImageFrames.p')


# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, train_test_split

# # Load the annotations
# annotations_path = '/backup/hatemm/Dataset/HateMM_annotation.csv'
# df = pd.read_csv(annotations_path)

# # Perform integer encoding for 'label' column
# df['label'] = df['label'].apply(lambda x: 1 if x == 'Hate' else 0)

# # Prepare folds
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
# folds_data = {}

# # Split data into folds
# for fold, (train_val_idx, test_idx) in enumerate(skf.split(df, df['label'])):
#     train_val_df = df.iloc[train_val_idx]
#     test_df = df.iloc[test_idx]
    
#     # Further split train_val into train and validation
#     train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['label'], random_state=2021)
    
#     # Prepare data for this fold
#     folds_data[f'fold{fold+1}'] = {
#         'train': (train_df['video_file_name'].tolist(), train_df['label'].tolist()),
#         'val': (val_df['video_file_name'].tolist(), val_df['label'].tolist()),
#         'test': (test_df['video_file_name'].tolist(), test_df['label'].tolist())
#     }

# # Save the fold details
# fold_details_path = os.path.join('/backup/hatemm/Dataset/', 'allFoldDetails.p')
# with open(fold_details_path, 'wb') as fp:
#     pickle.dump(folds_data, fp)

# print("Fold details saved to:", fold_details_path)



def convert_list_to_dict_in_pickle_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".p"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as fp:
                data_list = pickle.load(fp)
            
            # Extract the base filename without the .p extension, remove '_vit', and append .mp4
            base_filename = filename[:-2]  # Remove the .p extension
            if base_filename.endswith('_vit'):
                base_filename = base_filename[:-4]  # Remove '_vit'
            video_name_key = base_filename + ".mp4"
            data_dict = {video_name_key: data_list}
            
            with open(file_path, 'wb') as fp:
                pickle.dump(data_dict, fp)
            print(f"Converted {filename} to dictionary format.")

# Specify the directory containing the .p files
VITF_FOLDER = '/backup/hatemm/Dataset/VITF_new/'
# convert_list_to_dict_in_pickle_files(VITF_FOLDER)





# with open("/backup/hatemm/Dataset/allFoldDetails.p", 'rb') as fo:
#     existing_data1 = pickle.load(fo)
#     print(len(existing_data1))
#     print(list(existing_data1.values())[0])
#     print(len(list(existing_data1.values())[0]))

# with open("/backup/hatemm/Dataset/VITF_new/non_hate_video_289_vit.p", 'rb') as fo:
#     existing_data1 = pickle.load(fo)
#     print(len(existing_data1))
#     print(list(existing_data1.keys())[0])
#     print(len(list(existing_data1.values())[0]))

# with open("/backup/hatemm/Dataset/final_allImageFrames.p", 'rb') as fo:
#     existing_data1 = pickle.load(fo)
#     print(len(existing_data1))
#     print(list(existing_data1.values())[0])
#     print(len(list(existing_data1.values())[0]))

# with open("/backup/hatemm/Dataset/final_allNewData.p", 'rb') as fo:
#     existing_data1 = pickle.load(fo)
#     print(len(existing_data1))
#     print(list(existing_data1.values())[0])
#     print(len(list(existing_data1.values())[0]))

# with open("/backup/hatemm/Dataset/MFCCFeaturesNew.p", 'rb') as fo:
#     existing_data2 = pickle.load(fo)
#     print(len(existing_data2))
#     print(list(existing_data2.keys())[1])
#     # print(list(existing_data2.values())[0])
#     print(len(list(existing_data2.values())[1]))

# with open("/backup/hatemm/Dataset/all__video_vosk_audioMap.p", 'rb') as fo:
#     existing_data1 = pickle.load(fo)
#     print(len(existing_data1))
#     print(list(existing_data1.values())[0])
#     print(len(list(existing_data1.values())[0]))
#     for i in existing_data1:
#         print(existing_data1[i])
#         break

# with open("/backup/hatemm/Dataset/all_HateXPlainembedding.p", 'rb') as fp:
#     existing_data2 = pickle.load(fp)
#     print(len(existing_data2))
#     print(len(list(existing_data2.values())[0]))

# with open("/backup/hatemm/Dataset/all_rawBERTembedding.p", 'rb') as fq:
#     existing_data3 = pickle.load(fq)
#     print(len(existing_data3))
#     print(existing_data3['hate_video_108.mp4'])
#     print(list(existing_data3.keys())[0])
#     print(len(list(existing_data3.values())[0]))
