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
        if filename.endswith(".wav"):
            audio_paths[filename.replace(".wav", "")] = os.path.join(directory, filename)
    
    if os.path.getsize(pickle_file) > 0:        
        with open(pickle_file, 'rb') as fp:
            existing_data = pickle.load(fp)
    else:
        existing_data = {}

    existing_data.update(audio_paths)
    
    with open(pickle_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

# add_audio_paths_to_pickle('/backup/hatemm/Dataset/non_hate_videos/', '/backup/hatemm/Dataset/final_allNewData.p')


# with open("/backup/hatemm/Dataset/final_allNewData.p", 'rb') as fo:
#     existing_data1 = pickle.load(fo)
#     print(len(existing_data1))
#     print(list(existing_data1.keys())[0])
#     print(len(list(existing_data1.values())[0]))

with open("/backup/hatemm/Dataset/MFCCFeatures.p", 'rb') as fo:
    existing_data1 = pickle.load(fo)
    print(len(existing_data1))
    print(list(existing_data1.values())[0])
    print(len(list(existing_data1.values())[0]))

# with open("/backup/hatemm/Dataset/all__video_vosk_audioMap.p", 'rb') as fo:
#     existing_data1 = pickle.load(fo)
#     print(len(existing_data1))
#     print(len(list(existing_data1.values())[0]))

# with open("/backup/hatemm/Dataset/all_HateXPlainembedding.p", 'rb') as fp:
#     existing_data2 = pickle.load(fp)
#     print(len(existing_data2))
#     print(len(list(existing_data2.values())[0]))

# with open("/backup/hatemm/Dataset/all_rawBERTembedding.p", 'rb') as fq:
#     existing_data3 = pickle.load(fq)
#     print(len(existing_data3))
#     print(len(list(existing_data3.values())[0]))