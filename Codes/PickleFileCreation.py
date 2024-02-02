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

add_transcripts_to_pickle('./Dataset/hate_videos/', './Dataset/hate_videos/all__video_vosk_audioMap.p')



# with open("Dataset/hate_videos/all__video_vosk_audioMap.p", 'rb') as fo:
#     existing_data1 = pickle.load(fo)
#     # print(len(existing_data1))
#     print(len(list(existing_data1.values())[0]))

# with open("Dataset/hate_videos/all_HateXPlainembedding.p", 'rb') as fp:
#     existing_data2 = pickle.load(fp)
#     # print(len(existing_data2))
#     print(len(list(existing_data2.values())[0]))

# with open("Dataset/hate_videos/all_rawBERTembedding.p", 'rb') as fq:
#     existing_data3 = pickle.load(fq)
#     # print(len(existing_data3))
#     print(len(list(existing_data3.values())[0]))