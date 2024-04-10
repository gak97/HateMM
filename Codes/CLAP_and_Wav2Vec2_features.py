import os
import traceback
import torchaudio
import librosa
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor, ClapAudioModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2FeatureExtractor

FOLDER_NAME ='/backup/hatemm/Dataset/'

import pickle
with open(FOLDER_NAME+'final_allNewData.pkl', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)
    allVidList = list(allDataAnnotation.values())

def get_CLAP_features():
    # Load the CLAP model and processor
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")
    model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
    model = model.to("cuda")

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Get the CLAP features
    allAudioFeatures = {}
    failedList = []
    for i in tqdm(allVidList):
        try:
            # Load the audio file
            audio, _ = librosa.load(i, sr=48000)
            # waveform, sample_rate = torchaudio.load(i)
            # Process the audio waveform
            outputs = processor(audios=audio, return_tensors="pt", sampling_rate=48000)
            features = model(**outputs).last_hidden_state
            video_name = os.path.basename(i)
            allAudioFeatures[video_name.replace(".wav", ".mp4")] = features.cpu().detach().numpy()  # Convert tensors to numpy arrays for serialization
            # chunk_size = 30 * 48000  # 30 seconds * sample rate
            # audio_chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
            # all_features = []
            # for chunk in audio_chunks:
            #     # Process each audio chunk
            #     inputs = processor(audios=chunk, return_tensors="pt", sampling_rate=48000).input_values.to("cuda")
            #     with torch.no_grad():
            #         features = model(**inputs).last_hidden_state
            #         all_features.append(features.cpu().detach().numpy().squeeze(0))  # Remove the first dimension
            # # Concatenate all chunk features along the first dimension
            # concatenated_features = np.concatenate(all_features, axis=0)
            # video_name = os.path.basename(i)
            # allAudioFeatures[video_name.replace(".wav", ".mp4")] = concatenated_features
        except Exception as e:
            failedList.append(i)
            print(f"Failed to extract features for {i}")
            print(f"Error: {e}")
            traceback.print_exc()
    return allAudioFeatures, failedList

allAudioFeatures, failedList = get_CLAP_features()
print(failedList)

# Save the features
with open(FOLDER_NAME+'CLAP_features.pkl', 'wb') as fp:
    pickle.dump(allAudioFeatures, fp)




def get_Wav2Vec2_features():
    # Load the Wav2Vec2 model and processor
    # processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/tiny-wav2vec2-no-tokenizer")
    # processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    # model = Wav2Vec2Model.from_pretrained("patrickvonplaten/tiny-wav2vec2-no-tokenizer")
    model = model.to("cuda")

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Get the Wav2Vec2 features
    allAudioFeatures = {}
    failedList = []
    for i in tqdm(allVidList):
        try:
            # Load the audio file
            audio, _ = librosa.load(i, sr=16000)
            # Process the audio waveform
            # inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=16000).input_values.to("cuda")
            # with torch.no_grad():
            #     features = model(input_values=inputs).last_hidden_state
            # video_name = os.path.basename(i)
            # allAudioFeatures[video_name.replace(".wav", ".mp4")] = features.cpu().detach().numpy()
            # Chunk the audio into 30-second segments
            chunk_size = 30 * 16000  # 30 seconds * sample rate
            audio_chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]
            all_features = []
            for chunk in audio_chunks:
                # Process each audio chunk
                inputs = feature_extractor(chunk, return_tensors="pt", sampling_rate=16000).input_values.to("cuda")
                with torch.no_grad():
                    features = model(input_values=inputs).last_hidden_state
                    all_features.append(features.cpu().detach().numpy().squeeze(0))  # Remove the first dimension
            # Concatenate all chunk features along the first dimension
            concatenated_features = np.concatenate(all_features, axis=0)
            video_name = os.path.basename(i)
            allAudioFeatures[video_name.replace(".wav", ".mp4")] = concatenated_features
        except Exception as e:
            failedList.append(i)
            print(f"Failed to extract features for {i}")
            print(f"Error: {e}")
            traceback.print_exc()
    return allAudioFeatures, failedList

# allAudioFeatures, failedList = get_Wav2Vec2_features()
# print(failedList)

# Save the features
# with open(FOLDER_NAME+'Wav2Vec2_features_chunked.pkl', 'wb') as fp:
#     pickle.dump(allAudioFeatures, fp)




# from datasets import load_dataset
# from transformers import AutoProcessor, ClapAudioModel

# # dataset = load_dataset("ashraq/esc50")
# # audio_sample = dataset["train"]["audio"][0]["array"]
# audio_sample = '/backup/hatemm/Dataset/hate_videos/hate_video_95.wav'

# model = ClapAudioModel.from_pretrained("laion/clap-htsat-fused")
# processor = AutoProcessor.from_pretrained("laion/clap-htsat-fused")

# audio, _ = librosa.load(audio_sample)
# inputs = processor(audios=audio, return_tensors="pt")

# outputs = model(**inputs)
# last_hidden_state = outputs.last_hidden_state
# print(last_hidden_state.shape)
# print(last_hidden_state)
