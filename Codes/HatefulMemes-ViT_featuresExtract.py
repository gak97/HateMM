import torch
from tqdm import tqdm
import os, requests
import pickle
import base64
from PIL import Image
from io import BytesIO
import numpy as np

from transformers import ViTFeatureExtractor, ViTModel

from datasets import load_dataset
dataset = load_dataset('limjiayi/hateful_memes_expanded')

FOLDER_NAME = '/backup/hatemm/Dataset/'

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
print(device)

# Move the model to the GPU
model.to(device)

# Load processed IDs from a file
processed_hxp_ids = set()
try:
    with open('processed_img_test_ids.txt', 'r') as file:
        for line in file:
            processed_hxp_ids.add(line.strip())
except FileNotFoundError:
    pass

print("Starting image processing for ViT...")
split = 'test'
ImgEmbedding_train = {}
print(f"Processing split: {split}")

for example in tqdm(dataset[split]):
    try:
        image_id_hatexplain = example['id']
        # Check if the image ID ends with '.png' or '.jpg'
        if image_id_hatexplain.endswith('.png') or image_id_hatexplain.endswith('.jpg'):
            image_id_hatexplain = image_id_hatexplain
        else:
            image_id_hatexplain += '.png'
        if image_id_hatexplain in processed_hxp_ids:
            print(f"Skipping image with ID: {image_id_hatexplain}")
            continue

        # Download the image from the Hugging Face repository
        image_url = f"https://huggingface.co/datasets/limjiayi/hateful_memes_expanded/resolve/main/img/{image_id_hatexplain}"
        response = requests.get(image_url)

        try:
            if response.status_code == 200:
                image_bytes = response.content
                image = Image.open(BytesIO(image_bytes))
                # image = Image.open(requests.get(example['img'], stream=True).raw)
                inputs = feature_extractor(images=image, return_tensors="pt")

                # Ensure input tensor is on the same device as model's weights
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    last_hidden_states = outputs.last_hidden_state
                    ImgEmbedding_train[image_id_hatexplain] = [last_hidden_states[i][0].cpu().numpy() for i in range(last_hidden_states.shape[0])]

            else:
                print(f"Error downloading image: {image_id_hatexplain}")
                continue
        except Exception as decode_error:
            print(f"Error decoding image: {image_id_hatexplain}")
            print(decode_error)
            continue
            
        # Check if the pickle file already exists and has content
        pickle_file = FOLDER_NAME + f'hatefulmemes_{split}_VITembedding.pkl'
        if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
            with open(pickle_file, 'rb') as fp:
                existing_data = pickle.load(fp)
                existing_data.update(ImgEmbedding_train)
        # if os.path.exists(pickle_file):
            with open(pickle_file, 'wb') as fp:
                pickle.dump(existing_data, fp)
        else:
            with open(pickle_file, 'wb') as fp:
                pickle.dump(ImgEmbedding_train, fp)

        # Check if the .npy file already exists and has content
        # npy_file = FOLDER_NAME + f'hatefulmemes_{split}_VITembedding.npy'
        # if os.path.exists(npy_file) and os.path.getsize(npy_file) > 0:
        #     existing_data = np.load(npy_file, allow_pickle=True)
        #     existing_data = existing_data.item()
        #     existing_data.update(ImgEmbedding_train)
        #     np.save(npy_file, existing_data)
        # else:
        #     np.save(npy_file, ImgEmbedding_train)

        # Write the processed ID to the file
        with open('processed_img_test_ids.txt', 'a') as file:
            file.write(image_id_hatexplain + '\n')

    except Exception as e:
        print(f"Error processing image: {image_id_hatexplain}")
        print(e)
        pass

print("Image processing for hatexplain completed.")