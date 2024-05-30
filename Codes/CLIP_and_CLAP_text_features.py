FOLDER_NAME = '/backup/hatemm/Dataset/'

import os
import pickle
from datasets import load_dataset
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
# from transformers import AutoTokenizer, ClapTextModel

# model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# model = ClapTextModel.from_pretrained("laion/clap-htsat-unfused")
# tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

model.to('cuda')

dataset = load_dataset('limjiayi/hateful_memes_expanded')

# with open(FOLDER_NAME+'all_whisper_tiny_transcripts.pkl','rb') as fp:
#     transCript = pickle.load(fp)

# allEmbedding = {}
# for i in tqdm(transCript):
#     try:
#         apr = tokenizer(transCript[i], return_tensors='pt', padding=True, truncation=True)
#         apr = {k: v.to('cuda') for k, v in apr.items()}
#         with torch.no_grad():
#             outputs = model(**apr)
#             last_hidden_states = outputs.last_hidden_state
#             allEmbedding[i] = last_hidden_states[0][0].cpu().detach().numpy()
#     except Exception as e:
#         print(f"Error processing text with ID: {i}. Skipping this sample.")
#         continue

# with open(FOLDER_NAME + 'all_hatemm_clip_embedding_truncated.pkl', 'wb') as fp:
#     pickle.dump(allEmbedding, fp)


# Load processed IDs from a file
processed_hxp_ids = set()
try:
    with open('processed_text_ids.txt', 'r') as file:
        for line in file:
            processed_hxp_ids.add(line.strip())
except FileNotFoundError:
    pass

# Load skipped sample IDs from the file
skipped_samples = set()
with open('skipped_samples.txt', 'r') as file:
    for line in file:
        skipped_samples.add(line.strip())

print("Starting processing for CLIP text...")
split = 'train'
# for split in ['train', 'validation', 'test']:
print(f"Processing split: {split}")
allEmbedding_hatexplain = {}
for example in tqdm(dataset[split]):
    try:
        text_id_hatexplain = example['id']
        if text_id_hatexplain in skipped_samples:
            print(f"Skipping text with ID: {text_id_hatexplain}")
            continue
        if text_id_hatexplain in processed_hxp_ids:
            print(f"Skipping text with ID: {text_id_hatexplain}")
            continue

        text = example['text']
        # print(f"Processing text: {text}")
        inputs = tokenizer(text, return_tensors='pt', padding=True)
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with torch.no_grad():
            # embeddings = model(inputs['input_ids'], inputs['attention_masks'])[2][0].detach().numpy()  
            outputs = model(**inputs)        
            # allEmbedding_hatexplain[example['id']] = embeddings
            allEmbedding_hatexplain[example['id']] = outputs.text_embeds[0].cpu().detach().numpy()
            # last_hidden_states = outputs.last_hidden_state
            # allEmbedding_hatexplain[example['id']] = last_hidden_states[0].cpu().detach().numpy()
            # allEmbedding_hatexplain[example['id']] = last_hidden_states[0][0].detach().numpy()

        # Check if the pickle file already exists and has content
        pickle_file = FOLDER_NAME + f'all_hatememes_ext_{split}_clip_proj_embedding.pkl'
        if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
            with open(pickle_file, 'rb') as fp:
                existing_data = pickle.load(fp)
        else:
            existing_data = {}

        # Update the existing data
        existing_data.update(allEmbedding_hatexplain)

        # Save the updated data to the pickle file
        with open(pickle_file, 'wb') as fp:
            pickle.dump(existing_data, fp)

        # Update processed IDs set and save to file for each text example
        if text_id_hatexplain not in processed_hxp_ids:
            processed_hxp_ids.add(text_id_hatexplain)
            with open('processed_text_ids.txt', 'a') as file:
                file.write(text_id_hatexplain + '\n')

    except Exception as e:
        print(f"Error processing text with ID: {text_id_hatexplain}. Skipping this sample.")
        with open('skipped_samples.txt', 'a') as file:
            file.write(f"{text_id_hatexplain}\n")
        continue

# with open(FOLDER_NAME + f'all_hatefulmemes_{split}_hatexplain_embedding.p', 'wb') as fp:
#     pickle.dump(allEmbedding_hatexplain, fp)

print(f"Finished processing split: {split}")