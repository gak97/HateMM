import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, AutoModel


FOLDER_NAME = '/backup/hatemm/Dataset/'

import pickle, os
# with open(FOLDER_NAME+'all__video_vosk_audioMap.p','rb') as fp:
#     transCript = pickle.load(fp)


from datasets import load_dataset
dataset = load_dataset('limjiayi/hateful_memes_expanded')


from models import *

model = Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two")



class Text_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=Model_Rational_Label.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two", output_hidden_states = True)
        
    def forward(self,x,mask):
        embeddings = self.model(x, mask)
        return embeddings


tokenizer1 = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two")
def tokenize(sentences, padding = True, max_len = 512):
    input_ids, attention_masks, token_type_ids = [], [], []
    for sent in sentences:
        encoded_dict = tokenizer1.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=max_len, 
                                                    padding='max_length', 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt', 
                                                    truncation = True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return {'input_ids': input_ids, 'attention_masks': attention_masks}


model2 = Text_Model()


# from tqdm import tqdm
# allEmbedding ={}
# for i in tqdm(transCript):
#   try:
#     apr = tokenize([transCript[i]])
#     with torch.no_grad():
#         allEmbedding[i]= (model2(apr['input_ids'], apr['attention_masks'])[2][0]).detach().numpy()
#     del(apr)
#   except:
#     pass


# len(allEmbedding)
# with open(FOLDER_NAME+'all_HateXPlainembedding.p', 'wb') as fp:
#     pickle.dump(allEmbedding,fp)


# Load processed IDs from a file
processed_hxp_ids = set()
try:
    with open('processed_hxp_ids.txt', 'r') as file:
        for line in file:
            processed_hxp_ids.add(line.strip())
except FileNotFoundError:
    pass

print("Starting processing for hatexplain...")
split = 'train'
# for split in ['train', 'val', 'test']:
print(f"Processing split: {split}")
allEmbedding_hatexplain = {}
for example in dataset[split]:
    text_id_hatexplain = example['id']
    if text_id_hatexplain in processed_hxp_ids:
        print(f"Skipping text with ID: {text_id_hatexplain}")
        continue

    text = example['text']
    print(f"Processing text: {text}")
    inputs = tokenize(text)
    with torch.no_grad():
        embeddings = model2(inputs['input_ids'], inputs['attention_masks'])[2][0].detach().numpy()
        allEmbedding_hatexplain[example['id']] = embeddings
        # last_hidden_states = outputs.last_hidden_state
        # allEmbedding_hatexplain[example['id']] = last_hidden_states[0][0].detach().numpy()

    # Check if the pickle file already exists and has content
    pickle_file = FOLDER_NAME + f'all_hatefulmemes_{split}_hatexplain_embedding.p'
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
        with open('processed_hxp_ids.txt', 'a') as file:
            file.write(text_id_hatexplain + '\n')

# with open(FOLDER_NAME + f'all_hatefulmemes_{split}_hatexplain_embedding.p', 'wb') as fp:
#     pickle.dump(allEmbedding_hatexplain, fp)

print(f"Finished processing split: {split}")


tokenizer2 = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


# allEmbedding ={}
# for i in tqdm(transCript):
#   try:
#     inputs = tokenizer(transCript[i], return_tensors="pt", truncation = True, padding='max_length', add_special_tokens=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         last_hidden_states = outputs.last_hidden_state
#         allEmbedding[i]= last_hidden_states[0][0].detach().numpy()
#     del(outputs)
#   except:
#     pass


# len(allEmbedding)
# with open(FOLDER_NAME+'all_rawBERTembedding.p', 'wb') as fp:
#     pickle.dump(allEmbedding,fp)


# Load processed IDs from a file
processed_bert_ids = set()
try:
    with open('processed_bert_ids.txt', 'r') as file:
        for line in file:
            processed_bert_ids.add(line.strip())
except FileNotFoundError:
    pass

print("Starting processing for bert...")
split = 'train'
# for split in ['train', 'val', 'test']:
print(f"Processing split: {split}")
allEmbedding_bert = {}
for example in dataset[split]:
    text_id_bert = example['id']
    if text_id_bert in processed_bert_ids:
        print(f"Skipping text with ID: {text_id_bert}")
        continue

    text = example['text']
    print(f"Processing text: {text}")
    inputs = tokenizer2(text, return_tensors="pt", truncation=True, padding='max_length', add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        allEmbedding_bert[example['id']] = last_hidden_states[0][0].detach().numpy()

    # Check if the pickle file already exists and has content
    pickle_file = FOLDER_NAME + f'all_hatefulmemes_{split}_rawBERTembedding.p'
    if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
        with open(pickle_file, 'rb') as fp:
            existing_data = pickle.load(fp)
    else:
        existing_data = {}

    # Update the existing data
    existing_data.update(allEmbedding_bert)

    # Save the updated data to the pickle file
    with open(pickle_file, 'wb') as fp:
        pickle.dump(existing_data, fp)

    # Update processed IDs set and save to file for each text example
    if text_id_bert not in processed_bert_ids:
        processed_bert_ids.add(text_id_bert)
        with open('processed_bert_ids.txt', 'a') as file:
            file.write(text_id_bert + '\n')

# with open(FOLDER_NAME + f'all_hatefulmemes_{split}_rawBERTembedding.p', 'wb') as fp:
#     pickle.dump(allEmbedding_bert, fp)

print(f"Finished processing split: {split}")
    