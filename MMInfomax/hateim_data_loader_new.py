import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

FOLDER_NAME = '/backup/hatemm/Dataset/'
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

with open(FOLDER_NAME + 'noFoldDetails.pkl', 'rb') as fp:
    allDataAnnotation = pickle.load(fp)

with open(FOLDER_NAME + 'all_whisper_tiny_transcripts.pkl', 'rb') as f:
    transcript = pickle.load(f)

with open(FOLDER_NAME + 'CLAP_features.pkl', 'rb') as fo:
    audio_data = pickle.load(fo)

class HateMMDataset(Dataset):
    def __init__(self, folders, labels):
        self.labels = labels
        self.folders = folders

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        try:
            folder = self.folders[index]
            video_file_name_without_extension, _ = os.path.splitext(folder)
            pickle_file_path = os.path.join(FOLDER_NAME, "VITF_new", video_file_name_without_extension + "_vit.pkl")
            
            if folder in transcript:
                text_features = bert_tokenizer.encode_plus(transcript[folder], max_length=512, add_special_tokens=True, truncation=True, padding='max_length')
            else:
                raise ValueError(f"Text data not found for {folder}")
            
            try:
                with open(pickle_file_path, 'rb') as fp:
                    video_data = pickle.load(fp)
                    video_features = torch.tensor(list(video_data.values()), dtype=torch.float32).mean(dim=1)
            except FileNotFoundError:
                raise ValueError(f"Video data file not found: {pickle_file_path}")
            
            if folder in audio_data:
                audio_features = torch.tensor(audio_data[folder], dtype=torch.float32).reshape(-1, 768).mean(dim=0)
            else:
                raise ValueError(f"Audio data not found for {folder}")
            
            y = torch.LongTensor([self.labels[index]])
            
            return torch.tensor(text_features['input_ids'], dtype=torch.long), torch.tensor(text_features['token_type_ids'], dtype=torch.long), torch.tensor(text_features['attention_mask'], dtype=torch.bool), video_features, audio_features, y

        except Exception as e:
            print(f"Error loading data for index {index}: {e}")
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def get_loader():
    all_train_data, all_train_label = allDataAnnotation['train']
    all_val_data, all_val_label = allDataAnnotation['val'] 
    all_test_data, all_test_label = allDataAnnotation['test']
    
    train_set = HateMMDataset(all_train_data, all_train_label)
    val_set = HateMMDataset(all_val_data, all_val_label)
    test_set = HateMMDataset(all_test_data, all_test_label)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn) 
    valid_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader