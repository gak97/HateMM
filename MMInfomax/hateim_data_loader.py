from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import pickle
import pandas as pd

# path to a pretrained word embedding file
word_emb_path = '/home/diptesh/workspace/glove/glove.840B.300d.txt'

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

class HateMMDataset(Dataset):
    def __init__(self, folder_name):
        with open(folder_name + 'noFoldDetails.pkl', 'rb') as fp:
            self.labels = pickle.load(fp)
        with open(folder_name + 'all__video_vosk_audioMap.pkl', 'rb') as fb:
            self.transcript = pickle.load(fb)
        with open(folder_name + 'MFCCFeaturesNew.pkl', 'rb') as fc:
            self.audData = pickle.load(fc)
        with open(folder_name + 'final_allVideos.pkl', 'rb') as fd:
            self.allDataAnnotation = pickle.load(fd)
            self.allVidList = list(self.allDataAnnotation.values())

        # self.df = pd.read_csv(self.labels)
        # self.df['label'] = self.df['label'].apply(lambda x: 1 if x == 'Hate' else 0)

    def __len__(self):
        return len(self.transcript)

    def __getitem__(self, index):
        text = self.transcript[index]
        visual = self.allVidList[index]
        audio = self.audData[index]
        label = torch.tensor(self.df['label'][index], dtype=torch.float32)
        return text, visual, audio, label

def get_loader(folder_name, batch_size, shuffle=True):
    dataset = HateMMDataset(folder_name)

    def collate_fn(batch):
        texts, visuals, audios, labels = zip(*batch)
        # Process texts for BERT
        bert_sentences = []
        bert_sentence_types = []
        bert_sentence_att_masks = []
        for text in texts:
            encoded_bert_sent = bert_tokenizer.encode_plus(
                text, max_length=50, add_special_tokens=True, truncation=True, padding='max_length')
            bert_sentences.append(encoded_bert_sent['input_ids'])
            bert_sentence_types.append(encoded_bert_sent['token_type_ids'])
            bert_sentence_att_masks.append(encoded_bert_sent['attention_mask'])

        # Convert lists to tensors
        texts = torch.LongTensor(bert_sentences)
        visuals = torch.stack(visuals)
        audios = torch.stack(audios)
        labels = torch.stack(labels)

        return texts, visuals, audios, labels

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader