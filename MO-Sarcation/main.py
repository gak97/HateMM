import pickle
import torch
import numpy as np
from torch import nn
from torch.utils import data
from transformers import BartTokenizerFast
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from multimodal_bart_downstream import MultimodalBartForSequenceClassification

FOLDER_NAME = '/backup/hatemm/Dataset/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 1e-4

SOURCE_MAX_LEN = 768 # 500
ACOUSTIC_DIM = 768
ACOUSTIC_MAX_LEN = 1000
VISUAL_DIM = 768 # 2048
VISUAL_MAX_LEN = 1000 # 480


with open(FOLDER_NAME + 'all__video_vosk_audioMap.pkl', 'rb') as f:
  transcript = pickle.load(f)

with open(FOLDER_NAME + 'Wav2Vec2_features_chunked.pkl', 'rb') as fo:
  audio_data = pickle.load(fo)

with open(FOLDER_NAME + 'noFoldDetails.pkl', 'rb') as fp:
    video_labels = pickle.load(fp)

def pad_seq(tensor, dim, max_len):
  if max_len > tensor.shape[0] :
    return torch.cat([tensor, torch.zeros(max_len - tensor.shape[0], dim)])
  else:
    return tensor[:max_len]
  

model = MultimodalBartForSequenceClassification.from_pretrained("facebook/bart-base")

tokenizer = BartTokenizerFast.from_pretrained('facebook/bart-base')
# print("Tokenizer : ", tokenizer)

num_param = sum(p.numel() for p in model.parameters())
# print("Total parameters : ", num_param/1e6)


p = {
        # 'additional_special_tokens' : ['[CONTEXT]', '[UTTERANCE]']
        'additional_special_tokens' : ['[UTTERANCE]']
    }

tokenizer.add_special_tokens(p)
# print("Tokenizer after adding special tokens : ", tokenizer)

# print(model.resize_token_embeddings(len(tokenizer)))


class HateMMDataset(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, folders, labels):
        "Initialization"
        # print(folders, labels)
        self.labels = labels
        self.folders = folders

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def load_data_for_video(self, video):
        video_file_name_without_extension, _ = os.path.splitext(video)
        pickle_file_path = os.path.join(FOLDER_NAME, "VITF_new", video_file_name_without_extension + "_vit.p")
        
        # Load text data
        if video in transcript:
            # text_features = torch.tensor(np.array(transcript[video]), dtype=torch.float32)
            text_features = tokenizer(transcript[video], max_length = SOURCE_MAX_LEN, padding = 'max_length', truncation = True)
        else:
            raise ValueError(f"Text data not found for {video}")
        
        # Load video data
        try:
            with open(pickle_file_path, 'rb') as fp:
                video_data = pickle.load(fp)
                video_features = torch.tensor(np.array(list(video_data.values())), dtype=torch.float32)
                # video_features = torch.tensor(np.array(list(video_data.values().mean(dim=0))), dtype=torch.float32)
                video_features = video_features.mean(dim=0)
        except FileNotFoundError:
            raise ValueError(f"Video data file not found: {pickle_file_path}")
        
        # Load audio data
        if video in audio_data:
            audio_features = torch.tensor(np.array(audio_data[video]), dtype=torch.float32)
            audio_features = audio_features.mean(dim=0)
            # audio_features, _ = audio_features.max(dim=0)
        else:
            raise ValueError(f"Audio data not found for {video}")
        
        return text_features, video_features, audio_features

    def __getitem__(self, index):
        "Generates one sample of data"
        try:
            # Select sample
            folder = self.folders[index]
            # Load data
            X_text, X_vid, X_audio = self.load_data_for_video(folder)
            y = torch.LongTensor([self.labels[index]]) 
            
            # return X_text, X_vid, X_audio, y
            return torch.tensor(X_text['input_ids'], dtype=torch.long), torch.tensor(X_text['attention_mask'], dtype=torch.bool), X_audio, X_vid, y
        
        except Exception as e:
            # traceback.print_exc()
            print(f"Error loading data for index {index}: {e}")                        
            return None, None, None, None, None
        

all_train_data, all_train_label = video_labels['train']
all_val_data, all_val_label = video_labels['val']
all_test_data, all_test_label = video_labels['test']

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:  # Check if the batch is empty after filtering
        return None

    return torch.utils.data.dataloader.default_collate(batch)

# training parameters
k = 2            # number of target category
epochs = 20
batch_size = 32
log_interval = 100

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
valParams = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}

train_set, val_set, test_set = HateMMDataset(all_train_data, all_train_label), HateMMDataset(all_val_data, all_val_label), HateMMDataset(all_test_data, all_test_label)
train_loader = data.DataLoader(train_set, collate_fn = collate_fn, **params)
test_loader = data.DataLoader(test_set, collate_fn = collate_fn, **valParams)
valid_loader = data.DataLoader(val_set, collate_fn = collate_fn, **valParams)

model.to(DEVICE)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()


def train_epoch(model, data_loader):
      model.train()
      epoch_train_loss = 0.0


      for step, batch in enumerate(tqdm(data_loader, desc = 'Training Iteration')):
        # for i, t in enumerate(batch):
        #     print("Inside hello")
        #     print(i, " : ", type(t))
        # batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, attention_mask, acoustic_input, visual_input, labels = batch
        print("Input IDs shape:", input_ids.shape)
        print("Attention Mask shape:", attention_mask.shape)
        print("Audio shape:", acoustic_input.shape)
        print("Video shape:", visual_input.shape)
        print("Labels shape:", labels.shape)

        input_ids, attention_mask, acoustic_input, visual_input, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), acoustic_input.to(DEVICE), visual_input.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        # print("Input ids shape : ", input_ids.shape)
        # print("Input ids shape : ", input_ids.shape)
        outputs = model(input_ids = input_ids,
                        attention_mask = attention_mask,
                        # context_input_ids = context_input_ids,
                        # context_attention_mask = context_attention_mask,
                        acoustic_input = acoustic_input,
                        visual_input = visual_input,
                        labels = labels)

        loss = outputs['loss']
        epoch_train_loss += loss.item()

        # print("Batch wise loss : ", epoch_train_loss)

        loss.backward()
        optimizer.step()

      print("Epoch train loss : ", epoch_train_loss)


def valid_epoch(model, data_loader):
  model.eval()
  predictions = []
  gold = []

  valid_loss = 0.0
  with torch.no_grad():
    for step, batch in enumerate(tqdm(data_loader)):
      # batch = tuple(t.to(DEVICE) for t in batch)
      input_ids, attention_mask, acoustic_input, visual_input, labels = batch
      input_ids, attention_mask, acoustic_input, visual_input, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), acoustic_input.to(DEVICE), visual_input.to(DEVICE), labels.to(DEVICE)

      outputs = model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            # context_input_ids = context_input_ids,
                            # context_attention_mask = context_attention_mask,
                            acoustic_input = acoustic_input,
                            visual_input = visual_input,
                            labels = labels)

      logits = outputs['logits']
      loss = outputs['loss']

      valid_loss += loss.item()



      pred = logits.argmax(dim = -1)

      predictions.extend(pred.tolist())
      gold.extend(labels.tolist())

  return valid_loss, predictions, gold


def test_epoch(model, data_loader):
    model.eval()
    predictions = []
    gold = []

    correct = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader)):
            # batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, acoustic_input, visual_input, labels = batch
            input_ids, attention_mask, acoustic_input, visual_input, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), acoustic_input.to(DEVICE), visual_input.to(DEVICE), labels.to(DEVICE)

            outputs = model(input_ids = input_ids,
                            attention_mask = attention_mask,
                            # context_input_ids = context_input_ids,
                            # context_attention_mask = context_attention_mask,

                            acoustic_input = acoustic_input,
                            visual_input = visual_input,
                            labels = labels)

            logits = outputs['logits']

            pred = logits.argmax(dim = -1)

            predictions.extend(pred.tolist())

            gold.extend(labels.tolist())

            correct += int((pred == labels).sum())

    return correct/len(data_loader.dataset), predictions, gold


class EarlyStopping:
  def __init__(self, patience, min_delta):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation = np.inf

  def early_stop(self, valid_loss):
    if valid_loss < self.min_validation:
      self.min_validation = valid_loss
      self.counter = 0
    elif valid_loss > (self.min_validation + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False
  
early_stopper = EarlyStopping(patience = 15, min_delta = 0.2)


def train_and_validation(model, train_loader, valid_loader):
  # lowest_loss = 1e6
  best_f1 = 0.0
  # min_loss = 1e6
  for epoch in range(3):
    print("\n=============Epoch : ", epoch)
    train_epoch(model, train_loader)
    valid_loss, valid_pred, valid_gold = valid_epoch(model, valid_loader)

    if early_stopper.early_stop(valid_loss):
      break

    print("Length of predictions : ", len(valid_pred))
    print("Length of gold : ", len(valid_gold))
    print("Valid loss : ", valid_loss)
    print("\n Valid Accuracy : ", accuracy_score(valid_gold, valid_pred))
    print("\n Valid Precision : ", precision_score(valid_gold, valid_pred, average = 'weighted'))
    print("\n Valid Recall : ", recall_score(valid_gold, valid_pred, average = 'weighted'))
    print("\nValid F1 score : ", f1_score(valid_gold, valid_pred, average = 'weighted'))


    curr_f1 = f1_score(valid_gold, valid_pred, average = 'weighted')

    curr_loss = valid_loss
    # if((curr_f1 > best_f1) and (epoch>=4)):
    if(curr_f1 > best_f1):
    # if(curr_loss < min_loss):
    # if(curr_loss < lowest_loss):
      best_f1 = curr_f1
      # min_loss = curr_loss
      # print("Valid pred : ", valid_pred)
      # print('valid_gold : ', valid_gold)
      # torch.save(model.state_dict(), '/content/drive/MyDrive/Colab Notebooks/32/saved_model/best_case/best_model_epoch_'+str(epoch)+'_best_f1_'+str(int(best_f1*100))+'_foldNum_'+str(foldNum)+'.pt')
      # print("model saved\n")


train_and_validation(model, train_loader, valid_loader)