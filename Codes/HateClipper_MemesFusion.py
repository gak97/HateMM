from io import BytesIO
import os, requests, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MemeDataset(Dataset):
    def __init__(self, jsonl_file, processor, transform=None):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor
        self.images = {}
        for row in tqdm(self.data, desc="Loading images"):
            image_id = row['img']
            image_path = f'/backup/hatemm/Hateful_Memes_Extended/{image_id}'
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                self.images[image_id] = image
            else:
                print(f"Image file {image_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        text = row['text']
        image_id = row['img']

        if image_id not in self.images:
            print(f"Image {image_id} not available.")
            return None  # Skip this sample
        
        image = self.images[image_id]

        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension

        label = torch.tensor(row['label'], dtype=torch.float)
        return inputs, label

class BilinearPooling(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(BilinearPooling, self).__init__()
        self.fc = nn.Linear(input_dim1 * input_dim2, output_dim)
    
    def forward(self, x1, x2):
        outer_product = torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1))
        outer_product = outer_product.view(outer_product.size(0), -1)
        output = self.fc(outer_product)
        return output

class CLIPClassifier(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", dropout_rate=0.2, projection_dim=768, fusion_type='cross'):   # openai/clip-vit-base-patch32
        super(CLIPClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.text_projection = nn.Linear(768, projection_dim)
        self.image_projection = nn.Linear(768, projection_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fusion_type = fusion_type

        if fusion_type == 'concat':
            self.pre_output = nn.Linear(projection_dim * 2, projection_dim)
        elif fusion_type == 'cross':
            self.bilinear_pooling = BilinearPooling(projection_dim, projection_dim, projection_dim)
            self.pre_output = nn.Linear(projection_dim, projection_dim)
        else:
            self.pre_output = nn.Linear(projection_dim, projection_dim)

        self.output = nn.Linear(projection_dim, 1)

    def forward(self, input_ids, pixel_values, attention_mask=None):
        outputs = self.clip_model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        text_features = self.text_projection(outputs.text_embeds)
        image_features = self.image_projection(outputs.image_embeds)
        
        text_features = F.normalize(text_features, p=2, dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)

        if self.fusion_type == 'align':
            combined_features = text_features * image_features
        elif self.fusion_type == 'concat':
            combined_features = torch.cat((text_features, image_features), dim=1)
        elif self.fusion_type == 'cross':
            combined_features = self.bilinear_pooling(text_features, image_features)
        elif self.fusion_type == 'align_shuffle':
            combined_features = text_features * image_features
            idx = torch.randperm(combined_features.size(0))
            combined_features = combined_features[idx]

        combined_features = self.dropout(combined_features)
        pre_output = self.pre_output(combined_features)
        pre_output = self.dropout(pre_output)
        logits = self.output(pre_output).squeeze(1)

        return logits, text_features, image_features

    def calculate_loss(self, logits, labels, text_features, image_features):
        main_loss = F.binary_cross_entropy_with_logits(logits, labels)
        text_loss = F.mse_loss(text_features, labels.unsqueeze(1).expand_as(text_features))
        image_loss = F.mse_loss(image_features, labels.unsqueeze(1).expand_as(image_features))
        return main_loss + 0.5 * text_loss + 0.5 * image_loss

    # def calculate_loss(self, logits, labels):
    #     return F.binary_cross_entropy_with_logits(logits, labels)

    def calculate_metrics(self, logits, labels):
        preds = torch.sigmoid(logits)  # Convert to probabilities
        preds_binary = (preds >= 0.5).long()  # Convert to binary predictions
        
        accuracy = accuracy_score(labels.cpu(), preds_binary.cpu())
        precision = precision_score(labels.cpu(), preds_binary.cpu(), zero_division=0)
        recall = recall_score(labels.cpu(), preds_binary.cpu(), zero_division=0)
        f1 = f1_score(labels.cpu(), preds_binary.cpu(), zero_division=0)
        auc = roc_auc_score(labels.cpu(), preds.cpu()) if len(set(labels.cpu().numpy())) > 1 else float('nan') # AUC is not defined when there is only one class
        
        return accuracy, precision, recall, f1, auc

def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out None samples
    if not batch:
        return torch.tensor([]), torch.tensor([])  # Return empty tensors if batch is empty
    
    input_ids = [item[0]['input_ids'] for item in batch]
    pixel_values = [item[0]['pixel_values'] for item in batch]
    labels = torch.stack([item[1] for item in batch])

    # Pad input_ids
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)

    # Stack pixel_values
    pixel_values = torch.stack(pixel_values)

    # Ensure all attention masks are the same size as input_ids
    attention_masks = [torch.ones_like(id) for id in input_ids]
    attention_mask_padded = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # return {'input_ids': input_ids, 'pixel_values': pixel_values}, labels
    return {'input_ids': input_ids_padded, 'pixel_values': pixel_values, 'attention_mask': attention_mask_padded}, labels

# Load dataset from Hugging Face
# dataset = load_dataset('limjiayi/hateful_memes_expanded')
# train_data = dataset['train']
# val_data = dataset['validation']
# test_data = dataset['test']

train_data = "/backup/hatemm/Hateful_Memes_Extended/train.jsonl"
val_data = ["/backup/hatemm/Hateful_Memes_Extended/dev_seen.jsonl", "/backup/hatemm/Hateful_Memes_Extended/test_seen.jsonl"]
test_data = ["/backup/hatemm/Hateful_Memes_Extended/dev_unseen.jsonl", "/backup/hatemm/Hateful_Memes_Extended/test_unseen.jsonl"]

# Prepare dataset and dataloader
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")  
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

train_dataset = MemeDataset(train_data, processor)
# val_dataset = MemeDataset(val_data, processor)
# test_dataset = MemeDataset(test_data, processor)
val_datasets = [MemeDataset(data, processor) for data in val_data]
test_datasets = [MemeDataset(data, processor) for data in test_data]

batch_size = 64
learning_rate = 1e-4

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
val_dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) for dataset in val_datasets]
test_dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) for dataset in test_datasets]

# Instantiate model, optimizer, and loss function
model = CLIPClassifier(fusion_type='cross')
model.to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 10  # Set the number of epochs

# wandb.init(
#     project="hate-memes-classification",
#     config={
#         "learning_rate": 1e-4,
#         "architecture": "CLIP Text + CLIP Image (Proj Finetuning)",
#         "dataset": "Hateful Memes",
#         "epochs": num_epochs,
#         "batch_size": batch_size,
#     },
# )

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_dataloader:
        input_ids = inputs['input_ids'].to(device)
        pixel_values = inputs['pixel_values'].to(device)
        attention_mask = inputs.get('attention_mask', None).to(device) if 'attention_mask' in inputs else None
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, text_features, image_features = model(input_ids, pixel_values, attention_mask)
        loss = model.module.calculate_loss(logits, labels, text_features, image_features)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_dataloader)}")
    # accuracy, precision, recall, f1, auc = model.module.calculate_metrics(combined_logits, labels)
    # wandb.log({"Train Loss": total_loss/len(train_dataloader)})

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    for val_dataloader in val_dataloaders:
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                input_ids = inputs['input_ids'].to(device)
                pixel_values = inputs['pixel_values'].to(device)
            labels = labels.to(device)

            logits, text_features, image_features = model(input_ids, pixel_values)
            loss = model.module.calculate_loss(logits, labels, text_features, image_features)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            val_loss += loss.item()
    
    accuracy, precision, recall, f1, auc = model.module.calculate_metrics(torch.tensor(all_preds), torch.tensor(all_labels))
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}")
    # wandb.log({"Validation Loss": val_loss, "Validation Accuracy": accuracy, "Validation Precision": precision, 
    #            "Validation Recall": recall, "Validation F1": f1, "Validation ROC AUC": auc})

torch.save(model.state_dict(), 'clip_classifier.pth')

# Test Evaluation
model.eval()
all_preds = []
all_labels = []
for test_dataloader in test_dataloaders:
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            input_ids = inputs['input_ids'].to(device)
            pixel_values = inputs['pixel_values'].to(device)
            labels = labels.to(device)

            logits, text_features, image_features = model(input_ids, pixel_values)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

accuracy, precision, recall, f1, auc = model.module.calculate_metrics(torch.tensor(all_preds), torch.tensor(all_labels))
print(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}")
# wandb.log({"Test Accuracy": accuracy, "Test Precision": precision, "Test Recall": recall, 
#            "Test F1": f1, "Test ROC AUC": auc})
