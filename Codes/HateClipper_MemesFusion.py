from io import BytesIO
import requests
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MemeDataset(Dataset):
    def __init__(self, dataset, processor, transform=None):
        self.dataset = dataset
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text = row['text']
        image_id = row['id']
        if image_id.endswith('.png') or image_id.endswith('.jpg'):
            image_id = image_id
        else:
            image_id += '.png'
        # image = Image.open(row['image_path']).convert('RGB')
        image_url = f"https://huggingface.co/datasets/limjiayi/hateful_memes_expanded/resolve/main/img/{image_id}"
        # response = requests.get(image_url)
        image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')

        # try:
        #     if response.status_code == 200:
        #         image_bytes = response.content
        #         image = Image.open(BytesIO(image_bytes)).convert('RGB')
        #     else:
        #         raise Exception(f"Failed to download image from {image_url}")
        # except Exception as e:
        #     print(e)
        #     return None, None

        # if self.transform:
        #     image = self.transform(image)

        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension

        label = torch.tensor(row['label'], dtype=torch.float)
        return inputs, label

class CLIPClassifier(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", dropout_rate=0.2, image_weight=0.5, text_weight=0.5):
        super(CLIPClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        # self.image_output = nn.Linear(self.clip_model.vision_model.config.hidden_size, 1)
        # self.text_output = nn.Linear(self.clip_model.text_model.config.hidden_size, 1)
        self.image_output = nn.Linear(512, 1)
        self.text_output = nn.Linear(512, 1)
        self.image_weight = image_weight
        self.text_weight = text_weight
    
    def forward(self, input_ids, pixel_values, attention_mask=None):
        outputs = self.clip_model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        
        # Print shapes for debugging
        # print(f"Image features shape: {image_features.shape}")
        # print(f"Text features shape: {text_features.shape}")

        image_features = self.dropout(image_features)
        text_features = self.dropout(text_features)
        
        image_logits = self.image_output(image_features).squeeze(1)
        text_logits = self.text_output(text_features).squeeze(1)

        # Combine logits from both modalities
        combined_logits = (image_logits + text_logits) / 2
        
        return image_logits, text_logits, combined_logits
    
    def calculate_loss(self, image_logits, text_logits, labels):
        image_loss = F.binary_cross_entropy_with_logits(image_logits, labels)
        text_loss = F.binary_cross_entropy_with_logits(text_logits, labels)
        total_loss = self.image_weight * image_loss + self.text_weight * text_loss
        return total_loss

    def calculate_metrics(self, combined_logits, labels):
        # image_preds = torch.sigmoid(image_logits)
        # text_preds = torch.sigmoid(text_logits)
        combined_preds = torch.sigmoid(combined_logits)     # Convert to probabilities
        
        # image_preds_binary = (image_preds >= 0.5).long()
        # text_preds_binary = (text_preds >= 0.5).long()
        combined_preds_binary = (combined_preds >= 0.5).long()  # Convert to binary predictions
        
        # image_accuracy = accuracy_score(labels.cpu(), image_preds_binary.cpu())
        # text_accuracy = accuracy_score(labels.cpu(), text_preds_binary.cpu())
        
        accuracy = accuracy_score(labels.cpu(), combined_preds_binary.cpu())
        precision = precision_score(labels.cpu(), combined_preds_binary.cpu(), zero_division=0)
        recall = recall_score(labels.cpu(), combined_preds_binary.cpu(), zero_division=0)
        f1 = f1_score(labels.cpu(), combined_preds_binary.cpu(), zero_division=0)
        auc = roc_auc_score(labels.cpu(), combined_preds.cpu()) if len(set(labels.cpu().numpy())) > 1 else float('nan')

        # image_auc = roc_auc_score(labels.cpu(), image_preds.cpu()) if len(set(labels.cpu().numpy())) > 1 else float('nan')
        # text_auc = roc_auc_score(labels.cpu(), text_preds.cpu()) if len(set(labels.cpu().numpy())) > 1 else float('nan')
        
        return accuracy, precision, recall, f1, auc
        # return (image_accuracy, text_accuracy), (image_auc, text_auc)

def collate_fn(batch):
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
dataset = load_dataset('limjiayi/hateful_memes_expanded')
train_data = dataset['train'].select(list(range(8500)))
val_data = dataset['validation'].select(list(range(500)))
test_data = dataset['test'].select(list(range(1000)))

# Prepare dataset and dataloader
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = MemeDataset(train_data, processor, transform)
val_dataset = MemeDataset(val_data, processor, transform)
test_dataset = MemeDataset(test_data, processor, transform)

batch_size = 32
learning_rate = 1e-4

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Instantiate model, optimizer, and loss function
model = CLIPClassifier()
model.to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 20  # Set the number of epochs

wandb.init(
    project="hate-memes-classification",
    config={
        "learning_rate": 1e-4,
        "architecture": "CLIP Text + CLIP Image (Finetuning)",
        "dataset": "Hateful Memes",
        "epochs": num_epochs,
        "batch_size": batch_size,
    },
)

for epoch in tqdm(range(num_epochs)):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_dataloader:
        input_ids = inputs['input_ids'].to(device)
        pixel_values = inputs['pixel_values'].to(device)
        attention_mask = inputs.get('attention_mask', None).to(device) if 'attention_mask' in inputs else None
        labels = labels.to(device)
        
        optimizer.zero_grad()
        image_logits, text_logits, combined_logits = model(input_ids, pixel_values, attention_mask)
        loss = model.module.calculate_loss(image_logits, text_logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_dataloader)}")
    # accuracy, precision, recall, f1, auc = model.module.calculate_metrics(combined_logits, labels)
    wandb.log({"Train Loss": total_loss/len(train_dataloader)})

    # Validation
    model.eval()
    val_loss = 0.0
    all_image_preds = []
    all_text_preds = []
    all_combined_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            input_ids = inputs['input_ids'].to(device)
            pixel_values = inputs['pixel_values'].to(device)
            labels = labels.to(device)
            
            # image_logits, text_logits = model(input_ids, pixel_values)
            # (image_accuracy, text_accuracy), (image_auc, text_auc) = model.module.calculate_metrics(image_logits, text_logits, labels)
            # image_preds = torch.sigmoid(image_logits)
            # text_preds = torch.sigmoid(text_logits)
            
            # all_image_preds.extend(image_preds.cpu().numpy())
            # all_text_preds.extend(text_preds.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())

            image_logits, text_logits, combined_logits = model(input_ids, pixel_values)
            loss = model.module.calculate_loss(image_logits, text_logits, labels)
            all_combined_preds.extend(torch.sigmoid(combined_logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            val_loss += loss.item()
    
    accuracy, precision, recall, f1, auc = model.module.calculate_metrics(torch.tensor(all_combined_preds), torch.tensor(all_labels))
    print(f"Validation Loss: {val_loss}, Validation Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}")
    wandb.log({"Validation Loss": val_loss, "Validation Accuracy": accuracy, "Validation Precision": precision, 
               "Validation Recall": recall, "Validation F1": f1, "Validation ROC AUC": auc})

torch.save(model.state_dict(), 'clip_classifier.pth')

    # image_accuracy, text_accuracy = accuracy_score(all_labels, (np.array(all_image_preds) >= 0.5).astype(int)), accuracy_score(all_labels, (np.array(all_text_preds) >= 0.5).astype(int))
    # image_auc, text_auc = roc_auc_score(all_labels, all_image_preds), roc_auc_score(all_labels, all_text_preds)
    # print(f"Validation Image Accuracy: {image_accuracy}, AUC: {image_auc}")
    # print(f"Validation Text Accuracy: {text_accuracy}, AUC: {text_auc}")

# Test Evaluation
model.eval()
all_image_preds = []
all_text_preds = []
all_combined_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_dataloader:
        input_ids = inputs['input_ids'].to(device)
        pixel_values = inputs['pixel_values'].to(device)
        labels = labels.to(device)
        
        # image_logits, text_logits = model(input_ids, pixel_values)
        # image_preds = torch.sigmoid(image_logits)
        # text_preds = torch.sigmoid(text_logits)
        
        # all_image_preds.extend(image_preds.cpu().numpy())
        # all_text_preds.extend(text_preds.cpu().numpy())
        # all_labels.extend(labels.cpu().numpy())

        _, _, combined_logits = model(input_ids, pixel_values)
        all_combined_preds.extend(torch.sigmoid(combined_logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy, precision, recall, f1, auc = model.module.calculate_metrics(torch.tensor(all_combined_preds), torch.tensor(all_labels))
print(f"Test Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}, AUC: {auc}")
wandb.log({"Test Accuracy": accuracy, "Test Precision": precision, "Test Recall": recall, 
           "Test F1": f1, "Test ROC AUC": auc})

# image_accuracy, text_accuracy = accuracy_score(all_labels, (np.array(all_image_preds) >= 0.5).astype(int)), accuracy_score(all_labels, (np.array(all_text_preds) >= 0.5).astype(int))
# image_auc, text_auc = roc_auc_score(all_labels, all_image_preds), roc_auc_score(all_labels, all_text_preds)
# print(f"Test Image Accuracy: {image_accuracy}, AUC: {image_auc}")
# print(f"Test Text Accuracy: {text_accuracy}, AUC: {text_auc}")
