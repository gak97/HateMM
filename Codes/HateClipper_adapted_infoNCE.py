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
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MemeDataset(Dataset):
    def __init__(self, jsonl_file, processor):
        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor
        self.images = {}
        for row in tqdm(self.data, desc="Loading images"):
            # image_id = row['img']
            image_id = row['img'].replace('img/', 'img_masked/')
            # image_id = row['img'].replace('img/', 'img_inpainted/')
            image_path = f'/backup/girish_datasets/Hateful_Memes_Extended/{image_id}'
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
        # image_id = row['img']
        image_id = row['img'].replace('img/', 'img_masked/')
        # image_id = row['img'].replace('img/', 'img_inpainted/')

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
        outer_product = torch.bmm(x1.unsqueeze(2), x2.unsqueeze(1)) # [batch_size, input_dim1, input_dim2]
        outer_product = outer_product.view(outer_product.size(0), -1) # [batch_size, input_dim1 * input_dim2]
        output = self.fc(outer_product) # [batch_size, output_dim]
        return output


def collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out None samples
    if not batch:
        # return torch.tensor([]), torch.tensor([])  # Return empty tensors if batch is empty
        return None

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


class CLIPClassifier(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", dropout_rate=0.1, projection_dim=768, num_pre_output_layers=3):
        super(CLIPClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.text_projection = nn.Linear(self.clip_model.config.text_config.hidden_size, projection_dim)
        self.image_projection = nn.Linear(self.clip_model.config.vision_config.hidden_size, projection_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.bilinear_pooling = BilinearPooling(projection_dim, projection_dim, projection_dim)
        
        # Pre-output layers
        pre_output_layers = [nn.Linear(projection_dim, projection_dim), nn.ReLU()]
        for _ in range(num_pre_output_layers - 1):
            pre_output_layers.extend([nn.Linear(projection_dim, projection_dim), nn.ReLU()])
        self.pre_output = nn.Sequential(*pre_output_layers)
        
        self.output = nn.Linear(projection_dim, 1)
        # self.temperature = nn.Parameter(torch.ones([]) * 0.07)

        # Learnable temperature parameter
        self.log_temperature = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, pixel_values, attention_mask=None):
        text_features = self.clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        image_features = self.clip_model.vision_model(pixel_values=pixel_values).pooler_output
        
        text_features = self.text_projection(text_features)
        image_features = self.image_projection(image_features)
        
        text_features = F.normalize(text_features, p=2, dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)
        
        # Cross fusion
        fused_features = self.bilinear_pooling(text_features, image_features)
        
        # Pre-output layers
        pre_output = self.pre_output(fused_features)
        pre_output = self.dropout(pre_output)
        
        # Final output
        logits = self.output(pre_output).squeeze(-1)

        return logits, text_features, image_features

def create_mask(batch_size, num_classes, labels, device):
    # N, M = text_features.shape[0], image_features.shape[0]
    # mask = torch.full((N, M), -1, device=text_features.device)
    mask = torch.full((batch_size, num_classes), -1, device=device)
    labels = labels.long()
    
    # Set 0 for all items in the batch
    mask[:, :num_classes] = 0

    # Set 1 for the chosen item
    mask[torch.arange(batch_size), labels] = 1

    return mask

def contextual_infonce_loss(logits, text_features, image_features, labels, log_temperature):
    batch_size = logits.shape[0]
    device = logits.device

    mask = torch.eye(batch_size, device=device)     # identity matrix for self-comparison
    temperature = torch.exp(log_temperature)
    # print(f"Temperature: {temperature}")
    
    # Compute all pairwise cosine similarities
    similarities = torch.matmul(text_features, image_features.t()) / temperature
    
    similarities = similarities * mask      # masking self-comparisons

    exp_similarities = torch.exp(similarities)      # masked self-comparisons
    denominator = exp_similarities.sum(dim=1)     # sum over the masked self-comparisons
    infonce_loss = -torch.log(exp_similarities.diag() / denominator).mean()     # log-likelihood of the diagonal elements (self-comparisons)

    bce_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
    total_loss = bce_loss + infonce_loss

    return total_loss

def train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, text_features, image_features = model(input_ids, pixel_values, attention_mask)
            loss = contextual_infonce_loss(logits, text_features, image_features, labels, model.module.log_temperature)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
        # wandb.log({"Train Loss": round(avg_loss, 4)})
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch, labels in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                logits, text_features, image_features = model(input_ids, pixel_values, attention_mask)
                loss = contextual_infonce_loss(logits, text_features, image_features, labels, model.module.log_temperature)
                val_loss += loss.item()

                preds = torch.sigmoid(logits) >= 0.5

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_preds)

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
        # wandb.log({"Validation Loss": round(avg_val_loss, 4), "Validation Accuracy": round(accuracy, 4), "Validation Precision": round(precision, 4), 
        #            "Validation Recall": round(recall, 4), "Validation F1": round(f1, 4), "Validation ROC AUC": round(auc, 4)})
        
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "clip_classifier.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == 3:
                print("Early stopping!")
                break
    
    return model

if __name__ == "__main__":
    # Load your datasets as before
    train_data = "/backup/girish_datasets/Hateful_Memes_Extended/train.jsonl"
    val_data = "/backup/girish_datasets/Hateful_Memes_Extended/dev_seen.jsonl"
    test_data = "/backup/girish_datasets/Hateful_Memes_Extended/test_seen.jsonl"

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    train_dataset = MemeDataset(train_data, processor)
    val_dataset = MemeDataset(val_data, processor)
    test_dataset = MemeDataset(test_data, processor)

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = CLIPClassifier(num_pre_output_layers=3)
    model.to(device)

    # Calculate the number of parameters being fine-tuned
    num_finetune_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters being fine-tuned: {num_finetune_params / 1e6:.2f}M")

    # Parallelize model to multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    num_epochs = 15
    learning_rate = 2e-5

#     wandb.init(
#     project="hate-memes-classification",
#     config={
#         "learning_rate": learning_rate,
#         "architecture": "CLIP (Proj Finetuning) + Adaptive InfoNCE Loss",
#         "dataset": "Hateful Memes",
#         "epochs": num_epochs,
#         "batch_size": batch_size,
#     },
# )

    model = train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate)

    # Evaluation
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch, labels in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            logits, text_features, image_features = model(input_ids, pixel_values, attention_mask)
            loss = contextual_infonce_loss(logits, text_features, image_features, labels, model.module.log_temperature)
            test_loss += loss.item()
            
            # Compute predictions
            preds = torch.sigmoid(logits) >= 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_test_loss = test_loss / len(test_dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0, average='binary')
    recall = recall_score(all_labels, all_preds, zero_division=0, average='binary')
    f1 = f1_score(all_labels, all_preds, zero_division=0, average='binary')
    auc = roc_auc_score(all_labels, all_preds)

    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    # wandb.log({"Test Loss": round(avg_test_loss, 4), "Test Accuracy": round(accuracy, 4), "Test Precision": round(precision, 4), 
    #            "Test Recall": round(recall, 4), "Test F1": round(f1, 4), "Test ROC AUC": round(auc, 4)})
    # wandb.finish()
