from io import BytesIO
import os, requests, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, SiglipProcessor, SiglipModel
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve, roc_curve
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
import wandb

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
            # image_id = row['img']
            image_id = row['img'].replace('img/', 'img_masked/')
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

        if image_id not in self.images:
            print(f"Image {image_id} not available.")
            return None
        
        image = self.images[image_id]

        inputs = self.processor(text=text, images=image, return_tensors="pt", padding="max_length", truncation=True)
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

class SigLIPClassifier(nn.Module):
    def __init__(self, model_name="google/siglip-so400m-patch14-384", dropout_rate=0.1, projection_dim=768):        # google/siglip-large-patch16-384
        super(SigLIPClassifier, self).__init__()
        self.siglip_model = AutoModel.from_pretrained(model_name)
        # self.siglip_model = SiglipModel.from_pretrained(model_name, attn_implementation="sdpa", torch_dtype=torch.float16, device_map=device)
        for param in self.siglip_model.parameters():
            param.requires_grad = False

        self.text_projection = nn.Linear(self.siglip_model.config.text_config.hidden_size, projection_dim)
        self.image_projection = nn.Linear(self.siglip_model.config.vision_config.hidden_size, projection_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.bilinear_pooling = BilinearPooling(projection_dim, projection_dim, projection_dim)
        self.pre_output1 = nn.Linear(projection_dim, projection_dim)
        self.pre_output2 = nn.Linear(projection_dim, projection_dim)
        self.pre_output3 = nn.Linear(projection_dim, projection_dim)
        self.output = nn.Linear(projection_dim, 1)

    def forward(self, input_ids, pixel_values):
        text_features = self.siglip_model.get_text_features(input_ids=input_ids)
        image_features = self.siglip_model.get_image_features(pixel_values=pixel_values)

        text_features = self.text_projection(text_features)
        image_features = self.image_projection(image_features)
        
        text_features = F.normalize(text_features, p=2, dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)

        combined_features = self.bilinear_pooling(text_features, image_features)

        pre_output = self.pre_output1(combined_features)
        pre_output = F.relu(pre_output)
        pre_output = self.pre_output2(pre_output)
        pre_output = F.relu(pre_output)
        pre_output = self.pre_output3(pre_output)
        pre_output = self.dropout(pre_output)
        logits = self.output(pre_output).squeeze(1)

        return logits

    def calculate_metrics(self, logits, labels, threshold=0.5):
        preds = torch.sigmoid(logits)
        preds_binary = (preds >= threshold).long()
        
        accuracy = accuracy_score(labels.cpu(), preds_binary.cpu())
        precision = precision_score(labels.cpu(), preds_binary.cpu(), zero_division=0)
        recall = recall_score(labels.cpu(), preds_binary.cpu(), zero_division=0)
        f1 = f1_score(labels.cpu(), preds_binary.cpu(), zero_division=0)
        auc = roc_auc_score(labels.cpu(), preds.cpu()) if len(set(labels.cpu().numpy())) > 1 else float('nan')
        
        return accuracy, precision, recall, f1, auc

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    input_ids = torch.stack([item[0]['input_ids'] for item in batch])
    pixel_values = torch.stack([item[0]['pixel_values'] for item in batch])
    labels = torch.stack([item[1] for item in batch])

    return {'input_ids': input_ids, 'pixel_values': pixel_values}, labels

def optimal_threshold(y_true, y_pred):
    # precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    # f1_scores = 2 * (precision * recall) / (precision + recall)
    # best_threshold = thresholds[np.argmax(f1_scores)]
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    gm = np.sqrt(tpr * (1-fpr))
    best_threshold = thresholds[np.argmax(gm)]
    print(f"Optimal Threshold: {best_threshold}")

    return best_threshold

def train_and_evaluate(model, train_dataloader, val_dataloaders, test_dataloaders, optimizer, criterion, scheduler, num_epochs, device):
    best_val_auc = 0
    patience = 3
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if inputs is None:
                continue
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(**inputs)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss/len(train_dataloader):.4f}")
        # wandb.log({"Train Loss": round(total_loss/len(train_dataloader), 4)})

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        for val_dataloader in val_dataloaders:
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    if inputs is None:
                        continue
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    labels = labels.to(device)

                    logits = model(**inputs)
                    loss = criterion(logits, labels.float())
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    val_loss += loss.item()
        
        # opt_threshold = optimal_threshold(torch.tensor(all_labels), torch.tensor(all_preds))
        accuracy, precision, recall, f1, auc = model.module.calculate_metrics(torch.tensor(all_preds), torch.tensor(all_labels), threshold=0.6)
        print(f"Validation Loss: {val_loss/len(val_dataloaders):.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        # wandb.log({"Validation Loss": round(val_loss/len(val_dataloaders), 4), "Validation Accuracy": round(accuracy, 4), "Validation Precision": round(precision, 4), 
        #        "Validation Recall": round(recall, 4), "Validation F1": round(f1, 4), "Validation ROC AUC": round(auc, 4)})

        scheduler.step(auc)
        if auc > best_val_auc:
            best_val_auc = auc
            torch.save(model.state_dict(), 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print("Early stopping")
                break

    # Test Evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    all_preds = []
    all_labels = []
    for test_dataloader in test_dataloaders:
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                logits = model(**inputs)
                all_preds.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # opt_threshold = optimal_threshold(torch.tensor(all_labels), torch.tensor(all_preds))
    accuracy, precision, recall, f1, auc = model.module.calculate_metrics(torch.tensor(all_preds), torch.tensor(all_labels), threshold=0.6)
    print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}")
    # wandb.log({"Test Accuracy": round(accuracy, 4), "Test Precision": round(precision, 4), "Test Recall": round(recall, 4), "Test F1": round(f1, 4), "Test ROC AUC": round(auc, 4)})
    # wandb.finish()

def main():
    batch_size = 64
    learning_rate = 2e-5
    num_epochs = 5

    train_data = "/backup/girish_datasets/Hateful_Memes_Extended/train.jsonl"
    val_data = ["/backup/girish_datasets/Hateful_Memes_Extended/dev_seen.jsonl"]
    # val_data = ["/backup/girish_datasets/Hateful_Memes_Extended/dev_seen.jsonl", "/backup/girish_datasets/Hateful_Memes_Extended/test_seen.jsonl"]
    test_data = ["/backup/girish_datasets/Hateful_Memes_Extended/test_seen.jsonl"]
    # test_data = ["/backup/girish_datasets/Hateful_Memes_Extended/dev_unseen.jsonl", "/backup/girish_datasets/Hateful_Memes_Extended/test_unseen.jsonl"]

    # Initialize SigLIP processor and model
    processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    # processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    # processor = AutoProcessor.from_pretrained("google/siglip-large-patch16-384")
    
    # Prepare datasets and dataloaders
    train_dataset = MemeDataset(train_data, processor)
    val_datasets = [MemeDataset(data, processor) for data in val_data]
    test_datasets = [MemeDataset(data, processor) for data in test_data]

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) for dataset in val_datasets]
    test_dataloaders = [DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) for dataset in test_datasets]

    # wandb.init(
    #     project="hate-memes-classification",
    #     config={
    #         "learning_rate": learning_rate,
    #         "architecture": "SigLIP Text + SigLIP Image (Proj Finetuning)",
    #         "dataset": "Hateful Memes",
    #         "epochs": num_epochs,
    #         "batch_size": batch_size,
    #     },
    # )

    # Initialize model, optimizer, and loss function
    model = SigLIPClassifier()
    model.to(device)

    # Calculate the number of parameters being fine-tuned
    num_finetune_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters being fine-tuned: {num_finetune_params / 1e6:.2f}M")

    # Parallelize model if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    # Calculate class weights for weighted loss
    train_labels = [row['label'] for row in train_dataset.data]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    total_counts = sum(class_counts)
    weights = [total_counts / class_counts[i] for i in range(len(class_counts))]
    pos_weight = torch.tensor([weights[1] / weights[0]], device=device)
    print(f"Positive Weight: {pos_weight}")

    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

    # Train and evaluate the model
    train_and_evaluate(model, train_dataloader, val_dataloaders, test_dataloaders, optimizer, criterion, scheduler, num_epochs, device)

if __name__ == "__main__":
    main()