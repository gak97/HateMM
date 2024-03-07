import torch
import torch.nn as nn
import pickle
from torch.utils import data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import wandb

from datasets import load_dataset
dataset = load_dataset('limjiayi/hateful_memes_expanded')

FOLDER_NAME = '/backup/hatemm/Dataset/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Text_Model(nn.Module):
    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Linear(fc2_hidden, output_size),
        )
    def forward(self, xb):
        return self.network(xb)

class Image_Model(nn.Module):
    def __init__(self, input_size, fc1_hidden, fc2_hidden, output_size):
        super().__init__()
        self.network=nn.Sequential(
            nn.Linear(input_size, fc1_hidden),
            nn.ReLU(),
            nn.Linear(fc1_hidden, fc2_hidden),
            nn.ReLU(),
            nn.Linear(fc2_hidden, output_size),
        )
    def forward(self, xb):
        return self.network(xb)

class Combined_model(nn.Module):
    def __init__(self, text_model, image_model, num_classes):
        super().__init__()
        self.text_model = text_model
        self.image_model = image_model
        self.fc_output   = nn.Linear(128, num_classes)

    def forward(self, x_text, x_img):
        if x_text is not None:
            tex_out = self.text_model(x_text)
        else:
            tex_out = torch.zeros(x_img.size(0), 64).to(x_img.device) if x_img is not None else torch.zeros(x_text.size(0), 64).to(x_text.device)

        if x_img is not None:
            img_out = self.video_model(x_img)
        else:
            img_out = torch.zeros(x_text.size(0), 64).to(x_text.device) if x_text is not None else torch.zeros(x_img.size(0), 64).to(x_img.device)

        inp = torch.cat((tex_out, img_out), dim = 1)
        out = self.fc_output(inp)
        return out

class Dataset_ViT(data.Dataset):
    def __init__(self, image, labels, split='train'):
        "Initialization"
        self.labels = labels
        self.image_id = image
        self.split = split
    
    def load_data_for_image(self, image_id, split):
        # Load text and image data
        if split == 'train':
            text_data = torch.tensor(TextEmbedding_train[image_id])
            image_data = torch.tensor(ImgEmbedding_train[image_id])
        elif split == 'val':
            text_data = torch.tensor(TextEmbedding_val[image_id])
            image_data = torch.tensor(ImgEmbedding_val[image_id])
        else:
            text_data = torch.tensor(TextEmbedding_test[image_id])
            image_data = torch.tensor(ImgEmbedding_test[image_id])

        return text_data, image_data

    def __getitem__(self, image_id):
        "Generates one sample of data"

        # Load data
        X_text, X_img = self.load_data_for_image(image_id)

        # Load label
        y = self.labels[image_id]

        return X_text, X_img, y


import dill
with open(FOLDER_NAME + 'new_hatefulmemes_train_VITembedding.pkl', 'rb') as fp:
    ImgEmbedding_train = dill.load(fp)
# with open(FOLDER_NAME + 'new_hatefulmemes_train_VITembedding.pkl', 'rb') as fp:
#     ImgEmbedding_train = []
#     while True:
#         try:
#             ImgEmbedding_train.append(pickle.load(fp))
# # ImgEmbedding_train = pickle.load(fp)
#         except EOFError:
#             break

with open(FOLDER_NAME + 'all_hatefulmemes_train_rawBERTembedding.pkl', 'rb') as fp:
    TextEmbedding_train = pickle.load(fp)

with open(FOLDER_NAME + 'all_hatefulmemes_validation_rawBERTembedding.pkl', 'rb') as fp:
    TextEmbedding_val = pickle.load(fp)

with open(FOLDER_NAME + 'hatefulmemes_validation_VITembedding.pkl', 'rb') as fp:
    ImgEmbedding_val = pickle.load(fp)

with open(FOLDER_NAME + 'all_hatefulmemes_test_rawBERTembedding.pkl', 'rb') as fp:
    TextEmbedding_test = pickle.load(fp)

with open(FOLDER_NAME + 'hatefulmemes_test_VITembedding.pkl', 'rb') as fp:
    ImgEmbedding_test = pickle.load(fp)


def eval_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average='weighted')
    
    return accuracy, f1, precision, recall, roc_auc


def collate_fn(batch):
    text, image, label = zip(*batch)
    text = torch.stack(text)
    image = torch.stack(image)
    label = torch.tensor(label)
    return text, image, label

input_text_size = 768
input_image_size = 768
fc1_hidden = 128
fc2_hidden = 128

# training parameters
num_classes = 2
learning_rate = 1e-4
num_epochs = 2
batch_size = 16


wandb.init(
    project="hate-memes-classification",
    config={
        "learning_rate": learning_rate,
        "architecture": "BERT + ViT",
        "dataset": "Hateful Memes Extended",
        "epochs": num_epochs,
        "batch_size": batch_size,
    },
)

ext_data = {}

# DataLoaders
for split in dataset:
    labels = dataset[split]['label']
    image_ids = dataset[split]['id']
    ext_data[split] = Dataset_ViT(image_ids, labels, split)
    # ext_data[split] = Dataset_ViT(ImgEmbedding_train, TextEmbedding_train, dataset[split])

train_loader = data.DataLoader(ext_data['train'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = data.DataLoader(ext_data['val'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = data.DataLoader(ext_data['test'], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Model
text_model = Text_Model(input_text_size, fc1_hidden, fc2_hidden, num_classes).to(device)
image_model = Image_Model(input_image_size, fc1_hidden, fc2_hidden, num_classes).to(device)

model = Combined_model(text_model, image_model, num_classes).to(device)

 # Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for i, (text, image, labels) in enumerate(train_loader):
            train_y_true = []
            train_y_pred = []
            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(text, image)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_y_true.extend(labels.cpu().numpy())
            train_y_pred.extend(predicted.cpu().numpy())

        train_loss /= len(train_loader)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            val_y_true = []
            val_y_pred = []
            for text, image, labels in val_loader:
                text = text.to(device)
                image = image.to(device)
                labels = labels.to(device)

                outputs = model(text, image)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)

        wandb.log({"Train Loss": train_loss, "Validation Loss": val_loss, 
                   "Epoch": epoch + 1, "Train Accuracy": eval_metrics(train_y_true, train_y_pred)[0], 
                   "Validation Accuracy": eval_metrics(val_y_true, val_y_pred)[0],
                   "Train ROC AUC": eval_metrics(train_y_true, train_y_pred)[4], "Validation ROC AUC": eval_metrics(val_y_true, val_y_pred)[4]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer)

# Test the model
def test_model(model, test_loader, criterion):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for text, image, labels in test_loader:
            text = text.to(device)
            image = image.to(device)
            labels = labels.to(device)

            outputs = model(text, image)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

        accuracy, f1, precision, recall, roc_auc = eval_metrics(y_true, y_pred)
        wandb.log({"Test Accuracy": accuracy, "Test F1": f1, "Test Precision": precision, "Test Recall": recall, "Test ROC AUC": roc_auc})

        print(f'Test Accuracy: {accuracy:.4f}, Test F1: {f1:.4f}, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}, Test ROC AUC: {roc_auc:.4f}')

test_model(model, test_loader, criterion)