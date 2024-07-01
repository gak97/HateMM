from collections import deque
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from encoders_new import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet
from hateim_data_loader_new import get_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MMIM(nn.Module):
    def __init__(self):
        super(MMIM, self).__init__()
        self.text_encoder = LanguageEmbeddingLayer()
        self.visual_encoder = RNNEncoder(768, 16, 16)
        self.acoustic_encoder = RNNEncoder(768, 16, 16)
        
        self.mi_tv = MMILB(768, 16)
        self.mi_ta = MMILB(768, 16)
        self.mi_va = MMILB(16, 16)
        
        self.cpc_zt = CPC(768, 128)
        self.cpc_zv = CPC(16, 128)
        self.cpc_za = CPC(16, 128)
        
        self.fusion_prj = SubNet(768, 128, 2, 0.0)

    def forward(self, is_train, bert_sent, bert_sent_type, bert_sent_mask, visual, audio, y=None, mem=None):
        # print(f"Text: {bert_sent.shape}, Visual: {visual.shape}, Audio: {audio.shape}, Labels: {y.shape}")
        batch_size = bert_sent.size(0)
        encoded_text = self.text_encoder(bert_sent, bert_sent_type, bert_sent_mask)
        text_enc = encoded_text
        visual_enc = self.visual_encoder(visual)
        
        audio = audio.unsqueeze(1)
        audio_enc = self.acoustic_encoder(audio)

        # print(f"Text: {text_enc.shape}, Visual: {visual_enc.shape}, Audio: {audio_enc.shape}")

        lld_tv, tv_pn, H_tv = self.mi_tv(text_enc, visual_enc, y, mem['tv'] if mem else None)
        lld_ta, ta_pn, H_ta = self.mi_ta(text_enc, audio_enc, y, mem['ta'] if mem else None)
        lld_va, va_pn, H_va = self.mi_va(visual_enc, audio_enc, y, mem['va'] if mem else None)

        fusion, preds = self.fusion_prj(text_enc)
        # print(f"Fusion: {fusion.shape}, Preds: {preds.shape}")
        
        nce_t = self.cpc_zt(text_enc, fusion)
        nce_v = self.cpc_zv(visual_enc, fusion)
        nce_a = self.cpc_za(audio_enc, fusion)
        nce = nce_t + nce_v + nce_a
        nce = torch.where(torch.isnan(nce), torch.zeros_like(nce), nce)

        pn_dic = {'hate_tv': tv_pn, 'hate_ta': ta_pn, 'hate_va': va_pn}
        lld = lld_tv + lld_ta + lld_va 
        H = H_tv + H_ta + H_va
        return lld, nce, preds, pn_dic, H

class HateMMSolver:
    def __init__(self, train_loader, dev_loader, test_loader, model=None):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model = MMIM().to(device)

        # Parallelize model to multiple GPUs
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)

        self.mem_size = 1
        self.mem_hate_tv = deque(maxlen=self.mem_size)
        self.mem_hate_ta = deque(maxlen=self.mem_size)
        self.mem_hate_va = deque(maxlen=self.mem_size)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_main = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.optimizer_mmilb = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, patience=5, factor=0.5)
        self.scheduler_mmilb = ReduceLROnPlateau(self.optimizer_mmilb, patience=5, factor=0.5)

    def train(self, epoch, stage=1):
        self.model.train()
        total_loss = 0.0
        predictions = []
        true_labels = []
        
        for i_batch, batch in enumerate(self.train_loader):
            if batch is None:
                continue

            self.optimizer_main.zero_grad()
            self.optimizer_mmilb.zero_grad()  # Zero gradients for both optimizers

            input_ids, token_type_ids, attention_mask, visual, audio, y = [x.to(device) for x in batch]
            batch_size = input_ids.size(0)

            # Get memory for each modality pair
            mem_tv = self.mem_hate_tv if i_batch >= self.mem_size else None
            mem_ta = self.mem_hate_ta if i_batch >= self.mem_size else None
            mem_va = self.mem_hate_va if i_batch >= self.mem_size else None

            lld, nce, preds, pn_dic, H = self.model(is_train=True, bert_sent=input_ids, 
                                                    bert_sent_type=token_type_ids, bert_sent_mask=attention_mask, 
                                                    visual=visual, audio=audio, y=y, 
                                                    mem={'tv': mem_tv, 'ta': mem_ta, 'va': mem_va})

            if y.dtype == torch.long:
                y = y.squeeze(1)
                loss_task = self.criterion(preds, y)
            else:
                raise ValueError(f"Expected y to be of type torch.long, but got {y.dtype}")

            # Calculate individual loss components
            loss_nce = 0.3 * nce
            loss_lld = -0.3 * lld
            loss_entropy = -0.1 * H if i_batch > self.mem_size else 0.0 # Entropy loss

            # Combine losses 
            loss = loss_task + loss_nce + loss_lld + loss_entropy

            loss.backward()
            self.optimizer_main.step()
            self.optimizer_mmilb.step()

            # Update memories
            if pn_dic['hate_tv'] is not None:  # Check if positive samples are returned
                self.mem_hate_tv.append(pn_dic['hate_tv'].detach())
            if pn_dic['hate_ta'] is not None:
                self.mem_hate_ta.append(pn_dic['hate_ta'].detach())
            if pn_dic['hate_va'] is not None:
                self.mem_hate_va.append(pn_dic['hate_va'].detach())

            predictions.extend(preds.argmax(dim=1).cpu().numpy())
            true_labels.extend(y.cpu().numpy())

            # Backpropagate the main loss (for stage 1)
            # if stage == 1:
            #     loss.backward() 
            #     self.optimizer_main.step() 
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        print(f"Train Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
        
        return avg_loss
    
    def evaluate(self, test=False):
        self.model.eval()
        loader = self.test_loader if test else self.dev_loader
        total_loss = 0.0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in loader:
                if batch is None:
                    continue
                input_ids, token_type_ids, attention_mask, visual, audio, y = [x.to(device) for x in batch]
                batch_size = input_ids.size(0)
                
                outputs = self.model(False, input_ids, token_type_ids, attention_mask, visual, audio, y, None)
                _, _, preds, _, _ = outputs
                
                if y.dtype == torch.long:
                    y = y.squeeze(1)
                    loss = self.criterion(preds, y)
                else:
                    raise ValueError("Expected y to be of type torch.long, but got {}".format(y.dtype))
                
                total_loss += loss.item() * batch_size
                predictions.extend(preds.argmax(dim=1).cpu().numpy())
                true_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(loader.dataset)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        
        print(f"{'Test' if test else 'Validation'} Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}")
        
        return avg_loss

    def train_and_eval(self):
        best_val_loss = float('inf')
        
        for epoch in tqdm(range(3)):
            train_loss = self.train(epoch, stage=0)
            # train_loss = self.train(epoch, stage=1)
            val_loss = self.evaluate()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
                
            self.scheduler_main.step(val_loss)
            self.scheduler_mmilb.step(val_loss)

        self.model.load_state_dict(torch.load('best_model.pt'))
        self.evaluate(test=True)
            
        return best_val_loss

if __name__ == "__main__":
    train_loader, dev_loader, test_loader = get_loader()
    solver = HateMMSolver(train_loader, dev_loader, test_loader)
    solver.train_and_eval()