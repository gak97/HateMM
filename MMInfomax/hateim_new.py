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
        self.add_va = True

    def forward(self, is_train, bert_sent, bert_sent_type, bert_sent_mask, visual, audio, y=None, mem=None):
        # print(f"Text: {bert_sent.shape}, Visual: {visual.shape}, Audio: {audio.shape}, Labels: {y.shape}")
        batch_size = bert_sent.size(0)
        encoded_text = self.text_encoder(bert_sent, bert_sent_type, bert_sent_mask)
        text_enc = encoded_text
        visual_enc = self.visual_encoder(visual)
        
        audio = audio.unsqueeze(1)
        audio_enc = self.acoustic_encoder(audio)

        # print(f"Text: {text_enc.shape}, Visual: {visual_enc.shape}, Audio: {audio_enc.shape}")

        # lld_tv, tv_pn, H_tv = self.mi_tv(text_enc, visual_enc, y, mem['tv'] if mem else None)
        # lld_ta, ta_pn, H_ta = self.mi_ta(text_enc, audio_enc, y, mem['ta'] if mem else None)
        # lld_va, va_pn, H_va = self.mi_va(visual_enc, audio_enc, y, mem['va'] if mem else None)
        if y is not None:
            lld_tv, tv_pos_samples, tv_neg_samples, H_tv = self.mi_tv(text_enc, visual_enc, y, mem['tv'] if mem else None)
            lld_ta, ta_pos_samples, ta_neg_samples, H_ta = self.mi_ta(text_enc, audio_enc, y, mem['ta'] if mem else None)

            if self.add_va:
                lld_va, va_pos_samples, va_neg_samples, H_va = self.mi_va(visual_enc, audio_enc, y, mem['va'] if mem else None)
        else:
            lld_tv, tv_pos_samples, tv_neg_samples, H_tv = self.mi_tv(text_enc, visual_enc)
            lld_ta, ta_pos_samples, ta_neg_samples, H_ta = self.mi_ta(text_enc, audio_enc)

            if self.add_va:
                lld_va, va_pos_samples, va_neg_samples, H_va = self.mi_va(visual_enc, audio_enc)

        fusion, preds = self.fusion_prj(text_enc)
        # print(f"Fusion: {fusion.shape}, Preds: {preds.shape}")
        
        nce_t = self.cpc_zt(text_enc, fusion)
        nce_v = self.cpc_zv(visual_enc, fusion)
        nce_a = self.cpc_za(audio_enc, fusion)
        nce = nce_t + nce_v + nce_a
        nce = torch.where(torch.isnan(nce), torch.zeros_like(nce), nce)

        pn_dic = {'tv': {'pos': tv_pos_samples, 'neg': tv_neg_samples},
                  'ta': {'pos': ta_pos_samples, 'neg': ta_neg_samples},
                  'va': {'pos': va_pos_samples, 'neg': va_neg_samples} if self.add_va else None}
        lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)
        H = H_tv + H_ta + (H_va if self.add_va else 0.0)

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
        self.mem_tv = deque(maxlen=self.mem_size)
        self.mem_ta = deque(maxlen=self.mem_size)
        self.mem_va = deque(maxlen=self.mem_size)
        self.best_val_loss = float('inf')
        
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

        # Multi-GPU support
        # if torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs!")
        #     self.model = nn.DataParallel(self.model)

        for i_batch, batch in enumerate(self.train_loader):
            if batch is None:
                continue

            input_ids, token_type_ids, attention_mask, visual, audio, y = [x.to(device) for x in batch]
            batch_size = input_ids.size(0)

            # Get memory for each modality pair (only for stage 1)
            if stage == 1 and i_batch >= self.mem_size:
                mem = {'tv': self.mem_tv, 'ta': self.mem_ta, 'va': self.mem_va}
            else:
                mem = None

            # Forward pass
            lld, nce, preds, pn_dic, H = self.model(is_train=True, bert_sent=input_ids, 
                                                    bert_sent_type=token_type_ids, bert_sent_mask=attention_mask, 
                                                    visual=visual, audio=audio, y=y, mem=mem)

            # Handle DataParallel output
            # if isinstance(lld, tuple): # If using DataParallel
            #     lld = lld.mean()
            #     nce = nce.mean()
            #     preds = torch.cat([o for o in preds]) # Concatenate predictions from different GPUs
            #     H = H.mean()
            
            # Calculate loss based on stage
            if stage == 0:
                loss = -lld  
            else:
                if y.dtype == torch.long:
                    y = y.squeeze(1)
                    loss_task = self.criterion(preds, y) 
                else:
                    raise ValueError(f"Expected y to be of type torch.long, but got {y.dtype}")
                loss = loss_task + 0.1 * nce - 0.1 * lld 
                if i_batch > self.mem_size:
                    loss -= 0.1 * H

            loss.backward()

            # Optimizer step based on stage
            if stage == 0:
                self.optimizer_mmilb.step()
            else:
                self.optimizer_main.step()

            # Update memories (only for stage 1)
            if stage == 1:
                if pn_dic['tv']['pos'] is not None:
                    self.mem_tv.append(pn_dic['tv']['pos'].detach())
                if pn_dic['ta']['pos'] is not None:
                    self.mem_ta.append(pn_dic['ta']['pos'].detach())
                if pn_dic['va']['pos'] is not None:
                    self.mem_va.append(pn_dic['va']['pos'].detach())
                if pn_dic['tv']['neg'] is not None:
                    self.mem_tv.append(pn_dic['tv']['neg'].detach())
                if pn_dic['ta']['neg'] is not None:
                    self.mem_ta.append(pn_dic['ta']['neg'].detach())
                if pn_dic['va']['neg'] is not None:
                    self.mem_va.append(pn_dic['va']['neg'].detach())

            total_loss += loss.item() * batch_size
            predictions.extend(preds.argmax(dim=1).cpu().numpy())
            true_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader.dataset)
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)

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
        for epoch in tqdm(range(10)): 
            # Stage 0: Maximize MI
            train_loss = self.train(epoch, stage=0)
            print(f"Epoch {epoch} Stage 0 Train Loss: {train_loss:.4f}") 

            # Stage 1: Train with all losses
            train_loss = self.train(epoch, stage=1)
            print(f"Epoch {epoch} Stage 1 Train Loss: {train_loss:.4f}") 

            val_loss = self.evaluate()
            print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}") 

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')

            self.scheduler_main.step(val_loss)
            self.scheduler_mmilb.step(val_loss)

        self.model.load_state_dict(torch.load('best_model.pt'))
        self.evaluate(test=True)

        return self.best_val_loss

if __name__ == "__main__":
    train_loader, dev_loader, test_loader = get_loader()
    solver = HateMMSolver(train_loader, dev_loader, test_loader)
    solver.train_and_eval()