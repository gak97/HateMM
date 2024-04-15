import random
import numpy as np
import torch
import pickle
import pandas as pd
import time
from modules.encoders import LanguageEmbeddingLayer, CPC, MMILB, RNNEncoder, SubNet
from utils.tools import to_gpu

import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hateim_data_loader import get_loader

FOLDER_NAME = '/backup/hatemm/Dataset/'

def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(1114)

with open(FOLDER_NAME+'noFoldDetails.pkl', 'rb') as fp:
    labels = pickle.load(fp)

with open(FOLDER_NAME+'all__video_vosk_audioMap.pkl','rb') as fb:
    transcript = pickle.load(fb)

with open(FOLDER_NAME+'MFCCFeaturesNew.pkl','rb') as fc:
    audData = pickle.load(fc)

with open(FOLDER_NAME+'final_allVideos.pkl', 'rb') as fd:
    allDataAnnotation = pickle.load(fd)
    allVidList = list(allDataAnnotation.values())

# df = pd.read_csv(labels)   
# df['label'] = df['label'].apply(lambda x: 1 if x == 'Hate' else 0)
# label = df['label'].values

# divide the dataset into train, val, and test in the ratio of 70:10:20
train_split = int(0.7*len(transcript))
val_split = int(0.1*len(transcript))
test_split = int(0.2*len(transcript))

alpha = 0.5
beta = 0.5
num_epochs = 5


class MMIM(nn.Module):
    def __init__(self):
        super(MMIM, self).__init__()
        # Assuming default values from config.py for initialization
        self.d_vh = 16
        self.d_ah = 16
        self.d_vin = 16
        self.d_ain = 16
        self.d_vout = 16
        self.d_aout = 16
        self.d_tout = 16
        self.n_layer = 1
        self.bidirectional = True
        self.d_prjh = 128
        self.dropout_prj = 0.0

        self.add_va = True
        self.mmilb_last_activation = nn.Tanh()
        self.cpc_activation = nn.Tanh()
        self.cpc_layers = 1

        self.train_changed_pct = 0.5
        self.train_changed_modal = 'language'
        self.train_method = 'missing'

        self.text_encoder = LanguageEmbeddingLayer()
        self.visual_encoder = RNNEncoder(
            input_size=self.d_vin, 
            hidden_size=self.d_vh,
            output_size=self.d_vout,
            num_layers=self.n_layer, 
            dropout=0.0,
            bidirectional=self.bidirectional)
        
        self.acoustic_encoder = RNNEncoder(
            input_size=self.d_ain,
            hidden_size=self.d_ah,
            output_size=self.d_aout,
            num_layers=self.n_layer,
            dropout=0.0,
            bidirectional=self.bidirectional)
        
        self.mi_tv = MMILB(
            x_size=self.d_vout,
            y_size=self.d_aout,
            mid_activation=self.mmilb_last_activation,
            last_activation=self.mmilb_last_activation)
        
        self.mi_ta = MMILB(
            x_size=self.d_vout,
            y_size=self.d_aout,
            mid_activation=self.mmilb_last_activation,
            last_activation=self.mmilb_last_activation)
        
        if self.add_va:
            self.mi_va = MMILB(
                x_size=self.d_vout,
                y_size=self.d_aout,
                mid_activation=self.mmilb_last_activation,
                last_activation=self.mmilb_last_activation)
            
        dim_sum = self.d_vout + self.d_aout + self.d_tout

        # self.visual_rnn = nn.LSTM(input_size=self.d_vh, hidden_size=self.d_vout, num_layers=self.n_layer, bidirectional=self.bidirectional)
        # self.audio_rnn = nn.LSTM(input_size=self.d_ah, hidden_size=self.d_aout, num_layers=self.n_layer, bidirectional=self.bidirectional)
        # self.projection_network = nn.Sequential(
        #     nn.Linear(self.d_vout + self.d_aout, self.d_prjh),
        #     self.mmilb_last_activation
        # )
       
        self.cpc_zt = CPC(
            x_size=self.d_tout,
            y_size=self.d_prjh,
            n_layers=self.cpc_layers,
            activation=self.cpc_activation)
            
        self.cpc_zv = CPC(
            x_size=self.d_vout,
            y_size=self.d_prjh,
            n_layers=self.cpc_layers,
            activation=self.cpc_activation)
        
        self.cpc_za = CPC(
            x_size=self.d_aout,
            y_size=self.d_prjh,
            n_layers=self.cpc_layers,
            activation=self.cpc_activation)
        
        self.fusion = SubNet(
            in_size=self.d_tout,
            hidden_size=self.d_prjh,
            n_class=1,
            dropout=self.dropout_prj)

    def forward(self, is_train, text, visual, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask, y=None, mem=None):
        # visual_output, _ = self.visual_rnn(visual)
        # audio_output, _ = self.audio_rnn(audio)
        # combined_features = torch.cat((visual_output, audio_output), dim=-1)
        # output = self.projection_network(combined_features)
        # return output

        encoded_text = self.text_encoder(text, bert_sent, bert_sent_type, bert_sent_mask)
        text_enc = encoded_text[:.0,:]
        visual_enc = self.visual_encoder(visual, vlens)
        audio_enc = self.acoustic_encoder(audio, alens)

        if is_train:
            pct = self.train_changed_pct
            modal = self.train_changed_modal
            if modal == 'language':
                utterance = text_enc
            elif modal == 'visual':
                utterance = visual_enc
            elif modal == 'acoustic':
                utterance = audio_enc
            else:
                raise ValueError('modal can either be language, visual, or acoustic')

            if self.train_method == 'missing':
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * 0

            elif self.hp.train_method == 'g_noise':   # set modality to Noise
                noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float())
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * noise[i]
            elif self.hp.train_method == 'hybird':   # set half modality to 0, half modality to Noise
                noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float())
                sample_num = int(len(utterance) * pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list_0 = random.sample(sample_list, sample_num)
                sample_list_new = list(set(sample_list).difference(set(sample_list_0)))
                sample_list_N = random.sample(sample_list_new, sample_num)
                for i in sample_list_0:
                    utterance[i] = utterance[i] * 0
                for i in sample_list_N:
                    utterance[i] = utterance[i] * noise[i]
            else:
                raise ValueError('train_method can either be missing, g_noise, or hybird')
            
        if self.is_test:
            test_modal = self.test_modal
            test_pct = self.test_pct
            if test_modal == 'language':
                utterance = text_enc
            elif test_modal == 'visual':
                utterance = visual_enc
            elif test_modal == 'acoustic':
                utterance = audio_enc
            else:
                raise ValueError('modal can either be language, visual, or acoustic')
            
            if self.test_method == 'missing':
                sample_num = int(len(utterance) * test_pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * 0
            elif self.test_method == 'g_noise':   # set modality to Noise
                noise = to_gpu(torch.from_numpy(np.random.normal(0,1,utterance.size()[0])).float())
                sample_num = int(len(utterance) * test_pct)
                sample_list = [i for i in range(len(utterance))]
                sample_list = random.sample(sample_list, sample_num)
                for i in sample_list:
                    utterance[i] = utterance[i] * noise[i]
            else:
                raise ValueError('test_method can either be missing or g_noise')
            
        if y is not None:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text_enc, y=visual_enc, labels=y, mem=mem['tv'])
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text_enc, y=audio_enc, labels=y, mem=mem['ta'])

            # for ablation use
            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual_enc, y=audio_enc, labels=y, mem=mem['va'])
        else:
            lld_tv, tv_pn, H_tv = self.mi_tv(x=text_enc, y=visual_enc)
            lld_ta, ta_pn, H_ta = self.mi_ta(x=text_enc, y=audio_enc)

            if self.add_va:
                lld_va, va_pn, H_va = self.mi_va(x=visual_enc, y=audio_enc)

        fusion, preds = self.fusion_prj(text_enc)

        nce_t = self.cpc_zt(text_enc, fusion)
        nce_v = self.cpc_zv(visual_enc, fusion)
        nce_a = self.cpc_za(audio_enc, fusion)

        nce = nce_t + nce_v + nce_a

        pn_dic = {'hate_tv': tv_pn, 'hate_ta': ta_pn, 'hate_va': va_pn if self.add_va else None}
        lld = lld_tv + lld_ta + (lld_va if self.add_va else 0.0)
        H = H_tv + H_ta + (H_va if self.add_va else 0.0)

        return lld, nce, preds, pn_dic, H

class HateMMSolver(object):
    def __init__(self, train_loader, dev_loader, test_loader, model=None):
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader

        # Model
        if model is None:
            self.model = MMIM()
        else:
            self.model = model

        if torch.cuda.is_available():
            self.model.cuda()

        # Hyperparameters from config.py defaults
        self.batch_size = 32
        self.num_epochs = 40
        self.lr_main = 1e-3
        self.lr_bert = 5e-5
        self.lr_mmilb = 1e-3
        self.alpha = 0.1
        self.beta = 0.1
        self.clip = 1.0
        self.optim = 'Adam'
        self.weight_decay_main = 1e-4
        self.weight_decay_bert = 1e-4
        self.weight_decay_club = 1e-4
        self.patience = 10
        self.update_batch = 1
        self.log_interval = 100
        self.seed = 1111

        # Optimizers
        self.optimizer_mmilb = optim.Adam(self.model.parameters(), lr=self.lr_mmilb, weight_decay=self.weight_decay_club)
        self.optimizer_main = optim.Adam(self.model.parameters(), lr=self.lr_main, weight_decay=self.weight_decay_main)

        # Scheduler
        self.scheduler_mmilb = ReduceLROnPlateau(self.optimizer_mmilb, mode='min', patience=20, factor=0.5, verbose=True)
        self.scheduler_main = ReduceLROnPlateau(self.optimizer_main, mode='min', patience=20, factor=0.5, verbose=True)

        # Criterion
        self.criterion = nn.L1Loss()  # Assuming a regression task, replace with appropriate criterion        

    def train(epoch, model, mem_size, optimizer, criterion, train_loader, stage=1):
        epoch_loss = 0
        model.train()
        num_batches = 16
        proc_loss, proc_size = 0, 0
        nce_loss = 0.0
        ba_loss = 0.0
        start_time = time.time()
        
        mem_hate_tv = []
        mem_hate_av = []
        mem_nonhate_tv = []
        mem_nonhate_av = []

        for i_batch, batch_data in enumerate(train_loader):
            text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data

            model.zero_grad()
            with torch.cuda.device(0):
                text, visual, audio, y, l = text.to(device), visual.to(device), audio.to(device), y.to(device), l.to(device)
                bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)
                # text, visual, audio, y, l = text.float(), visual.float(), audio.float(), y.float(), l.float()
                # bert_sent, bert_sent_type, bert_sent_mask = bert_sent.float(), bert_sent_type.float(), bert_sent_mask.float()

            batch_size = y.size(0)

            if stage == 0:
                y = None
                mem = None
            elif stage == 1 and i_batch >= mem_size:
                mem = {'hate_tv': mem_hate_tv, 'hate_av': mem_hate_av, 
                    'nonhate_tv': mem_nonhate_tv, 'nonhate_av': mem_nonhate_av}
            else:
                mem = None

            is_train = model.training
            lld, nce, preds, pn_dic, H = model(is_train, text, visual, audio, vlens, alens,
                                            bert_sent, bert_sent_type, bert_sent_mask, y, mem)
            if stage == 1:
                y_loss = criterion(preds, y)

                if len(mem_hate_tv) < mem_size:
                    mem_hate_tv.append(pn_dic['hate_tv'].detach())
                    mem_hate_av.append(pn_dic['hate_av'].detach())
                    mem_nonhate_tv.append(pn_dic['nonhate_tv'].detach())
                    mem_nonhate_av.append(pn_dic['nonhate_av'].detach())

                else:
                    oldest = i_batch % mem_size
                    mem_hate_tv[oldest] = pn_dic['hate_tv'].detach()
                    mem_hate_av[oldest] = pn_dic['hate_av'].detach()
                    mem_nonhate_tv[oldest] = pn_dic['nonhate_tv'].detach()
                    mem_nonhate_av[oldest] = pn_dic['nonhate_av'].detach()

                loss = y_loss + alpha * nce - beta * lld

                if i_batch > mem_size:
                    loss -= beta * H
                with autograd.detect_anomaly():
                    loss.backward()
                    # optimizer.step()

            elif stage == 0:
                loss = -lld
                with autograd.detect_anomaly():
                    loss.backward()
                    # optimizer.step()

            else:
                raise ValueError('stage index can either be 0 or 1')
            
            left_batch -= 1
            if left_batch == 0:
                left_batch = num_batches
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()

            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += loss.item() * batch_size
            nce_loss += nce.item() * batch_size
            ba_loss += (-H - lld) * batch_size

            if i_batch % 100 == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed = time.time() - start_time
                avg_nce = nce_loss / proc_size
                avg_ba = ba_loss / proc_size
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | nce {:5.2f} | ba {:5.2f}'.format(
                    epoch, i_batch, len(train_loader), optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / 100, avg_loss, avg_nce, avg_ba))
                
                proc_loss, proc_size = 0, 0
                nce_loss = 0.0
                ba_loss = 0.0
                start_time = time.time()

        return epoch_loss / proc_size

    def evaluate(model, criterion, test_loader=None, dev_loader=None, test=False):
        model.eval()
        loader = test_loader if test else dev_loader
        total_loss = 0.0
        total_l1_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for batch in loader:
                text, visual, vlens, audio, alens, y, bert_sent, bert_sent_type, bert_sent_mask, ids = batch

                with torch.cuda.device(0):
                    text, visual, audio, y, l = text.to(device), visual.to(device), audio.to(device), y.to(device), l.to(device)
                    bert_sent, bert_sent_type, bert_sent_mask = bert_sent.to(device), bert_sent_type.to(device), bert_sent_mask.to(device)
                    text, visual, audio, y, l = text.float(), visual.float(), audio.float(), y.float(), l.float()
                    bert_sent, bert_sent_type, bert_sent_mask = bert_sent.float(), bert_sent_type.float(), bert_sent_mask.float()

                batch_size = y.size(0)

                is_train = model.training
                
                _, _, preds, _, _ = model(is_train, text, visual, audio, vlens, alens, bert_sent, bert_sent_type, bert_sent_mask)

                if test:
                    criterion = nn.L1Loss()
                total_loss += criterion(preds, y).item() * batch_size

                results.append(preds)
                truths.append(y)

        avg_loss = total_loss / len(loader.dataset)
        results = torch.cat(results, dim=0)
        truths = torch.cat(truths, dim=0)

        return avg_loss, results, truths

    def train_and_eval(model, train_loader, dev_loader, test_loader, optimizer_mmilb, optimizer_main, scheduler_main, scheduler_mmilb, criterion, contrast_loss=True):
        mem_size = 1
        best_val_loss = float('inf')
        best_mae = float('inf')

        print("Build Graph:")
        for name, param in model.named_parameters():
            print('\t' + name, param.requires_grad)
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            if contrast_loss == True:
                train_loss = train(epoch, model, mem_size, optimizer_mmilb, criterion, train_loader, stage=0)
            train_loss = train(epoch, model, mem_size, optimizer_main, criterion, train_loader, stage=1)
            
            val_loss, _, _ = evaluate(model, criterion, dev_loader=dev_loader)
            test_loss, results, truths = evaluate(model, criterion, test_loader=test_loader, test=True)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, test_loss))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_mae = test_loss
                accuracy = (results == truths).sum().item() / len(truths)
                torch.save(model, 'best_model.pt')
                print('Model saved!')

            scheduler_main.step()
            scheduler_mmilb.step()

        print('Best MAE: {:5.2f}'.format(best_mae))
        print('Best Validation Loss: {:5.2f}'.format(best_val_loss))
        print('Accuracy: {:5.2f}'.format(accuracy))

        return best_val_loss, best_mae, accuracy


train_loader = get_loader(FOLDER_NAME, batch_size=32, shuffle=True)
dev_loader = get_loader(FOLDER_NAME, batch_size=32, shuffle=False)
test_loader = get_loader(FOLDER_NAME, batch_size=32, shuffle=False)

solver = HateMMSolver(train_loader, dev_loader, test_loader)
# solver.train
# solver.evaluate
solver.train_and_eval

 