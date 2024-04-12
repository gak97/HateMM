import torch
import pickle
import pandas as pd
import time

import torch.autograd as autograd
import torch.nn as nn

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

with open(FOLDER_NAME+'HateMM_annotation.csv','rb') as fa:
    labels = pickle.load(fa)

with open(FOLDER_NAME+'all__video_vosk_audioMap.pkl','rb') as fb:
    transcript = pickle.load(fb)

with open(FOLDER_NAME+'MFCCFeaturesNew.pkl','rb') as fc:
    audData = pickle.load(fc)

with open(FOLDER_NAME+'final_allVideos.pkl', 'rb') as fd:
    allDataAnnotation = pickle.load(fd)
    allVidList = list(allDataAnnotation.values())

df = pd.read_csv(labels)   
df['label'] = df['label'].apply(lambda x: 1 if x == 'Hate' else 0)
# label = df['label'].values

# divide the dataset into train, val, and test in the ratio of 70:10:20
train_split = int(0.7*len(transcript))
val_split = int(0.1*len(transcript))
test_split = int(0.2*len(transcript))

alpha = 0.5
beta = 0.5


        

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

num_epochs = 1

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

