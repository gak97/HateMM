import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

class LanguageEmbeddingLayer(nn.Module):
    def __init__(self):
        super(LanguageEmbeddingLayer, self).__init__()
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)

    def forward(self, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent, attention_mask=bert_sent_mask, token_type_ids=bert_sent_type)
        return bert_output[0][:, 0, :]

class SubNet(nn.Module):
    def __init__(self, in_size, hidden_size, n_class, dropout):
        super(SubNet, self).__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, n_class)

    def forward(self, x):
        dropped = self.drop(x)
        y_1 = torch.tanh(self.linear_1(dropped))
        y_2 = torch.tanh(self.linear_2(y_1))
        y_3 = self.linear_3(y_2)
        return y_2, y_3

class MMILB(nn.Module):
    def __init__(self, x_size, y_size, mid_activation='ReLU', last_activation='Tanh'):
        super(MMILB, self).__init__()
        self.mid_activation = getattr(nn, mid_activation)
        self.last_activation = getattr(nn, last_activation)
        self.mlp_mu = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size)
        )
        self.mlp_logvar = nn.Sequential(
            nn.Linear(x_size, y_size),
            self.mid_activation(),
            nn.Linear(y_size, y_size),
        )
        self.entropy_prj = nn.Sequential(
            nn.Linear(y_size, y_size // 4),
            nn.Tanh()
        )
        self.y_size = y_size
        self.x_size = x_size

    def forward(self, x, y, labels=None, mem=None):
        mu, logvar = self.mlp_mu(x), self.mlp_logvar(x)
        positive = -(mu - y)**2/2./torch.exp(logvar)
        lld = torch.mean(torch.sum(positive, -1))

        # pos_y = neg_y = None
        H = 0.0
        # sample_dict = {'pos':None, 'neg':None}
        sample_output = None

        if labels is not None:
            y = self.entropy_prj(y)
            labels = labels.view(-1)  # Reshape labels to match the shape of y
            pos_mask = labels > 0
            # neg_mask = labels < 0
            
            # Check if there are any positive or negative labels
            if pos_mask.any():
                pos_y = y[pos_mask]
                # sample_dict['pos'] = pos_y
                sample_output = pos_y
            # if neg_mask.any():
            #     neg_y = y[neg_mask]
            #     sample_dict['neg'] = neg_y

            # print(f"mem: {mem}")
            # if mem is not None and mem.get('pos', None) is not None and mem.get('neg', None) is not None:
            if mem is not None: 
                # pos_history = mem['pos']
                # neg_history = mem['neg']
                pos_history = mem
                
                if pos_y is not None:
                    pos_all = torch.cat([*pos_history, pos_y], dim=0) 
                    mu_pos = pos_all.mean(dim=0)
                    sigma_pos = torch.mean(torch.bmm((pos_all-mu_pos).unsqueeze(-1), (pos_all-mu_pos).unsqueeze(1)), dim=0)
                else:
                    sigma_pos = torch.eye(self.y_size // 4).to(y.device)
                
                # if neg_y is not None:
                #     neg_all = torch.cat(neg_history + [neg_y], dim=0)
                #     mu_neg = neg_all.mean(dim=0)
                #     sigma_neg = torch.mean(torch.bmm((neg_all-mu_neg).unsqueeze(-1), (neg_all-mu_neg).unsqueeze(1)), dim=0)
                # else:
                #     sigma_neg = torch.eye(self.y_size // 4).to(y.device)
                
                H = 0.25 * (torch.logdet(sigma_pos))

        return lld, sample_output, H
      
class CPC(nn.Module):
    def __init__(self, x_size, y_size, n_layers=1, activation='Tanh'):
        super().__init__()
        self.activation = getattr(nn, activation)
        if n_layers == 1:
            self.net = nn.Linear(y_size, x_size)
        else:
            net = []
            for i in range(n_layers):
                if i == 0:
                    net.append(nn.Linear(y_size, x_size))
                    net.append(self.activation())
                else:
                    net.append(nn.Linear(x_size, x_size))
            self.net = nn.Sequential(*net)
        
    def forward(self, x, y):
        x_pred = self.net(y)

        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x_pred.dim() == 1:
            x_pred = x_pred.unsqueeze(0)

        x_pred = x_pred / (x_pred.norm(dim=-1, keepdim=True) + 1e-8)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)

        mask = torch.rand(x.shape, device=x.device) < 0.15  # Mask 15% of the input elements
        masked_x = x * mask  # Apply the mask

        pos = torch.sum(masked_x*x_pred, dim=-1)
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)
        nce = -(pos - neg).mean()
        return nce

class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear((2 if bidirectional else 1)*hidden_size, out_size)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        _, (h, _) = self.rnn(x)
        # print(f"Hidden state shape: {h.shape}")

        if self.bidirectional:
            # h = self.dropout(torch.cat((h[-2, :], h[-1, :]), dim=0))
            h = self.dropout(torch.cat((h[-2, :, :], h[-1, :, :]), dim=1))
        else:
            h = self.dropout(h[-1])
        y_1 = self.linear_1(h)
        # print(f"Output shape: {y_1.shape}")

        return y_1