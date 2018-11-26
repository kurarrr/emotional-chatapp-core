
# coding: utf-8

# `data_preprocessed_cut_2_Valence.csv`と`data_preprocessed_cut_2_Arousal.csv`をtrainする  
# 
# 引数でdictを受け取るver
# devで相関係数を出す

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import json


# In[2]:


import gensim
gensim_model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec/data.bin', binary=True)


# In[3]:


weights = gensim_model.vectors


# In[4]:


embedding_dim = weights.shape[1]
print(embedding_dim,weights.shape[0])


# In[5]:


weights = np.append(weights,np.zeros((1,embedding_dim)),axis=0)
# 末尾にunknown_wordを追加


# In[6]:


vocab_size = weights.shape[0]
print(vocab_size)


# In[7]:


out_size = 1


# In[8]:


cuda = torch.cuda.is_available()
# cuda = False


# In[9]:


# wordのindexを取得
# print(gensim_model.wv.vocab['always'].index)
# 100番目のwordを取得
# print(gensim_model.wv.index2word[100])


# In[10]:


import re
import nltk


# In[11]:


def prepare_sequence(seq):
    vocab = gensim_model.wv.vocab
    idxs = [vocab[w].index if w in vocab else vocab_size - 1 for w in seq]
    res = torch.tensor(idxs, dtype=torch.long)
    if cuda:
        res = res.cuda()
    return res


# In[12]:


def sentence2vec(sentence):
    w_list = sentence.split()
    res_seq = prepare_sequence(w_list)
    return res_seq


# In[13]:


# s = "I'm always fucking you."
# sentence2vec(s)


# In[14]:


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, option, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = option['hidden_size']
        self.num_layers = option['num_layers']
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        self.bi = (2 if option['bidirectional'] else 1)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        option['input_size'] = embedding_dim
        option['batch_first'] = True

        self.lstm = nn.LSTM(**option)

        # The linear layer that maps from hidden state space to tag space
        self.out = nn.Linear(self.hidden_dim*self.bi, tagset_size)
        self.hidden = self.init_hidden(1)

    def init_hidden(self,batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h = torch.zeros(self.num_layers*self.bi, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_layers*self.bi, batch_size, self.hidden_dim)
        if cuda:
            h = h.cuda()
            c = c.cuda()
        return (h,c)

    def forward(self, sentence, lengths):
        embeds = self.word_embeddings(sentence)
#         print(embeds.size())
#         batch_size = embeds.size()[0]
        packed = nn.utils.rnn.pack_padded_sequence(embeds,lengths,batch_first=True)
        lstm_output, self.hidden = self.lstm(packed, self.hidden)
        unpacked,_ = nn.utils.rnn.pad_packed_sequence(lstm_output,batch_first=True)
        # batch * hidden 
        output = self.out(unpacked[:,-1,:])
        output = F.tanh(output)
        return output


# In[15]:


def make_model(option):
    # 学習済みパラメータ
    torch.manual_seed(1)
    model = LSTMTagger(embedding_dim, option, vocab_size, out_size)
    model.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(weights).float())
    return model


# In[16]:


# ops = {
#     'hidden_size': 3,
#     'num_layers' : 2,
#     'bidirectional' : True,
# }
# make_model(ops)


# In[17]:


def save_model(model,model_name):
    model_state_dict = model.state_dict()
    model_state_dict.pop('word_embeddings.weight')
    torch.save(model_state_dict,model_name)


# In[18]:


def load_model(hidden_dim,model_name):
    torch.manual_seed(1)
    model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, out_size)
    model_state_dict = torch.load(model_name)
    model_state_dict['word_embeddings.weight'] = torch.from_numpy(weights).float()
    model.load_state_dict(model_state_dict)
    # Freeze
    model.word_embeddings.weight.requires_grad = False
    return model


# In[19]:


# vad_type='Valence'
# data_cut = pd.read_csv('./data_preprocessed_{0}.csv'.format(vad_type),encoding='utf-16')
# data_cut = data_cut[data_cut['words']>=2]
# X_train = data_cut[data_cut['data_type']=='train']['reg'].as_matrix()
# Y_train = data_cut[data_cut['data_type']=='train'][['{0}_reg'.format(vad_type)]].as_matrix()
# samples = []
# for sentence,target in zip(X_train,Y_train):
#     sentence_vec = sentence2vec(sentence)
#     y_hat = torch.tensor(target, dtype=torch.float)
#     if cuda:
#         y_hat = y_hat.cuda()
#     samples.append((sentence_vec,y_hat))


# In[20]:


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    res = torch.cat([vec, (torch.zeros(*pad_size,dtype=torch.long).cuda()                            if cuda else torch.zeros(*pad_size,dtype=torch.long))], dim=dim)
    if cuda:
        res = res.cuda()
    return res

class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        lengths = list(map(lambda x: x[0].shape[self.dim], batch))
        # find longest sequence
        max_len = max(lengths)
        # pad according to max_len
        xs = torch.zeros([len(lengths),max_len],dtype=torch.long)
        if cuda:
            xs = xs.cuda()
        for idx,(seq,seqlen) in enumerate(zip(batch,lengths)):
            xs[idx,:seqlen] = seq[0]
        ys = torch.FloatTensor(list(map(lambda x: x[1], batch)))
        lengths_tensor = torch.tensor(lengths)

        if cuda:
            xs = xs.cuda()
            ys = ys.cuda()
            lengths_tensor = lengths_tensor.cuda()
        return xs, ys, lengths_tensor

    def __call__(self, batch):
        return self.pad_collate(batch)


# In[21]:


# train_loader = DataLoader(samples,batch_size=4,shuffle=True,collate_fn=PadCollate(dim=0))


# In[22]:


# for sam in train_loader:
#     sam
#     print(sam[0].size(),sam[1].size())
#     break


# In[23]:


def make_dataset(X,Y):
    ds = []
    for sentence,target in zip(X,Y):
        sentence_vec = sentence2vec(sentence)
        y_hat = torch.tensor(target, dtype=torch.float)
        if cuda:
            y_hat = y_hat.cuda()
        ds.append((sentence_vec,y_hat))
    return ds


# In[31]:


def make_model_and_train_option(option,epochs,vad_type,csv_path='./data_cut_only.csv',metric='MSELoss',
                         save_dir='./dat_model_json',learning_rate=0.01,batch_size=2,print_result=True):
    
    # args
    # option : lstmに渡すoption
    # csv_path : csvへのパス
    # save_dir : modelとjsonをsaveするdir (中にjson,modelディレクトリを含む)
    
    if not 'hidden_size' in option:
        print('hidden_size is necessary')
        return
    hidden_dim = option['hidden_size']
    
    if not 'bidirectional' in option:
        bidirectional = 0
    else:
        bidirectional = (1 if option['bidirectional'] else 0)
    
    if not 'num_layers' in option:
        num_layers = 1
    else:
        num_layers = option['num_layers']
    
    json_name = './{0}/json/{1}_layer_{2}_bi_{3}_hd_{4}_bs_{5}_lr_{6}.json'.format(                        save_dir,vad_type,num_layers,bidirectional,hidden_dim,batch_size,learning_rate)
    base_model_name = './{0}/model/{1}_layer_{2}_bi_{3}_hd_{4}_bs_{5}_lr_{6}'.format(                        save_dir,vad_type,num_layers,bidirectional,hidden_dim,batch_size,learning_rate)

    model = make_model(option)
    train_loss = []
    dev_loss = []
    dev_coef = []
    epoch_start = 0

    if cuda:
        model.cuda()

    data_cut = pd.read_csv('{}'.format(csv_path),encoding='utf-16')
    data_cut = data_cut[data_cut['words']>=2]

    X = data_cut['reg'].as_matrix()
    Y = data_cut['{0}_reg'.format(vad_type)].as_matrix()
    X_train, X_dev, Y_train, Y_dev = train_test_split(X,Y,test_size=0.2)

    ds_train = make_dataset(X_train,Y_train)
    train_loader = DataLoader(ds_train,batch_size=batch_size,shuffle=True,collate_fn=PadCollate(dim=0))

    ds_dev = make_dataset(X_dev,Y_dev)
    dev_loader = DataLoader(ds_dev,batch_size=batch_size,shuffle=True,collate_fn=PadCollate(dim=0))

    if metric == 'MSELoss':
        loss_function = nn.MSELoss(size_average=False)
    elif metric == 'L1Loss':
        loss_function = nn.L1Loss(size_average=False)
    else:
        print('no loss funciton')
        return
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # import time
    # t1 = time.time()

    for epoch in range(epoch_start+1,epoch_start+1+epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        train_loss_sum = 0
        
        for x_batch,y_batch,lengths in train_loader:
            model.zero_grad()
            model.hidden = model.init_hidden(len(x_batch))
            
            # sort
            lengths, perm_idx = lengths.sort(0, descending=True)
            x_batch = x_batch[perm_idx]
            y_batch = y_batch[perm_idx]

            y = model(x_batch,lengths)
            y = y.view(-1)
            loss = loss_function(y, y_batch)
            train_loss_sum += loss.data.item()
            loss.backward()
            optimizer.step()

        if epoch%5==0:
            y_true = []
            y_pred = []
            dev_loss_sum = 0
            
            for x_batch,y_batch,lengths in dev_loader:
                model.zero_grad()
                model.hidden = model.init_hidden(len(x_batch))
                
                # sort
                lengths, perm_idx = lengths.sort(0, descending=True)
                x_batch = x_batch[perm_idx]
                y_batch = y_batch[perm_idx]
                y_true.append(y_batch.cpu().detach().numpy()[0])

                y = model(x_batch,lengths)
                y = y.view(-1)
                y_pred.append(y.cpu().detach().numpy()[0])
                loss = loss_function(y, y_batch)
                dev_loss_sum += loss.data.item()

            dev_loss_av = dev_loss_sum / len(X_dev)
            dev_loss.append(dev_loss_av)
            r = np.corrcoef(y_true,y_pred)[0,1]
            dev_coef.append(r)
          
        if epoch%10==0:
            save_model(model,base_model_name+"_epoch_{0}".format(epoch))
        
        train_loss_av = train_loss_sum/len(X_train)
        if print_result:
            print("epoch {0}: loss {1}".format(epoch,train_loss_av))
        train_loss.append(train_loss_av)

    # t2 = time.time()

    loss_data = {
        'train' : train_loss,
        'dev' : dev_loss,
        'coef' : dev_coef
    }
    with open(json_name,'w') as f:
        json.dump(loss_data,f)


# In[33]:


# op1 = {
#     'hidden_size' : 16,
#     'bidirectional' : False,   
#     'num_layers' : 2,
# }
# op2 = {
#     'hidden_size' : 32,
#     'bidirectional' : True,   
#     'num_layers' : 3,
# }
# make_model_and_train_option(op1,10,'Valence',metric='L1Loss',csv_path='./data_cut_only.csv',
#                             save_dir='./dat_model_json/dat_word_cut_only/',
#                             learning_rate=0.01,batch_size=4)
# make_model_and_train_option(op2,10,'Valence',metric='L1Loss',dat_base_name='./data_preprocessed_cut_2',
#                             save_dir='./dat_model_json/dat_word_cut_l1loss_mullayer_bidirectional',
#                             learning_rate=0.01,batch_size=4)


# In[32]:


vad_types = ['Valence','Arousal']
bss = [4,8]
lrs = [0.005,0.001,0.0001,0.00001]

options = []

hidden_dims = [32,64,128]
num_layers = [1,2]
bis = [True,False]

for num_layer in num_layers:
    for bi in bis:
        for hidden_dim in hidden_dims:
            options.append({
                'hidden_size' : hidden_dim,
                'bidirectional' : bi,   
                'num_layers' : num_layer,
            })

cnt = 0
ma = len(vad_types)*len(lrs)*len(bss)*len(options)

for vad_type in vad_types:
    for lr in lrs:
        for bs in bss:
            for option in options:
                epoch_num = 50
                make_model_and_train_option(option,epoch_num,vad_type,metric='L1Loss',csv_path='./data_cut_only.csv',
                            save_dir='./dat_model_json/dat_word_cut_only/',
                            learning_rate=lr,batch_size=bs,print_result=False)
                cnt += 1
                print('{}/{}'.format(cnt,ma))

