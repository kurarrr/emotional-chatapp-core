
# coding: utf-8

# `data_preprocessed_cut_2_Valence.csv`と`data_preprocessed_cut_2_Arousal.csv`をtrainする

# In[2]:


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


# In[3]:


import gensim
gensim_model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec/data.bin', binary=True)


# In[8]:


weights = gensim_model.vectors


# In[10]:


embedding_dim = weights.shape[1]
print(embedding_dim,weights.shape[0])


# In[11]:


weights = np.append(weights,np.zeros((1,embedding_dim)),axis=0)
# 末尾にunknown_wordを追加


# In[13]:


vocab_size = weights.shape[0]
print(vocab_size)


# In[14]:


out_size = 1


# In[15]:


cuda = torch.cuda.is_available()


# In[9]:


# wordのindexを取得
# print(gensim_model.wv.vocab['always'].index)
# 100番目のwordを取得
# print(gensim_model.wv.index2word[100])


# In[16]:


import re
import nltk


# In[17]:


def prepare_sequence(seq):
    vocab = gensim_model.wv.vocab
    idxs = [vocab[w].index if w in vocab else vocab_size - 1 for w in seq]
    res = torch.tensor(idxs, dtype=torch.long)
    if cuda:
        res = res.cuda()
    return res


# In[18]:


def sentence2vec(sentence):
    w_list = sentence.split()
    res_seq = prepare_sequence(w_list)
    return res_seq


# In[13]:


# s = "I'm always fucking you."
# sentence2vec(s)


# In[19]:


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.out = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden(1)

    def init_hidden(self,batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)
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


# model.state_dict().keys()


# In[20]:


def make_model(hidden_dim):
    # 学習済みパラメータ
    torch.manual_seed(1)
    model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, out_size)
    model.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(weights).float())
    return model


# In[21]:


def save_model(model,model_name):
    model_state_dict = model.state_dict()
    model_state_dict.pop('word_embeddings.weight')
    torch.save(model_state_dict,model_name)


# In[22]:


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


# In[24]:


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


# In[26]:


def make_dataset(X,Y):
    ds = []
    for sentence,target in zip(X,Y):
        sentence_vec = sentence2vec(sentence)
        y_hat = torch.tensor(target, dtype=torch.float)
        if cuda:
            y_hat = y_hat.cuda()
        ds.append((sentence_vec,y_hat))
    return ds


# In[35]:


def make_model_and_train(hidden_dim,epochs,vad_type,dat_base_name='./data_preprocessed',metric='MSELoss',
                         save_dir='./dat_model_json',learning_rate=0.01,batch_size=2,overwrite=False,model_name=""):
    # args
    # dat_base_name : _{Valence,Arousal}.csv以前へのパス
    # save_dor : modelとjsonをsaveするdir (中にjson,modelディレクトリを含む)
    json_name = '{0}/json/loss_{1}_hidden_dim_{2}_batch_{3}_lr_{4}.json'.format(                        save_dir,vad_type,hidden_dim,batch_size,learning_rate)
    base_model_name = '{0}/model/model_{1}_hidden_dim_{2}_batch_{3}_lr_{4}'.format(                        save_dir,vad_type,hidden_dim,batch_size,learning_rate)

    if not overwrite:
        model = make_model(hidden_dim)
        train_loss = []
        dev_loss = []
        epoch_start = 0
    else:
        model = load_model(hidden_dim,model_name)
        with open(json_name,'r') as f:
            dat = json.load(f)            
        train_loss = dat['train']
        dev_loss = dat['dev']
        epoch_start = len(train_loss)

    if cuda:
        model.cuda()

    data_cut = pd.read_csv('{0}_{1}.csv'.format(dat_base_name,vad_type),encoding='utf-16')
    data_cut = data_cut[data_cut['words']>=2]

    X_train = data_cut[data_cut['data_type']=='train']['reg'].as_matrix()
    X_dev   = data_cut[data_cut['data_type']=='dev']['reg'].as_matrix()
    X_test  = data_cut[data_cut['data_type']=='test']['reg'].as_matrix()
    Y_train = data_cut[data_cut['data_type']=='train'][['{0}_reg'.format(vad_type)]].as_matrix()
    Y_dev   = data_cut[data_cut['data_type']=='dev'][['{0}_reg'.format(vad_type)]].as_matrix()
    Y_test  = data_cut[data_cut['data_type']=='test'][['{0}_reg'.format(vad_type)]].as_matrix()

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
            dev_loss_sum = 0
            for x_batch,y_batch,lengths in dev_loader:
                model.zero_grad()
                model.hidden = model.init_hidden(len(x_batch))
                
                # sort
                lengths, perm_idx = lengths.sort(0, descending=True)
                x_batch = x_batch[perm_idx]
                y_batch = y_batch[perm_idx]

                y = model(x_batch,lengths)
                y = y.view(-1)
                loss = loss_function(y, y_batch)
                dev_loss_sum += loss.data.item()

            dev_loss_av = dev_loss_sum / len(X_dev)
            dev_loss.append(dev_loss_av)
          
        if epoch%10==0:
            save_model(model,base_model_name+"_epoch_{0}".format(epoch))
        
        train_loss_av = train_loss_sum/len(X_train)
        print("epoch {0}: loss {1}".format(epoch,train_loss_av))
        train_loss.append(train_loss_av)

    # t2 = time.time()
    ds_test = make_dataset(X_test,Y_test)
    test_loader = DataLoader(ds_test,batch_size=batch_size,shuffle=False,collate_fn=PadCollate(dim=0))
    test_loss_sum = 0
    for x_batch,y_batch,lengths in test_loader:
        model.zero_grad()
        model.hidden = model.init_hidden(len(x_batch))
        # sort
        lengths, perm_idx = lengths.sort(0, descending=True)
        x_batch = x_batch[perm_idx]
        y_batch = y_batch[perm_idx]
        y = model(x_batch,lengths)
        y = y.view(-1)
        loss = loss_function(y, y_batch)
        test_loss_sum += loss.data.item()
    test_loss = test_loss_sum / len(X_test)
    # print(t2-t1)

    loss_data = {
        'train' : train_loss,
        'dev' : dev_loss,
        'test' : test_loss
    }
    with open(json_name,'w') as f:
        json.dump(loss_data,f)


# In[36]:


# make_model_and_train(8,9,'Valence',dat_base_name='./data_preprocessed_cut_2',
#                          save_dir='./dat_model_json/dat_word_cut',learning_rate=0.01,batch_size=4,overwrite=False,model_name="")


# In[25]:


vad_types = ['Valence','Arousal']
hidden_dims = [8,16,32,64,128]
#bss = [4,8,16,32]
bss = [4]
lrs = [0.01, 0.005,0.001]
for lr in lrs:
    for bs in bss:
        for hidden_dim in hidden_dims:
            for vad_type in vad_types:
                epoch_num = 50
                make_model_and_train(hidden_dim,epoch_num,vad_type,dat_base_name='./data_preprocessed_cut_2',metric='L1Loss',
                             save_dir='./dat_model_json/dat_word_cut_l1loss',learning_rate=lr,batch_size=bs,overwrite=False)


# In[36]:


# def make_prediction(hidden_dim,vad_type,model_name,df):

#     model = load_model(hidden_dim,model_name)

#     if cuda:
#         model.cuda()

#     X = df['reg'].as_matrix()
       
#     pred_list = []

#     for sentence in X:
#         sentence_vec = sentence2vec(sentence)
#         sentence_vec = sentence_vec.view(1,-1)
#         print
#         model.zero_grad()
#         model.hidden = model.init_hidden(1)

#         y = model(sentence_vec,[sentence_vec.size()[1]])
#         y = y.view(-1)
#         if cuda:
#             y = y.cpu()
#         pred_list.append(y.detach().numpy())
    
#     pred_list = np.array(pred_list)
    
#     return pred_list


# In[60]:


# hidden_dim = 16
# model_path = {
#     'Valence' : './dat_to1_3/model_Valence_hidden_dim_16_batch_2_lr_0.01_epoch_50',
#     'Arousal' : './dat_to1_3/model_Arousal_hidden_dim_16_batch_2_lr_0.01_epoch_50',
#     'Dominance' : './dat_to1_3/model_Dominance_hidden_dim_16_batch_2_lr_0.01_epoch_50'
# }
# # vad_types = ['Valence']
# vad_types = ['Valence','Arousal','Dominance']

# for vad_type in vad_types:
#     df = pd.read_csv('./data_preprocessed_{0}.csv'.format(vad_type),encoding='utf-16')
#     df = df[df['words']>=2]
#     df = df.reset_index()
#     ary = make_prediction(hidden_dim,vad_type,model_path[vad_type],df)
#     ary = ary * 2 + 3
#     pred_df = df.assign(
#             **{ "{0}_pred".format(vad_type) : pd.Series(ary.reshape(-1))}
#         )
#     to_name = "./prediction/{0}_2.csv".format(vad_type)

#     pred_df.to_csv(to_name,encoding='utf-16',sep="\t")

