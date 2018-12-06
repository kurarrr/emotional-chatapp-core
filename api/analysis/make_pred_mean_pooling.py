
# coding: utf-8

# `data_preprocessed_cut_2_Valence.csv`と`data_preprocessed_cut_2_Arousal.csv`をtrainする  
# 
# 引数でdictを受け取るver
# devで相関係数を出す
# 
# cross-validationをやる

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
from sklearn.model_selection import KFold

import json


# In[2]:


from nltk import word_tokenize
from nltk.corpus import wordnet
import gensim


# In[3]:


# from gensim.test.utils import datapath, get_tmpfile
# from gensim.models import KeyedVectors

# from gensim.scripts.glove2word2vec import glove2word2vec
# # transform : glove -> tmp
# glove2word2vec('./stanford_glove.txt', './stanford_w2v.txt')


# In[4]:


import re
import nltk


# In[29]:


def prepare_sequence(seq,gensim_model):
    vocab = gensim_model.wv.vocab
    idxs = [vocab[w].index if w in vocab else vocab_size - 1 for w in seq]
    res = torch.tensor(idxs, dtype=torch.long)
    #if cuda:
    #    res = res.cuda()
    return res


# In[7]:


# s = "I'm always fucking you."
# sentence2vec(s)


# In[8]:


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
        
        # 2layer固定 調べるのめんどい
        self.lstm.weight_hh_l0.data.uniform_(-0.01,0.01)        
#         self.lstm.weight_hh_l1.data.uniform_(-0.01,0.01)        
        self.lstm.weight_ih_l0.data.uniform_(-0.01,0.01)        
#         self.lstm.weight_ih_l1.data.uniform_(-0.01,0.01)       
        
#         self.lstm.weight_hh_l0_reverse.data.uniform_(-0.01,0.01)        
#         self.lstm.weight_hh_l1_reverse.data.uniform_(-0.01,0.01)        
#         self.lstm.weight_ih_l0_reverse.data.uniform_(-0.01,0.01)        
#         self.lstm.weight_ih_l1_reverse.data.uniform_(-0.01,0.01)       
        
        # The linear layer that maps from hidden state space to tag space
        self.out = nn.Linear(self.hidden_dim*self.bi, tagset_size)
        self.out.weight.data.uniform_(-0.01,0.01)

        self.hidden = self.init_hidden(1)

    def init_hidden(self,batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h = torch.zeros(self.num_layers*self.bi, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_layers*self.bi, batch_size, self.hidden_dim)
        #if cuda:
        #    h = h.cuda()
        #    c = c.cuda()
        return (h,c)

    def forward(self, sentence, lengths):
        embeds = self.word_embeddings(sentence)
#         print(embeds.size())
#         batch_size = embeds.size()[0]
        packed = nn.utils.rnn.pack_padded_sequence(embeds,lengths,batch_first=True)
        lstm_output, self.hidden = self.lstm(packed, self.hidden)
        unpacked,_ = nn.utils.rnn.pad_packed_sequence(lstm_output,batch_first=True)
        # print(unpacked.size())
        # :batch * max(len(lengths)) * hidden
        
        unpacked = torch.mean(unpacked,1)
        # print(unpacked.size())
        # :batch * hidden
        output = self.out(unpacked)
        output = F.tanh(output)
        return output


# In[9]:


# h = torch.nn.LSTM(1,2,1)
# h.weight_hh_l0


# In[10]:


def load_model(option,model_name,embedding_dim,vocab_size,out_size,weights):
    torch.manual_seed(1)
    model = LSTMTagger(embedding_dim, option, vocab_size, out_size)
    # model_state_dict = torch.load(model_name)
    model_state_dict = torch.load(model_name,map_location=lambda storage, loc:storage)
    model_state_dict['word_embeddings.weight'] = torch.from_numpy(weights).float()
    model.load_state_dict(model_state_dict)
    # Freeze
    model.word_embeddings.weight.requires_grad = False
    return model


# In[11]:


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
    res = torch.cat([vec, (torch.zeros(*pad_size,dtype=torch.long))], dim=dim)

#     if cuda:
#         res = res.cuda()
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
#         if cuda:
#             xs = xs.cuda()
        for idx,(seq,seqlen) in enumerate(zip(batch,lengths)):
            xs[idx,:seqlen] = seq[0]
        ys = torch.FloatTensor(list(map(lambda x: x[1], batch)))
        lengths_tensor = torch.tensor(lengths)

#         if cuda:
#             xs = xs.cuda()
#             ys = ys.cuda()
#             lengths_tensor = lengths_tensor.cuda()
        return xs, ys, lengths_tensor

    def __call__(self, batch):
        return self.pad_collate(batch)


# In[12]:


def sentence2vec(sentence,gensim_model,debug=False):
    sentence = sentence.replace("."," ").replace(","," ").replace("!"," ").replace("'","").replace("\"","").replace("“","").replace("”","")
    w_list = word_tokenize(sentence)
    w_list = [wordnet.morphy(w).lower() if wordnet.morphy(w) is not None else w.lower() for w in w_list]
    if debug:
        print(w_list)
    res_seq = prepare_sequence(w_list,gensim_model)
    return res_seq


# In[42]:


def make_pred(model,sentences,gensim_model):
    # args
    #   model : loaded model
    #   sentences : list of sentence
    sentence_vec = [(sentence2vec(sentence,gensim_model),0) for sentence in sentences]
    pad = PadCollate(dim=0)
    sentence_tensor,_,lengths = pad(sentence_vec)
    lengths, perm_idx = lengths.sort(0, descending=True)
    sentence_tensor = sentence_tensor[perm_idx]

    model.zero_grad()
    model.hidden = model.init_hidden(len(sentences))        
    y = model(sentence_tensor,lengths)
    
    _, inv_perm_idx = perm_idx.sort(0)

    # print(perm_idx[inv_perm_idx])
    # this tensor is [0,1,2,..]
    y = y[inv_perm_idx]
    return y.cpu().detach().numpy()


# In[43]:


def make_pred_va(models,sentences,gensim_model):
    v = make_pred(models['Valence'],sentences,gensim_model)
    a = make_pred(models['Arousal'],sentences,gensim_model)
    res = np.r_['1',v,a]
    # res : nparray of [v,a] for sentences
    return res


# In[44]:


def make_pred_va_sentence(models, sentence, gensim_model):
    # arg 
    # sentence : string
    v = float(make_pred(models['Valence'],[sentence],gensim_model))
    a = float(make_pred(models['Arousal'],[sentence],gensim_model))
    # res : float value of v,a
    return v,a


# In[64]:


# def run():
#     gensim_model = gensim.models.KeyedVectors.load_word2vec_format('./stanford_w2v.txt')
#     weights = gensim_model.vectors
#     embedding_dim = weights.shape[1]
#     weights = np.append(weights,np.zeros((1,embedding_dim)),axis=0)
#     # 末尾にunknown_wordを追加
#     vocab_size = weights.shape[0]
#     print("vocab : {}".format(vocab_size))
#     out_size = 1

#     base_path = './dat_model_json/best/'

#     ops_a = {
#         'hidden_size': 60,
#         'num_layers' : 1,
#         'bidirectional' : False,
#     }
#     bi_a = (1 if ops_a['bidirectional'] else 0)
#     bs_a = 200
#     lr_a = 0.001
#     optimizer_a = 'Adagrad'
#     model_name_a = base_path + './{}_layer_{}_bi_{}_hd_{}_bs_{}_lr_{}_{}'.format(
#         'Arousal',ops_a['num_layers'],bi_a,ops_a['hidden_size'],bs_a,lr_a,optimizer_a
#     )
#     model_a = load_model(ops_a,model_name_a,embedding_dim,vocab_size,out_size,weights)

#     ops_v = {
#         'hidden_size': 240,
#         'num_layers' : 2,
#         'bidirectional' : True,
#     }
#     bi_v = (1 if ops_v['bidirectional'] else 0)
#     bs_v = 50
#     lr_v = 0.03
#     optimizer_v = 'Adagrad'
#     model_name_v = base_path + './{}_layer_{}_bi_{}_hd_{}_bs_{}_lr_{}_{}'.format(
#         'Valence',ops_v['num_layers'],bi_v,ops_v['hidden_size'],bs_v,lr_v,optimizer_v
#     )
#     model_v = load_model(ops_v,model_name_v,embedding_dim,vocab_size,out_size,weights)

#     models = {
#         'Valence' : model_v,
#         'Arousal' : model_a
#     }

#     sentences = [
#         "I love you very much",    
#         "I'm very angry",
#         'Are you kidding?',
#         "fugaaaaaaaaaaa",
#     ]
#     print(make_pred_va(models,sentences,gensim_model))

