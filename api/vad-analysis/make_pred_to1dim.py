
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from nltk.corpus import wordnet
import json,os,sys


print('word2vec loading ...')

import gensim
base_dir_word2vec = './'
gensim_model = gensim.models.KeyedVectors.load_word2vec_format(\
    base_dir_word2vec+'/word2vec/data.bin', binary=True)


weights = gensim_model.wv.syn0
# weights.shape
embedding_dim = weights.shape[1]


# In[4]:

weights = np.append(weights,np.zeros((1,embedding_dim)),axis=0)
# 末尾にunknown_wordを追加


# In[5]:

weights_tensor = torch.from_numpy(weights).float()


# In[6]:

vocab_size = weights.shape[0]


# In[7]:

# cuda = torch.cuda.is_available()
cuda = False


# In[8]:

out_size = 1


# In[9]:

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
        lengths, perm_idx = lengths.sort(0, descending=True)
        sentence = sentence[perm_idx]

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


# In[10]:

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


# In[11]:

import re
import nltk
from nltk import word_tokenize


# In[12]:

def prepare_sequence(seq):
    vocab = gensim_model.wv.vocab
    idxs = [vocab[w].index if w in vocab else vocab_size - 1 for w in seq]
    res = torch.tensor(idxs, dtype=torch.long)
    if cuda:
        res = res.cuda()
    return res


# In[13]:

def sentence2vec(sentence,debug=False):
    sentence = sentence.replace("'"," ").replace("."," ").replace(","," ").replace("\""," ")
    w_list = word_tokenize(sentence)
    w_list = [wordnet.morphy(w).lower() if wordnet.morphy(w) is not None else w.lower() for w in w_list]
    if debug:
        print(w_list)
    res_seq = prepare_sequence(w_list)
    return res_seq


# In[14]:

def load_model(hidden_dim,model_name):
    torch.manual_seed(1)
    model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, out_size)
    model_state_dict = torch.load(model_name)
    model_state_dict['word_embeddings.weight'] = weights_tensor
    model.load_state_dict(model_state_dict)
    model.word_embeddings.weight.requires_grad = False
    return model


def make_pred(model,sentences):
    # args
    #   model : loaded model
    #   sentences : list of sentence
    sentence_vec = [(sentence2vec(sentence),0) for sentence in sentences]
    pad = PadCollate(dim=0)
    sentence_tensor,_,lengths = pad(sentence_vec)
    model.zero_grad()
    model.hidden = model.init_hidden(len(sentences))        
    y = model(sentence_tensor,lengths)

    return y.cpu().detach().numpy()


def make_pred_va(models,sentences):
    v = make_pred(models['Valence'],sentences)
    a = make_pred(models['Arousal'],sentences)
    res = np.r_['1',v,a]
    # res : list of [v,a] for sentences
    return res

