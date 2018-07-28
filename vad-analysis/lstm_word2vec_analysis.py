
# coding: utf-8

# In[21]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


weights = gensim_model.wv.syn0


# In[4]:


embedding_dim = weights.shape[1]


# In[5]:


weights = np.append(weights,np.zeros((1,embedding_dim)),axis=0)
# 末尾にunknown_wordを追加


# In[6]:


vocab_size = weights.shape[0]


# In[7]:


out_size = 3


# In[8]:


cuda = torch.cuda.is_available()


# In[12]:


# wordのindexを取得
# print(gensim_model.wv.vocab['always'].index)
# 100番目のwordを取得
# print(gensim_model.wv.index2word[100])


# In[9]:


import re
import nltk


# In[10]:


def prepare_sequence(seq):
    vocab = gensim_model.wv.vocab
    idxs = [vocab[w].index if w in vocab else vocab_size - 1 for w in seq]
    res = torch.tensor(idxs, dtype=torch.long)
    if cuda:
        res = res.cuda()
    return res


# In[11]:


def sentence2vec(sentence):
    w_list = sentence.split()
    res_seq = prepare_sequence(w_list)
    return res_seq


# In[17]:


# s = "I'm always fucking you."
# sentence2vec(s)


# In[12]:


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.out = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        h = torch.zeros(1, 1, self.hidden_dim)
        c = torch.zeros(1, 1, self.hidden_dim)
        if cuda:
            h = h.cuda()
            c = c.cuda()
        return (h,c)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
#         print(embeds.size())

        lstm_output, self.hidden = self.lstm(
            embeds.view(len(sentence),1,-1), self.hidden)

        output = self.out(lstm_output.view(len(sentence),-1))
        output = F.tanh(output)
        return output


# In[13]:


# model.state_dict().keys()


# In[14]:


def make_model(hidden_dim):
    # 学習済みパラメータ
    torch.manual_seed(1)
    model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, out_size)
    model.word_embeddings = nn.Embedding.from_pretrained(torch.from_numpy(weights).float())
    return model


# In[15]:


def save_model(model,model_name):
    model_state_dict = model.state_dict()
    model_state_dict.pop('word_embeddings.weight')
    torch.save(model_state_dict,model_name)


# In[16]:


def load_model(hidden_dim,model_name):
    torch.manual_seed(1)
    model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, out_size)
    model_state_dict = torch.load(model_name)
    model_state_dict['word_embeddings.weight'] = torch.from_numpy(weights).float()
    model.load_state_dict(model_state_dict)
    model.word_embeddings.weight.requires_grad = False
    return model

def make_model_and_train(hidden_dim,epochs,overwrite=False,model_name=""):

    json_name = './dat/loss_data_{0}.json'.format(hidden_dim)
    base_model_name = './dat/model_data_{0}'.format(hidden_dim)

    if not overwrite:
        model = make_model(hidden_dim)
        train_loss = []
        dev_loss = []
    else:
        model = load_model(hidden_dim,model_name)
        with open(json_name,'r') as f:
            dat = json.load(f)            
        train_loss = dat['train']
        dev_loss = dat['dev']

    if cuda:
        model.cuda()

    data_cut = pd.read_csv('./data_cut.csv',encoding='utf-16')
    data_cut = data_cut[data_cut['words']>=2]

    X_train = data_cut[data_cut['data_type']=='train']['reg'].as_matrix()
    X_dev   = data_cut[data_cut['data_type']=='dev']['reg'].as_matrix()
    X_test  = data_cut[data_cut['data_type']=='test']['reg'].as_matrix()
    Y_train = data_cut[data_cut['data_type']=='train'][['Valence_reg','Arousal_reg','Dominance_reg']].as_matrix()
    Y_dev   = data_cut[data_cut['data_type']=='dev'][['Valence_reg','Arousal_reg','Dominance_reg']].as_matrix()
    Y_test  = data_cut[data_cut['data_type']=='test'][['Valence_reg','Arousal_reg','Dominance_reg']].as_matrix()

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    

    # import time
    # t1 = time.time()

    for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        train_loss_sum = 0
        X_train,Y_train = shuffle(X_train,Y_train)
        for sentence, target in zip(X_train,Y_train):
            model.zero_grad()
            model.hidden = model.init_hidden()
            sentence_in = sentence2vec(sentence)
            y = model(sentence_in)[-1,:]
            y_hat = torch.tensor(target, dtype=torch.float)
            if cuda:
                y_hat = y_hat.cuda()
            loss = loss_function(y, y_hat)
            train_loss_sum += loss.data.item()
            loss.backward()
            optimizer.step()

        if (epoch+1)%5==0:
            dev_loss_sum = 0
            for sentence, target in zip(X_dev,Y_dev):
                model.zero_grad()
                model.hidden = model.init_hidden()
                sentence_in = sentence2vec(sentence)
                y = model(sentence_in)[-1,:]
                y_hat = torch.tensor(target, dtype=torch.float)
                if cuda:
                    y_hat = y_hat.cuda()
                loss = loss_function(y, y_hat)
                dev_loss_sum += loss.data.item()

            dev_loss_av = dev_loss_sum / len(X_dev)
            dev_loss.append(dev_loss_av)
          
        if (epoch+1)%10==0:
            save_model(model,base_model_name+"_epoch_{0}".format(epoch))
        
        train_loss_av = train_loss_sum/len(X_train)
        print("epoch {0}: loss {1}".format(epoch,train_loss_av))
        train_loss.append(train_loss_av)

    # t2 = time.time()

    # print(t2-t1)

    loss_data = {
        'train' : train_loss,
        'dev' : dev_loss
    }
    with open(json_name,'w') as f:
        json.dump(loss_data,f)


def main():
    hidden_dims = [3,6]
    for hidden_dim in hidden_dims:
        make_model_and_train(hidden_dim,60,overwrite=True,model_name="./dat/model_data_{0}_epoch_59".format(hidden_dim))
    
    hidden_dims = [12, 24]
    for hidden_dim in hidden_dims:
        make_model_and_train(hidden_dim,100,overwrite=False)
#     make_model_and_train(hidden_dim,overwrite=True,model_name='./dat/model_data_2_epoch_0')

if __name__=='__main__':
    main()
