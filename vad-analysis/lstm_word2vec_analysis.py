
# coding: utf-8

# In[3]:


# %%bash
# pip install gensim


# In[4]:


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
import json


# In[5]:


import gensim
gensim_model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec/data.bin', binary=True)


# In[6]:


weights = gensim_model.wv.syn0
# weights.shape


# In[7]:


vocab_size = weights.shape[0]
embedding_dim = weights.shape[1]


# In[8]:


weights = np.append(weights,np.zeros((1,embedding_dim)),axis=0)
# 末尾にunknown_wordを追加


# In[9]:


vocab_size = weights.shape[0]


# In[10]:


# # wordのindexを取得
# print(gensim_model.wv.vocab["'d"].index)
# # 100番目のwordを取得
# print(gensim_model.wv.index2word[100])


# In[11]:


cuda = torch.cuda.is_available()


# In[12]:


import re
import nltk
from nltk import word_tokenize


# In[13]:


def prepare_sequence(seq):
    vocab = gensim_model.wv.vocab
    idxs = [vocab[w].index if w in vocab else vocab_size - 1 for w in seq]
    res = torch.tensor(idxs, dtype=torch.long)
    if cuda:
        res = res.cuda()
    return res


# In[14]:


def sentence2vec(sentence,debug=False):
    w_list = word_tokenize(sentence)
    w_list = [wordnet.morphy(w) if wordnet.morphy(w) is not None else w for w in w_list]
    if debug:
        print(w_list)
    res_seq = prepare_sequence(w_list)
    return res_seq


# In[15]:


# s = "I'm always fucking dogs."
# sentence2vec(s,debug=True)


# In[25]:


def make_model_and_train(hidden_dim,overwrite=False):
    
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

    out_size = 3
    model = LSTMTagger(embedding_dim, hidden_dim, vocab_size, out_size)
    if cuda:
        model.cuda()

    json_name = './dat/loss_data_{0}.json'.format(hidden_dim)
    model_name = './dat/model_data_{0}'.format(hidden_dim)
    
    if overwrite:
        # 上書きする
        model.load_state_dict(torch.load(model_name))
        
        with open(json_name,'r') as f:
            dat = json.load(f)
            
        train_loss = dat['train']
        dev_loss = dat['dev']
    else:
        # 学習済みパラメータ
        pretrained_weights = torch.from_numpy(weights).float()
        if cuda:
            pretrained_weights = pretrained_weights.cuda()
        model.word_embeddings = nn.Embedding.from_pretrained(pretrained_weights)
        
        train_loss = []
        dev_loss = []

        

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

    # sample = [
    #     "I'm always fucking you hey",
    #     "oh my godness"
    # ] 

    # # forwardのsample
    # with torch.no_grad():
    #     inputs = sentence2vec(sample[0])
    #     output = model(inputs)
    # #     print(output[-1,:])

    # dataのload
    data_pre = pd.read_csv('./data_preprocessed.csv')

    # 2word以上のsentence
    data_pre = data_pre[data_pre['words']>=2]

    X_orig = data_pre['reg'].as_matrix()
    Y_v = data_pre['Valence'].as_matrix()
    Y_a = data_pre['Arousal'].as_matrix()
    Y_d = data_pre['Dominance'].as_matrix()
    Y_orig = np.c_[Y_v,Y_a,Y_d]
    a=1
    b=5
    Y_orig = (2*(Y_orig-a)/(b-a))-1
    # [-1,1]で正規化

    X = X_orig
    Y = Y_orig

    train_size = 0.7
    dev_size = 0.2
    X_train, X_rest, Y_train, Y_rest = train_test_split(X, Y, test_size=1-train_size)
    X_dev, X_test, Y_dev, Y_test = train_test_split(X_rest,Y_rest,test_size=1-(train_size+dev_size))

    epochs = 20

    # import time
    # t1 = time.time()

    for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
        train_loss_sum = 0
        X_train,Y_train = shuffle(X_train,Y_train)
        for sentence, target in zip(X_train,Y_train):
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            sentence_in = sentence2vec(sentence)

            # Step 3. Run our forward pass.
            y = model(sentence_in)[-1,:]

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
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
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Also, we need to clear out the hidden state of the LSTM,
                # detaching it from its history on the last instance.
                model.hidden = model.init_hidden()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = sentence2vec(sentence)

                # Step 3. Run our forward pass.
                y = model(sentence_in)[-1,:]

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                y_hat = torch.tensor(target, dtype=torch.float)
                if cuda:
                    y_hat = y_hat.cuda()
                loss = loss_function(y, y_hat)
                dev_loss_sum += loss.data.item()

            dev_loss_av = dev_loss_sum / len(X_dev)
            dev_loss.append(dev_loss_av)

        print("epoch {0}: loss {1}".format(epoch,train_loss_sum/len(X_train)))


        train_loss.append(train_loss_sum/len(X_train))

    # t2 = time.time()

    # print(t2-t1)

    if True:
        torch.save(model.state_dict(),model_name)

#     print(train_loss)
#     print(dev_loss)

    loss_data = {
        'train' : train_loss,
        'dev' : dev_loss
    }
    with open(json_name,'w') as f:
        json.dump(loss_data,f)


# In[26]:


hidden_dims = [6]
for hidden_dim in hidden_dims:
    make_model_and_train(hidden_dim,overwrite=True)

