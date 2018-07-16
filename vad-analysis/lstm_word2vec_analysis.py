
# coding: utf-8

# In[5]:


# %%bash
# pip install gensim


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


# In[2]:


import gensim
gensim_model = gensim.models.KeyedVectors.load_word2vec_format('./data.bin', binary=True)


# In[3]:


weights = gensim_model.wv.syn0
# weights.shape


# In[4]:


vocab_size = weights.shape[0]
embedding_dim = weights.shape[1]


# In[5]:


weights = np.append(weights,np.zeros((1,embedding_dim)),axis=0)
# 末尾にunknown_wordを追加


# In[6]:


vocab_size = weights.shape[0]


# In[7]:


# # wordのindexを取得
# print(gensim_model.wv.vocab["'d"].index)
# # 100番目のwordを取得
# print(gensim_model.wv.index2word[100])


# In[23]:


#cuda = torch.cuda.is_available()
cuda = True

# In[8]:


import re
import nltk
from nltk import word_tokenize


# In[30]:


def prepare_sequence(seq):
    vocab = gensim_model.wv.vocab
    idxs = [vocab[w].index if w in vocab else vocab_size - 1 for w in seq]
    res = torch.tensor(idxs, dtype=torch.long)
    if cuda:
        res = res.cuda()
    return res


# In[10]:


def sentence2vec(sentence,debug=False):
    w_list = word_tokenize(sentence)
    w_list = [wordnet.morphy(w) if wordnet.morphy(w) is not None else w for w in w_list]
    if debug:
        print(w_list)
    res_seq = prepare_sequence(w_list)
    return res_seq


# In[11]:


# s = "I'm always fucking dogs."
# sentence2vec(s,debug=True)


# In[78]:


def make_model_and_train(hidden_dim):
    
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

    # 学習済みパラメータ
    pretrained_weights = torch.from_numpy(weights).float()
    if cuda:
        pretrained_weights = pretrained_weights.cuda()
    model.word_embeddings = nn.Embedding.from_pretrained(pretrained_weights)

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

    epochs = 30

    train_loss = []
    dev_loss = []

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


        train_loss.append(loss_sum/len(X_train))

    # t2 = time.time()

    # print(t2-t1)

    if hidden_dim >= 16:
        torch.save(model.state_dict(),'./dat/model_data_{0}'.format(hidden_dim))

#     print(train_loss)
#     print(dev_loss)
    import json
    loss_data = {
        'train' : train_loss,
        'dev' : dev_loss
    }
    json_name = './dat/loss_data_{0}.json'.format(hidden_dim)
    with open(json_name,'w') as f:
        json.dump(loss_data,f)


# In[73]:


hidden_dims = [6]
for hidden_dim in hidden_dims:
    make_model_and_train(hidden_dim)

