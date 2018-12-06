# -*- coding: utf-8 -*-
# sample.py
import falcon
import json,os,sys
import numpy as np
from analysis import make_pred_mean_pooling as analysis

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

from nltk import word_tokenize
from nltk.corpus import wordnet
import gensim



class ItemsResource:

    def __init__(self):
        print('model loading...')

        w2v_path = './analysis/stanford_w2v.txt'
        self.gensim_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)
        weights = self.gensim_model.vectors
        embedding_dim = weights.shape[1]
        weights = np.append(weights,np.zeros((1,embedding_dim)),axis=0)
        # 末尾にunknown_wordを追加
        vocab_size = weights.shape[0]
        print("vocab : {}".format(vocab_size))
        out_size = 1
        
        
        base_path = './analysis/dat_model_json/best/'

        ops_a = {
            'hidden_size': 60,
            'num_layers' : 1,
            'bidirectional' : False,
        }
        bi_a = (1 if ops_a['bidirectional'] else 0)
        bs_a = 200
        lr_a = 0.001
        optimizer_a = 'Adagrad'
        model_name_a = base_path + './{}_layer_{}_bi_{}_hd_{}_bs_{}_lr_{}_{}'.format(
            'Arousal',ops_a['num_layers'],bi_a,ops_a['hidden_size'],bs_a,lr_a,optimizer_a
        )
        model_a = analysis.load_model(ops_a,model_name_a,embedding_dim,vocab_size,out_size,weights)

        ops_v = {
            'hidden_size': 240,
            'num_layers' : 2,
            'bidirectional' : True,
        }
        bi_v = (1 if ops_v['bidirectional'] else 0)
        bs_v = 50
        lr_v = 0.03
        optimizer_v = 'Adagrad'
        model_name_v = base_path + './{}_layer_{}_bi_{}_hd_{}_bs_{}_lr_{}_{}'.format(
            'Valence',ops_v['num_layers'],bi_v,ops_v['hidden_size'],bs_v,lr_v,optimizer_v
        )
        model_v = analysis.load_model(ops_v,model_name_v,embedding_dim,vocab_size,out_size,weights)
        
        self.models = {
            'Valence' : model_v,
            'Arousal' : model_a
        }

        print('model loaded')



    def on_get(self, req, resp):
        """
        params
        - { msg : message}
        reponse
        - { Valence, Arousal }
        """
        
        # data : dict
        data = req.params

        msg = data['msg']
        valence, arousal = analysis.make_pred_va_sentence(self.models,msg,self.gensim_model)

        items = {
            'Valence' : valence,
            'Arousal' : arousal
        }

        resp.status = falcon.HTTP_200
        resp.content_type = 'text/plain'
        resp.body = json.dumps(items,ensure_ascii=False)

class CORSMiddleware:
    def process_request(self, req, resp):
        resp.set_header('Access-Control-Allow-Origin', '*')
        resp.set_header('Access-Control-Allow-Methods', '*')
        resp.set_header('Access-Control-Allow-Headers', '*')
        resp.set_header('Access-Control-Max-Age', 1728000)

api = falcon.API(middleware=[CORSMiddleware()])
itemResource = ItemsResource()
api.add_route('/prediction_api',itemResource)

if __name__ == "__main__":
    from wsgiref import simple_server



    httpd = simple_server.make_server("127.0.0.1", 8000, api)
    httpd.serve_forever()
