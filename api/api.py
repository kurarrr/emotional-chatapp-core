# -*- coding: utf-8 -*-
# sample.py
import falcon
import json,os,sys
import numpy as np
from analysis import make_pred_to1dim_mean_pooling as analysis

class ItemsResource:

    def __init__(self):
        print('model loading...')

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
        model_a = load_model(ops_a,model_name_a)

        ops_v = {
            'hidden_size': 240,
            'num_layers' : 2,
            'bidirectional' : True,
        }
        bi_v = (1 if ops['bidirectional'] else 0)
        bs_v = 50
        lr_v = 0.03
        optimizer_v = 'Adagrad'
        model_name_v = base_path + './{}_layer_{}_bi_{}_hd_{}_bs_{}_lr_{}_{}'.format(
            'Valence',ops_v['num_layers'],bi_v,ops_v['hidden_size'],bs_v,lr_v,optimizer_v
        )
        model_v = load_model(ops_v,model_name_v)


        self.models['Valence'] = model_v
        self.models['Arousal'] = model_a 

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
        valence, arousal = analysis.make_pred_va_sentence(self.models,msg)

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
