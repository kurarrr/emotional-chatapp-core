# -*- coding: utf-8 -*-
# sample.py
import falcon
import json,os,sys
import numpy as np
from analysis import make_pred_to1dim as analysis

class ItemsResource:

    def __init__(self):
        print('model loading...')
        base_path = './analysis/dat_model_json/'
        self.model_path = {
            'Valence' : base_path + 'model_Valence_hidden_dim_32_batch_8_lr_0.005_epoch_50',
            'Arousal' : base_path + 'model_Arousal_hidden_dim_32_batch_8_lr_0.005_epoch_50',
        }

        va = ['Valence','Arousal']
        hidden_dim = 32
        self.models = {}
        for va_type in va:
            self.models[va_type] = analysis.load_model(hidden_dim,self.model_path[va_type])

        print('model loaded')



    def on_get(self, req, resp):
        """
        params
        - { msg : message}
        reponse
        - { Valence, Arousal }
        """
        
        body = req.stream.read()
        data = json.loads(body)
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
