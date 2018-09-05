# -*- coding: utf-8 -*-
# sample.py
import falcon
import json,os,sys
import numpy as np
from analysis import make_pred_to1dim as analysis

class ItemsResource:

    def on_post(self, req, resp):
        """
        params
        - {  msg           : message
             msg_user_attr : native or non-native  }
        reponse
        - {
        effect : 
            { native or non-native : css_class_name }
        }
        """
        
        va_numpy = analysis.make_pred_va([msg])
        print(va_numpy)
        

        resp.status = falcon.HTTP_200
        resp.content_type = 'text/plain'
        resp.body = json.dumps(items,ensure_ascii=False)

api = falcon.API()
api.add_route('/prediction_api', ItemsResource())

if __name__ == "__main__":
    from wsgiref import simple_server

    # modelの読み込み
    print('model loading...')
    base_path = './analysis/dat_model_json/'
    model_path = {
        'Valence' : base_path + 'model_Valence_hidden_dim_32_batch_8_lr_0.005_epoch_50',
        'Arousal' : base_path + 'model_Arousal_hidden_dim_32_batch_8_lr_0.005_epoch_50',
    }

    va = ['Valence','Arousal']
    hidden_dim = 32
    for va_type in va:
        model[va_type] = analysis.load_model(hidden_dim,model_path[va_type])

    print('model loaded')



    httpd = simple_server.make_server("127.0.0.1", 80, api)
    httpd.serve_forever()
