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

        json_path = os.path.dirname(os.path.abspath(__file__))+'/analysis/font100_vad_reg.json'
        with open(json_path,'r') as fp:
            self.dat = json.load(fp)
        ary = []
        for dic in self.dat:
            ary.append([float(dic['Valence']),float(dic['Arousal'])])
        self.ary_np = np.array(ary)
        print('VA value loaded')


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
        
        body = req.stream.read()
        data = json.loads(body)
        msg = data['msg']
        idx = analysis.most_closest(self.models,msg,self.ary_np)
        font_name = self.dat[idx]['name']
        msg_user_attr = data['msg_user_attr']
        attr = ''
        if msg_user_attr == 'native':
            attr = 'non-native'
        else:
            attr = 'native'
        items = { attr : 'font-'+font_name.replace('.','-')}

        resp.status = falcon.HTTP_200
        resp.content_type = 'text/plain'
        resp.body = json.dumps(items,ensure_ascii=False)

class CORSMiddleware:
    def process_request(self, req, resp):
        resp.set_header('Access-Control-Allow-Origin', '*')

api = falcon.API(middleware=[CORSMiddleware()])
itemResource = ItemsResource()
api.add_route('/prediction_api',itemResource)

if __name__ == "__main__":
    from wsgiref import simple_server



    httpd = simple_server.make_server("127.0.0.1", 8000, api)
    httpd.serve_forever()
