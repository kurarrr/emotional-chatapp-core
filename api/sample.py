# -*- coding: utf-8 -*-
# sample.py
import falcon
import json,os,sys
import numpy as np
from analysis import make_pred_to1dim as analysis

class ItemsResource:

    def on_post(self, req, resp):
        
        items = {
          'message' : 'ok'
        }

        resp.status = falcon.HTTP_200
        resp.content_type = 'text/plain'
        resp.body = json.dumps(items,ensure_ascii=False)

api = falcon.API()
api.add_route('/prediction_api', ItemsResource())

if __name__ == "__main__":
    from wsgiref import simple_server

    httpd = simple_server.make_server("127.0.0.1", 80, api)
    httpd.serve_forever()
