# -*- coding: utf-8 -*-
# sample.py
import falcon
import json,os,sys
import numpy as np

class ItemsResource:

    def on_get(self, req, resp):
        params = req.params
        print(params)
        data = json.loads(params)

        items = {
		'Valence' : 0.5,
		'Arousal' : 0.45
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
#api = falcon.API()
api.add_route('/prediction_api', ItemsResource())

if __name__ == "__main__":
    from wsgiref import simple_server

    httpd = simple_server.make_server("127.0.0.1", 8000, api)
    httpd.serve_forever()
