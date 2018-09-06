# -*- coding: utf-8 -*-
# sample.py
import falcon
import json,os,sys
import numpy as np

class ItemsResource:

    def on_post(self, req, resp):
        body = req.stream.read()
        data = json.loads(body)
        print(data)
        items = {
          'message' : 'ok'
        }

        resp.status = falcon.HTTP_200
        resp.content_type = 'text/plain'
        resp.body = json.dumps(items,ensure_ascii=False)


class CORSMiddleware:
    def process_request(self, req, resp):
        resp.set_header('Access-Control-Allow-Origin', '*')


api = falcon.API(middleware=[CORSMiddleware()])
api.add_route('/prediction_api', ItemsResource())

if __name__ == "__main__":
    from wsgiref import simple_server

    httpd = simple_server.make_server("127.0.0.1", 8000, api)
    httpd.serve_forever()
