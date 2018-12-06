# coding: utf-8
from flask import Flask,request,jsonify,make_response
from flask_cors import CORS
import json

api = Flask(__name__)
CORS(api)

@api.route("/prediction_api",methods=['GET'])
def hello():
    msg = request.args.get("msg",default="hoge")
    res = {
        'msg' : msg,
        'Valence' : 0.5,
        'Arousal' : 0.5
    }
    return make_response(jsonify(res))

if __name__ == "__main__":
    api.run()