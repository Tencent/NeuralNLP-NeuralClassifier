from infer import Inferer
from config import Config
import sys
from flask import Flask, redirect,url_for,request
from flask_cors import *
import json
app = Flask(__name__)





@app.route("/sentiment/predict",methods = ["POST"])
def predict():
    
    text_list = request.get_json()['text_list']
    result = inferer.infer(text_list)
    result = json.dumps(result,ensure_ascii = False)
    return result


if __name__ == "__main__":

    config = Config(config_file = sys.argv[1])
    inferer = Inferer(config)
    
    CORS(app,supports_credentials = True)
    app.run(host='0.0.0.0',debug =True, port = 16666)
