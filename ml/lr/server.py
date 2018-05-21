# -*- coding: utf-8 -*-

from flask import Flask
from flask import request
import logging
import sys
import os
from model.LogisticRegression import LogisticRegression

app = Flask(__name__)
lr_model=None


@app.route('/', methods=['GET', 'POST'])
def home():
    app.logger.info('visit home')
    return "{'home': 'this is home'}"


@app.route('/signin', methods=['GET'])
def signin_form():
    return '''<form action="/signin" method="post">
              <p><input name="username"></p>
              <p><input name="password" type="password"></p>
              <p><button type="submit">Sign In</button></p>
              </form>'''

@app.route('/push/getPushItemScore', methods=['GET'])
def getPushItemScore():
    if lr_model is None:
        app.logger.info("LR model not exist!")
        return "0.0"
    featureString=request.args.get('featureString').strip('"')
    app.logger.info(featureString)
    fea_list=featureString.split(' ')
    fea_dict={}
    for fea in fea_list:
        print(fea)
        fea_dict[int(fea.split(':')[0])]=float(fea.split(':')[1])
    print(fea_dict)
    score=lr_model.get_predict(fea_dict)
    return '{"score": %f}' % score
    #str(lr_model.get_predict(fea_dict))

@app.route('/signin', methods=['POST'])
def signin():
    # 需要从request对象读取表单内容：
    if request.form['username']=='admin' and request.form['password']=='password':
        return '<h3>Hello, admin!</h3>'
    return '<h3>Bad username or password.</h3>'



if __name__ == '__main__':
    # if len(sys.argv)<1:
    #     os._exit(0)
    #     exit(0)
    # model_path=sys.argv[1]
    # if model_path==None:
    #     exit(0)
    handler = logging.FileHandler('server.log', encoding='UTF-8')
    logging_format = logging.Formatter('[%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s -] %(message)s')
    handler.setFormatter(logging_format)
    app.logger.addHandler(handler)
    lr_model=LogisticRegression()
    weight_path='D:\\push_platform\\weight_file_201805070904'
    mapping='D:\\push_platform\\feature_map_file_201805070904'
    lr_model.load_model(weight_path, mapping)
    app.run(debug=True, host='0.0.0.0')
    pass

