#!/usr/bin/python3
# -*- coding: utf-8 -*-

#import adres
import calc
import hardcode
from hardcode import check_hardcode, check_hard_calc
from calc import casco
from app import app
import pandas as pd
#import echo
#!flask/bin/python
from flask import Flask, jsonify

app = Flask(__name__)

texts =  pd.read_csv('hardcoded_phrases.csv', names=['code', 'answer', 'frequency'])
from flask import request
@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def create_task():
    print(texts)
    print(texts[texts.code == "E"])
    print( request.json)
    hist = request.json
    ans = hist["lname"]
    chh = check_hardcode(request.json)   
    if chh != "":
        ans = {'text' : chh}
        resp = {
            'context': "none",
            'ans': ans
                                        }

        return jsonify({'resp': resp}), 201
    
    #calc = casco(request.json["messages"])
    if check_hard_calc(request.json) or request.json["context"]=="calc":
        import calc
        calc = casco(request.json["messages"])
        del calc
        calc = casco(request.json["messages"])
        print(calc)
        c_resp = calc.message()
        resp = {                      
                        'context': "none" if c_resp["exit"] else "calc",
                             'ans': c_resp 
                            }
    
        return jsonify({'resp': resp}), 201
    resp = {
        'context': u"none",
        'ans': {'text':u"Ваш вопрос перенаправлен специалисту"}
        }
    return jsonify({'resp':resp})
@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])

def get_task(task_id):
    task = filter(lambda t: t['id'] == task_id, tasks)
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})


app.run(debug = True, host = "oh.rpg.ru")
