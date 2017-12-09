#!/usr/bin/python
from app import app

#!flask/bin/python
from flask import Flask, jsonify

app = Flask(__name__)

tasks = [
        {
            'id': 1,
            'title': u'Buy groceries',
            'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
            'done': False
                },
        {
            'id': 2,
            'title': u'Learn Python',
            'description': u'Need to find a good Python tutorial on the web',
            'done': False
                }
    ]
from flask import request
@app.route('/todo/api/v1.0/tasks', methods=['POST'])
def create_task():
    if not request.json or not 'title' in request.json:
                abort(400)
    task = {
                        'id': tasks[-1]['id'] + 1,
                        'title': request.json['title'],
                        'description': request.json.get('description', ""),
                        'done': False,
                             'text': "spasido za obraewenie dosvidania" 
                            }
    tasks.append(task)
    return jsonify({'task': task}), 201
                
@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])

def get_task(task_id):
    task = filter(lambda t: t['id'] == task_id, tasks)
    if len(task) == 0:
        abort(404)
    return jsonify({'task': task[0]})


app.run(debug = True, host = "oh.rpg.ru")
