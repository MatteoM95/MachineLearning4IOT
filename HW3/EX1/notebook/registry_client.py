import json
import requests
import base64

with open("../../../tfliteCNN.tflite", "rb") as tflite_model:
    encoded_model = base64.b64encode(tflite_model.read())

body = {'model': encoded_model.decode("utf-8"), 'name': "tfliteCNN.tflite"}
r = requests.put('http://0.0.0.0:8080/add', json=body)
if r.status_code == 200:
    print(r.content.decode("utf-8"))
else:
    print("Error with model storing")
    exit(-1)

r = requests.get('http://0.0.0.0:8080/list')
if r.status_code == 200:
    print(r.json()['models'])
else:
    print("Error while getting the models' list")
    exit(-1)

r = requests.post('http://0.0.0.0:8080/request')
if r.status_code == 200:
    print(r.json())
else:
    print("Error starting predict function")
    exit(-1)
