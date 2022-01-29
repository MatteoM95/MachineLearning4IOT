import json
import requests
import base64

with open("../kws_dscnn_True.tflite", "rb") as tflite_model:
    encoded_model = base64.b64encode(tflite_model.read())

body = {'model': encoded_model.decode("utf-8"), 'name': "kws_dscnn_True.tflite"}
r = requests.put('http://0.0.0.0:8080/add', json=body)
print(r.content.decode("utf-8"))

with open("../kws_dscnn_False.tflite", "rb") as tflite_model:
    encoded_model = base64.b64encode(tflite_model.read())

body = {'model': encoded_model.decode("utf-8"), 'name': "kws_dscnn_False.tflite"}
r = requests.put('http://0.0.0.0:8080/add', json=body)
print(r.content.decode("utf-8"))

r = requests.get('http://0.0.0.0:8080/list')
print(r.json())
