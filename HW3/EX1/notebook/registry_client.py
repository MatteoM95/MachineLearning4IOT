import argparse
import json
import requests
import base64

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, required=False, default="127.0.0.1", help='IP of registry service')
parser.add_argument('--port', type=int, required=False, default="8080", help='Port of registry service')
args = parser.parse_args()


def addModel(model_name):
    with open(model_name, "rb") as tflite_model:
        encoded_model = base64.b64encode(tflite_model.read())

    body = {'model': encoded_model.decode("utf-8"), 'name': model_name}
    r = requests.put(f'http://{args.ip}:{args.port}/add', json=body)
    if r.status_code == 200:
        print(f"Model STORED -> {r.content.decode('utf-8')}")
    else:
        print("Error with model storing")
        exit(-1)
    pass


def getModelList():
    r = requests.get(f'http://{args.ip}:{args.port}/list')
    if r.status_code == 200:
        print(f"Model LIST -> {r.json()['models']}")
    else:
        print("Error while getting the models' list")
        exit(-1)


def predict():
    r = requests.get(f'http://{args.ip}:{args.port}/predict/?model=cnn.tflite&tthresh=0.1&hthresh=0.2')
    if r.status_code == 200:
        print(r.json()['response'])
    else:
        print(f"Error with predicting function {r.text}")
        exit(-1)


def main():
    mlp_model_path = 'mlp.tflite'
    cnn_model_path = 'cnn.tflite'

    addModel(mlp_model_path)
    addModel(cnn_model_path)
    getModelList()
    predict()


if __name__ == '__main__':
    main()
