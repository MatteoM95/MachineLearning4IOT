import argparse
import json
import requests
import base64

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, required=False, default="127.0.0.1", help='IP of registry service')
parser.add_argument('--port', type=int, required=False, default="8080", help='Port of registry service')
parser.add_argument('--model', type=str, required=False, default="tfliteCNN.tflite", help='Model name')
args = parser.parse_args()


def sendModel():
    with open(args.model, "rb") as tflite_model:
        encoded_model = base64.b64encode(tflite_model.read())

    body = {'model': encoded_model.decode("utf-8"), 'name': args.model}
    r = requests.put(f'http://{args.ip}:{args.port}/add', json=body)
    if r.status_code == 200:
        print(r.content.decode("utf-8"))
    else:
        print("Error with model storing")
        exit(-1)
    pass


def getModelList():
    r = requests.get(f'http://{args.ip}:{args.port}/list')
    if r.status_code == 200:
        print(r.json()['models'])
    else:
        print("Error while getting the models' list")
        exit(-1)


def predict():
    body = {'model': args.model, 'tthresh': 0.1, 'hthresh': 0.2}
    r = requests.post(f'http://{args.ip}:{args.port}/request', json=body)
    if r.status_code == 200:
        print(r.json()['response'])
    else:
        print("Error with predicting function")
        exit(-1)


def main():
    sendModel()
    getModelList()
    predict()


if __name__ == '__main__':
    main()
