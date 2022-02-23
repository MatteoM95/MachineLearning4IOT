import argparse
import requests


parser = argparse.ArgumentParser()
parser.add_argument('command', nargs=1, type=str)
parser.add_argument('op1', nargs=1, type=float)
parser.add_argument('op2', nargs=1, type=float)
args = parser.parse_args()


command = args.command[0]
op1 = args.op1[0]
op2 = args.op2[0]


url = 'http://0.0.0.0:8080/{}?op1={}&op2={}'.format(command, op1, op2)

r = requests.get(url)

commands = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}

if r.status_code == 200:
    body = r.json()
    print(body['op1'], commands[body['command']], body['op2'], '=',
            body['result'])
else:
    print('Error:', r.status_code)
