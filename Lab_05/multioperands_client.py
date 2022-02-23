import argparse
import requests


parser = argparse.ArgumentParser()
parser.add_argument('command', nargs=1, type=str)
parser.add_argument('operands', nargs='+', type=float)
args = parser.parse_args()


command = args.command[0]
operands = args.operands


url = 'http://0.0.0.0:8080'

body = {'command': command, 'operands': operands}

r = requests.put(url, json=body)

commands = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}

if r.status_code == 200:
    body = r.json()
    string = commands[body['command']].join(map(str, operands))
    print(string, '=', body['result'])
else:
    print('Error:', r.status_code)
