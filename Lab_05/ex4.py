import argparse
import requests

def main(args):

    command_list = {'add' : '+', 'sub' : '-', 'mul' : '*', 'div' : ':'}    

    command = args.command
    operands = args.operands

    if len(operands) != 2:
        raise ValueError('You have to select two operands!')

    op1, op2 = operands[0], operands[1]

    url = f"http://127.0.0.1:8080/{command}?op1={op1}&op2={op2}"
    
    r = requests.get(url)

    if r.status_code == 200:
        body = r.json()
        output = '{} {} {} = {}'.format(body['op1'], command_list[body['command']], body['op2'], body['result'])
    else:
        output = 'Error {} : {}'.format(r.status_code, r.text)

    print(output)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--command', type=str, required=True, choices=['add','mul','div','sub'])
    parser.add_argument('-o','--operands', nargs='+', type=float, required=True)
    args = parser.parse_args()

    main(args)