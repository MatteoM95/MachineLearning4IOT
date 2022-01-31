import json

# example - string to json 
string = '{"name": "Tony", "surname": "Stark"}'
obj = json.loads(string)

# example - json to string
obj_2 = {"num1": 12,"num2": 34}
string_v2 = json.dumps(obj)
