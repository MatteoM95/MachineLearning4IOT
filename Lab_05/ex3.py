import cherrypy
import json

class Calculator:
    exposed = True

    def GET(self, *path, **query):

        if len(path) != 1:
            raise cherrypy.HTTPError(400, "Wrong path")
        if len(query) != 2:
            raise cherrypy.HTTPError(400, "Wrong query")

        operation = path[0]
        op1 = query.get("op1")
        op2 = query.get("op2")

        if not op1 or not op2:
            raise cherrypy.HTTPError(400, "Missing arguments")
        else:
            op1, op2 = float(op1), float(op2)
        
        
        if operation == "add":
            result = op1 + op2

        elif operation == "sub":
            result = op1 - op2

        elif operation == "mul":
            result = op1 * op2

        elif operation == "div":
            if op2 == 0:
                raise cherrypy.HTTPError(400, "Can not divide per zero")
            result = op1 / op2

        else:
            raise cherrypy.HTTPError(400, "Wrong operation")
            
        output = {"command":operation,"op1":op1,"op2":op2,"result":result}
        output_json = json.dumps(output)
        return output_json

    def PUT(self, *path, **query):
        body_string = cherrypy.request.body.read()
        body_dict = json.loads(body_string)
        operation = body_dict["command"]
        op_list = body_dict["operands"]

        # initialize the result
        if operation == 'add':
            result = 0
        elif operation in ['sub','mul','div']:
            result = op_list[0]
            op_list = op_list[1:]
        else:
            raise cherrypy.HTTPError(400, "Wrong operation!")

        # compute the result
        for curr_op in op_list:

            if operation == 'add':
                result += curr_op
            elif operation == 'sub':
                result -= curr_op
            elif operation == 'mul':
                result *= curr_op
            elif operation == 'div':
                if curr_op == 0:
                    raise cherrypy.HTTPError(400, "Cannot divide by zero!")
                else:
                    result /= curr_op

        output = {"command":operation,"op":op_list,"result":result}
        output_json = json.dumps(output)
        return output_json
        
def main():
    conf = {"/":{"request.dispatch": cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Calculator(), "", conf)
    cherrypy.engine.start()
    cherrypy.engine.block()    


if __name__ == "__main__":
    main()