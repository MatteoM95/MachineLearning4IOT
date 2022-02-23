import cherrypy
import json


class Calculator(object):
    exposed = True

    def GET(self, *path, **query):

        if len(path) != 1:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 2:
            raise cherrypy.HTTPError(400, 'Wrong query')

        operation = path[0]

        operand1 = query.get('op1')
        if operand1 is None:
            raise cherrypy.HTTPError(400, 'op1 missing')
        else:
            operand1 = float(operand1)

        operand2 = query.get('op2')
        if operand2 is None:
            raise cherrypy.HTTPError(400, 'op2 missing')
        else:
            operand2 = float(operand2)

        if operation == 'add':
            result = operand1 + operand2
        elif operation == 'sub':
            result = operand1 - operand2
        elif operation == 'mul':
            result = operand1 * operand2
        elif operation == 'div':
            if operand2 == 0:
                raise cherrypy.HTTPError(400, 'op2 must be different form 0')

            result = operand1 / operand2
        else:
            raise cherrypy.HTTPError(400, 'Wrong operation')

        output = {
                    'command': operation,
                    'op1': operand1,
                    'op2': operand2,
                    'result': result
                }

        output_json = json.dumps(output)

        return output_json

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Calculator(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
