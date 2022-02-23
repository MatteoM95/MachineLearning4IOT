import cherrypy
import json
from functools import reduce


class Calculator(object):
    exposed = True

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')

        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')

        body = cherrypy.request.body.read()
        body = json.loads(body)

        operation = body.get('command')

        if operation is None:
            raise cherrypy.HTTPError(400, 'command missing')

        operands = body.get('operands')
        if operands is None:
            raise cherrypy.HTTPError(400, 'operands missing')

        if len(operands) < 1:
            raise cherrypy.HTTPError(400, 'number of operands must be > 0')

        if operation == 'add':
            result = reduce((lambda x, y: float(x) + y), operands)
        elif operation == 'sub':
            result = reduce((lambda x, y: float(x) - y), operands)
        elif operation == 'mul':
            result = reduce((lambda x, y: float(x) * y), operands)
        elif operation == 'div':
            if 0 in operands[1:]:
                raise cherrypy.HTTPError(400, 'found division by 0')
            result = reduce((lambda x, y: float(x) / y), operands)
        else:
            raise cherrypy.HTTPError(400, 'Wrong operation')

        output = {
                    'command': operation,
                    'operands': operands,
                    'result': result
                }

        output_json = json.dumps(output)

        return output_json

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Calculator(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
