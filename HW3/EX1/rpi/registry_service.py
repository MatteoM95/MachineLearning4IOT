import argparse
import base64
import os
import cherrypy
import json
import tempHumAlerts


class ModelRegistry:
    exposed = True

    def GET(self, *path, **query):
        if len(path) != 1 or path[0] != "list":
            raise cherrypy.HTTPError(400, 'Wrong path')

        response = {'models': os.listdir("./models")}

        return json.dumps(response)

    def POST(self, *path, **query):
        if len(path) != 1 or path[0] != "request":
            raise cherrypy.HTTPError(400, 'Wrong path')
        if len(query) != 3 or \
                'model' not in query or \
                'tthresh' not in query or \
                'hthresh' not in query:
            raise cherrypy.HTTPError(400, 'Wrong number of parameter')

        params = query
        tempHumAlerts.begin(model=params['model'],
                            tthresh=float(params['tthresh']),
                            hthresh=float(params['hthresh']))

        return json.dumps({'response': "OK"})

    def PUT(self, *path, **query):
        if len(path) != 1 or path[0] != "add":
            raise cherrypy.HTTPError(400, 'Wrong path')

        body = cherrypy.request.body.read()
        body = json.loads(body)

        encoded_model = body['model']
        model_name = f"./models/{body.get('name')}"

        model = base64.b64decode(encoded_model.encode('utf-8'))

        if not os.path.exists("./models"):
            os.mkdir("./models")
        with open(model_name, "wb") as m:
            m.write(model)

        return os.path.abspath(model_name)

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=False, default="127.0.0.1", help='IP of slow_service')
    parser.add_argument('--port', type=int, required=False, default="8080", help='Port of slow_service')
    args = parser.parse_args()

    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(ModelRegistry(), '', conf)
    cherrypy.config.update({'server.socket_host': f'{args.ip}'})
    cherrypy.config.update({'server.socket_port': args.port})
    cherrypy.engine.start()
    cherrypy.engine.block()
