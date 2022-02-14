import argparse
import base64
import os
import cherrypy
import json
import tempHumAlerts


class AddService:
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
        encoded_model = body.get('model')
        model_path = f"./models/{body.get('name')}"

        model = base64.b64decode(encoded_model.encode('utf-8'))

        if not os.path.exists("./models"):
            os.mkdir("./models")
        with open(model_path, "wb") as m:
            m.write(model)

        return os.path.abspath(model_path)

    def DELETE(self, *path, **query):
        pass


class ListService:
    exposed = True

    def GET(self, *path, **query):
        if len(path) > 0:
            raise cherrypy.HTTPError(400, 'Wrong path')
        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Wrong query')

        response = {'models': os.listdir("./models")}

        return json.dumps(response)

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


class PredictService:
    exposed = True

    def GET(self, *path, **query):
        # if len(path) > 0:
        #     raise cherrypy.HTTPError(400, 'Wrong path')
        # if len(query) != 3 or \
        #         'model' not in query or \
        #         'tthresh' not in query or \
        #         'hthresh' not in query:
        #     raise cherrypy.HTTPError(400, 'Wrong parameters')

        params = query
        tempHumAlerts.begin(model=params['model'],
                            tthresh=float(params['tthresh']),
                            hthresh=float(params['hthresh']))

        return json.dumps({'response': "OK"})

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=False, default="127.0.0.1", help='IP of slow_service')
    parser.add_argument('--port', type=int, required=False, default="8080", help='Port of slow_service')
    args = parser.parse_args()

    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(AddService(), '/add', conf)
    cherrypy.tree.mount(ListService(), '/list', conf)
    cherrypy.tree.mount(PredictService(), '/predict', conf)

    cherrypy.config.update({'server.socket_host': f'{args.ip}'})
    cherrypy.config.update({'server.socket_port': args.port})
    cherrypy.engine.start()
    cherrypy.engine.block()
