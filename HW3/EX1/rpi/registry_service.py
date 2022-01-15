import base64
import os
import cherrypy
import json


class ModelRegistry:
    exposed = True

    def GET(self, *path, **query):
        if len(path) != 1 or path[0] != "list":
            raise cherrypy.HTTPError(400, 'Wrong path')

        response = {'models': os.listdir("./models")}

        return json.dumps(response)

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        if len(path) != 1 or path[0] != "add":
            raise cherrypy.HTTPError(400, 'Wrong path')

        body = cherrypy.request.body.read()
        body = json.loads(body)

        encoded_model = body.get("model")
        model_name = f"./models/{body.get('name')}"

        model = base64.b64decode(encoded_model)

        if not os.path.exists("./models"):
            os.mkdir("./models")
        with open(model_name, "wb") as m:
            m.write(model)

        return os.path.abspath(model_name)

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(ModelRegistry(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
