import random
import string

import cherrypy

class StringGenerator(object):

    @cherrypy.expose
    def index(self):
        return "Hello world!"

    @cherrypy.expose
    def generate(self, length=8):
    	some_string = ''.join(random.sample(string.hexdigits, int(length)))
    	cherrypy.session['mystring'] = some_string
    	return some_string

    @cherrypy.expose
    def display(self):
        return cherrypy.session['mystring'] 

if __name__ == '__main__':
    conf = {
        '/': {
        'tools.sessions.on': True
        }
    }
    cherrypy.tree.mount (StringGenerator(), '/', conf)
    
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
