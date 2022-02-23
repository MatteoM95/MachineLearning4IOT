import random 
import string 
import cherrypy 

class StringGeneratorWebService(object): 
	exposed = True 


	def GET (self, *uri, **params):
		return ("URI: %s; Parameters %s" % (str (uri), str(params)))

	def POST (self, *uri, **params): 
		mystring = "POST RESPONSE\n"
		mystring += ("URI: %s; Parameters %s\n" % (str (uri), str(params)))
		mystring += ("BODY: %s" % cherrypy.request.body.read())
		return mystring

	def PUT (self, *uri, **params): 
		mystring = "PUT RESPONSE\n"
		mystring += ("URI: %s; Parameters %s\n" % (str (uri), str(params)))
		mystring += ("BODY: %s" % cherrypy.request.body.read())
		return mystring
	
	def DELETE (self): 
		pass

if __name__ == '__main__':
	conf = {
		'/': {
			'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
			'tools.sessions.on': True
		}
	}
	cherrypy.tree.mount(StringGeneratorWebService(), '/string', conf)
	cherrypy.config.update({'server.socket_host': '0.0.0.0'})
	cherrypy.config.update({'server.socket_port': 8080})
	cherrypy.engine.start()
	cherrypy.engine.block()



