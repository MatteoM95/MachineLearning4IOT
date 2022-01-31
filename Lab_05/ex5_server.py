import cherrypy
import json
import base64
from board import D4
import adafruit_dht
import datetime
import pyaudio

class Sensors():

    exposed = True

    def __init__(self):
        super().__init__()
        self.dht_device = adafruit_dht.DHT11(D4)

        self.rate = 48000
        self.frames_per_buffer = 4800
        self.seconds = 1.0
        
        audio = pyaudio.PyAudio()
        self.stream = audio.open(format=pyaudio.paInt16, rate=self.rate, 
                                 channels=1, input=True, frames_per_buffer=self.frames_per_buffer)
        self.stream.stop_stream()

        return

    def GET(self, *path, **query):        
        # get datetime informations
        now = datetime.datetime.now()
        timestamp = int(now.timestamp())
        
        # DHT11 informations
        temperature = self.dht_device.temperature
        humidity = self.dht_device.humidity

        # recording + base 64 encoding
        frames = []
        self.stream.start_stream()
        for _ in range(0, int(self.rate / self.frames_per_buffer * self.seconds)):
            data = self.stream.read(self.frames_per_buffer)
            frames.append(data)
        self.stream.stop_stream()

        audio_b64bytes = base64.b64encode(b"".join(frames))
        audio_string = audio_b64bytes.decode()


        body = {
            "host": "localhost",
            "timestamp": timestamp,
            "measures": [
                {"what": "temperature", "unit": "Cel", "value": temperature},
                {"what": "humidity", "unit": "%RH", "value": humidity},
                {"what": "audio", "unit": "_", "value": audio_string}
            ]
        }

        body = json.dumps(body)

        return body


def main():
    conf = {"/":{"request.dispatch": cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(Sensors(), "", conf)
    cherrypy.engine.start()
    cherrypy.engine.block()    


if __name__ == "__main__":
    main()