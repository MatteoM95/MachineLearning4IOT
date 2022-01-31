from DoSomething import DoSomething
import time
import json
from datetime import datetime

from board import D4
import adafruit_dht
import base64
import pyaudio


class CollectorClient(DoSomething):

    def __init__(self, clientID):
        audio = pyaudio.PyAudio()
        self.stream = audio.open(format=pyaudio.paInt16, rate=48000, channels=1, input=True, frames_per_buffer=4800)
        self.stream.stop_stream()

    def notify(self, topic, msg):
        json_body = json.loads(msg)
        events = json_body["e"]

        for event in events:
            if event["n"] == "record":
                if event["vb"]:
                    
                    now = datetime.datetime.now()
                    timestamp = int(now.timestamp())

                    self.stream.start_stream()    
                    rate = 48000
                    buffer = 4800
                    frames = []
                    for _ in range(int(buffer // rate)): # 10 in this case
                        frames.append(self.stream.read(4800))
                    self.stream.stop_stream()

                    audio_b64bytes = base64.b64encode(b"".join(frames))
                    audio_string = audio_b64bytes.decode()

                    out_body = {
                        "bn": "192.168.1.9",
                        "bt": timestamp,
                        "e": [
                            {"n":"audio", "u":"/", "t":0, "vd":audio_string}
                        ]
                    }

                    out_json = json.dumps(out_body)
                    self.myMqttClient.myPublish("/206803/audio", out_json)

if __name__ == "__main__":
    
    test = DoSomething("Board publisher")
    test.run()

    dht_device = adafruit_dht.DHT11(D4)

    # every 10 seconds temp and humidity after 20 seconds

    ten = True
    for i in range(10):
        now = datetime.now()
        
        timestamp = str((now.timestamp()))

        if ten is False:
            # every 20 seconds
            ten = True

            humidity = dht_device.humidity

            body_hum = {
				"bn": "http://192.168.1.9/",
				"bt": timestamp,
				"e":[
					{"n": "humidity", "u":"RH", "t":0, "v":humidity},
				]
        	}
            
            body_hum_json = json.dumps(body_hum)
            test.myMqttClient.myPublish("/206803/humidity", body_hum_json)

        else:
            ten = False

        temperature = dht_device.temperature

        body_temp = {
                "bn": "http://192.168.1.9/",
                "bt": timestamp,
                "e":[
                    {"n": "temperature", "u":"C", "t":0, "v":temperature},
                ]
        }

        body_temp_json = json.dumps(body_temp)
        test.myMqttClient.myPublish("/206803/humidity", body_temp_json)

        time.sleep(10)

    test.end()
    