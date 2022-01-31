from DoSomething import DoSomething
import time
import json

from board import D4
import adafruit_dht
import base64
import pyaudio
import wave


class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)

        # save audio 
        audio_bytes = base64.b64decode(input_json['e'][0]['vt'])
        output_audio = "{}.wav".format(input_json['bt'])
        wavefile = wave.open(output_audio, "wb")
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(48000)
        wavefile.writeframes(audio_bytes)
        wavefile.close()


if __name__ == "__main__":
    test = Subscriber("subscriber 2")
    test.run()
    test.myMqttClient.mySubscribe("/206803/audio")


    while True:
        time.sleep(1)
