import json
import time
from datetime import datetime
import numpy as np
import paho.mqtt.client as PahoMQTT
import tensorflow as tf
import adafruit_dht
from board import D4


class Alerts:
    def __init__(self, clientID):
        self.clientID = clientID

        self._paho_mqtt = PahoMQTT.Client(self.clientID, False)
        self._paho_mqtt.on_connect = self.myOnConnect

        self.messageBroker = 'test.mosquitto.org'

    def start(self):
        self._paho_mqtt.connect(self.messageBroker, 1883)
        self._paho_mqtt.loop_start()

    def stop(self):
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()

    def myPublish(self, topic, message):
        self._paho_mqtt.publish(topic, message, 2)

    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print(f"Connected to {self.messageBroker} with result code: {rc}")


def begin(model, tthresh, hthresh):
    alerts = Alerts("Temperature/Humidity Alerts")
    alerts.start()

    interpreter = tf.lite.Interpreter(f"./models/{model}")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    window = np.zeros([1, 6, 2], dtype=np.float32)
    MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
    STD = np.array([8.654227, 16.557089], dtype=np.float32)

    dht_device = adafruit_dht.DHT22(D4)
    i = 0
    while True:
        # Try except in order to manage occasional sensor failure
        try:
            temp = dht_device.temperature
            hum = dht_device.humidity
        except:
            time.sleep(2)
            temp = dht_device.temperature
            hum = dht_device.humidity

        if i < 6:
            window[0, i, 0] = np.float32(temp)
            window[0, i, 1] = np.float32(hum)
            i += 1
        else:
            y_true = np.array([np.float32(temp), np.float32(hum)], dtype=np.float32)

            window = (window - MEAN) / STD
            interpreter.set_tensor(input_details[0]['index'], window)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index']).reshape(2, )

            abs_error = np.abs(prediction - y_true)
            print(abs_error)

            if abs_error[0] > tthresh:
                response = {
                    "bn": "Temperature Alert",
                    "bt": int(datetime.now().timestamp()),
                    "e": [
                        {"n": "pred", "u": "°C", "t": 0, "v": str(prediction[0])},
                        {"n": "actual", "u": "°C", "t": 0, "v": str(y_true[0])}
                    ]
                }
                alerts.myPublish("/alerts", json.dumps(response))
            if abs_error[1] > hthresh:
                response = {
                    "bn": "Humidity Alert",
                    "bt": int(datetime.now().timestamp()),
                    "e": [
                        {"n": "pred", "u": "%", "t": 0, "v": str(prediction[1])},
                        {"n": "actual", "u": "%", "t": 0, "v": str(y_true[1])}
                    ]
                }
                alerts.myPublish("/alerts", json.dumps(response))

            window[:, 0:5, :] = window[:, 1:6, :]
            window[:, -1, 0] = y_true[0]
            window[:, -1, 1] = y_true[1]

        time.sleep(1)

    alerts.stop()
