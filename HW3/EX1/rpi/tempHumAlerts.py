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
    MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
    STD = np.array([8.654227, 16.557089], dtype=np.float32)

    dht_device = adafruit_dht.DHT11(D4)
    while True:
        window = np.zeros([1, 6, 2], dtype=np.float32)
        i = 0
        if i < 6:
            # Try except in order to manage occasional sensor failure
            try:
                window[0, i, 0] = dht_device.temperature
                window[0, i, 1] = dht_device.humidity
                time.sleep(1)
                i += 1
            except:
                pass
        else:
            # Try except in order to manage occasional sensor failure
            try:
                y_true = np.array([dht_device.temperature, dht_device.humidity])
            except:
                time.sleep(2)
                y_true = np.array([dht_device.temperature, dht_device.humidity])

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

    alerts.stop()
