import json
import time
from datetime import datetime
import paho.mqtt.client as PahoMQTT


class Monitor:
    def __init__(self, clientID):
        self.topic = None
        self.clientID = clientID

        self._paho_mqtt = PahoMQTT.Client(clientID, False)

        self._paho_mqtt.on_connect = self.myOnConnect
        self._paho_mqtt.on_message = self.myOnMessageReceived

        self.messageBroker = 'test.mosquitto.org'

    def start(self):
        self._paho_mqtt.connect(self.messageBroker, 1883)
        self._paho_mqtt.loop_start()

    def subscribe(self, topic):
        self.topic = topic
        self._paho_mqtt.subscribe(self.topic, 2)

    def stop(self):
        self._paho_mqtt.unsubscribe(self.topic)
        self._paho_mqtt.loop_stop()
        self._paho_mqtt.disconnect()

    def myOnConnect(self, paho_mqtt, userdata, flags, rc):
        print(f"Connected to {self.messageBroker} with result code: {rc}")

    def myOnMessageReceived(self, paho_mqtt, userdata, msg):
        message = json.loads(msg.payload.decode('utf-8'))

        timestamp = datetime.fromtimestamp(message['bt']).strftime('%d/%m/%Y %H:%M:%S')
        alert_name = message['bn']
        pred_value = message['e'][0]['v']
        pred_value_unit = message['e'][0]['u']
        actual_value = message['e'][1]['v']
        actual_value_unit = message['e'][1]['u']

        print(f"({timestamp}) {alert_name}: Predicted={pred_value}{pred_value_unit} "
              f"Actual={actual_value}{actual_value_unit}")


if __name__ == "__main__":
    monitor = Monitor("Monitoring Client")
    monitor.start()
    monitor.subscribe("/alerts")

    while True:
        time.sleep(1)

    # monitor.end()
