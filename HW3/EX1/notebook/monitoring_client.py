import time
from datetime import datetime
from MyMQTT import MyMQTT


class Monitor:
    def __init__(self, clientID):
        # create an instance of MyMQTT class
        self.clientID = clientID
        self.myMqttClient = MyMQTT(self.clientID, "test.mosquitto.org", 1883, self)

    def run(self):
        # if needed, perform some other actions befor starting the mqtt communication
        print(f"Running {self.clientID}")
        self.myMqttClient.start()

    def end(self):
        # if needed, perform some other actions befor ending the software
        print(f"Ending {self.clientID}")
        self.myMqttClient.stop()

    def notify(self, topic, msg):
        # manage here your received message. You can perform some error-check here
        now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        print(f"({now}) {msg}")


if __name__ == "__main__":
    monitor = Monitor("Monitoring Client")
    monitor.run()
    monitor.myMqttClient.mySubscribe("/alerts")

    a = 0
    while True:
        time.sleep(1)

    # monitor.end()
