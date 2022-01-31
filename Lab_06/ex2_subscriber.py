from DoSomething import DoSomething
import time
import json


class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        print(topic,input_json['datetime'],sep="\t")


if __name__ == "__main__":
    test = Subscriber("subscriber 1")
    test.run()
    test.myMqttClient.mySubscribe("/206803/datetime")

    while True:
        time.sleep(1)
