from DoSomething import DoSomething
from datetime import datetime
import time
import json


class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)

        timestamp = input_json['timestamp']
        now = datetime.fromtimestamp(float(timestamp))
        datetime_str = now.strftime('%d-%m-%y %H:%M:%S')
        
        print(topic, datetime_str, sep="\t")


if __name__ == "__main__":
    test = Subscriber("subscriber 2")
    test.run()
    test.myMqttClient.mySubscribe("/206803/timestamp")

    while True:
        time.sleep(1)
