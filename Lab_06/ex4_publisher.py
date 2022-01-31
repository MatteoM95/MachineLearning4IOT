from DoSomething import DoSomething
import time
import json
from datetime import datetime

from board import D4
import adafruit_dht


if __name__ == "__main__":
    test = DoSomething("publisher 1")
    test.run()

    dht_device = adafruit_dht.DHT11(D4)

    # available for 10 minutes
    for i in range(10):
        
        now = datetime.now()

        events = []
        for _ in range(6):
            temp_event = {"n": "temperature", "u":"C", "t":0, "v":dht_device.temperature}
            hum_event = {"n": "humidity", "u":"RH", "t":0, "v":dht_device.humidity}
            events.append(temp_event)
            events.append(hum_event)
            time.sleep(10) # one recording every 10 seconds


        datetime_str = str(now.strftime('%d-%m-%y %H:%M:%S'))
        
        body = {
				"bn": "http://192.168.1.9/",
				"bt": datetime_str,
				"e": events
        }
            
        body_json = json.dumps(body)
        test.myMqttClient.myPublish("/206803/dht11", body_json)

        print("\n")

    test.end()
