from DoSomething import DoSomething
import time
import json
from datetime import datetime


if __name__ == "__main__":
    test = DoSomething("publisher 1")
    test.run()

    ten = True
    for i in range(30):
        now = datetime.now()

        if ten is False:
            ten = True
        elif ten is True:
            ten = False
            timestamp = str((now.timestamp()))
            timestamp_json = json.dumps({'timestamp': timestamp})
            test.myMqttClient.myPublish("/206803/timestamp", timestamp_json)

        datetime_str = now.strftime('%d-%m-%y %H:%M:%S')
        datetime_json = json.dumps({'datetime': datetime_str})

        test.myMqttClient.myPublish ("/206803/datetime", datetime_json)

        print("\n")
        time.sleep(5)

    test.end()
