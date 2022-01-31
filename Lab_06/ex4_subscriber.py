import datetime
from DoSomething import DoSomething
import time
import json

import tensorflow as tf
import numpy as np



class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        
        datetime = input_json['datetime']

        # the batch must be:
        # [[t1,...,t6],[h1,...,h6]]
        temperatures, humidities = [], []
        for event in input_json['e']:
            
            if event['n'] == 'temperature':
                temperatures.append(event['v'])
            else:
                humidities.append(event['v'])

        interpreter = tf.lite.Interpreter(model_path='./models/mlp_temp_hum_6.tflite')
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        window = np.zeros([1, 6, 2], dtype=np.float32)

        MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
        STD = np.array([ 8.654227, 16.557089], dtype=np.float32)

        for i in range(7): # the last one is the prediction
            temperature = temperatures[i]
            humidity = humidities[i]

            if i < 6:
                window[0, i, 0] = np.float32(temperature)
                window[0, i, 1] = np.float32(humidity)
            if i == 6:

                window = (window - MEAN) / STD
                interpreter.set_tensor(input_details[0]['index'], window)
                interpreter.invoke()
                
                predicted = interpreter.get_tensor(output_details[0]['index'])

                print('Datetime = {}'.format(datetime))
                print('Previous temperatures = {}'.format(temperatures))
                print('Previous humidities = {}'.format(humidities))
                print('\n')
                print('Predicted temperature = {:1.f}'.format(predicted[0, 0]))
                print('Predicted humidity = {:1.f}'.format(predicted[0, 1]))


if __name__ == "__main__":
    test = Subscriber("subscriber 1")
    test.run()
    test.myMqttClient.mySubscribe("/206803/dht11")

    while True:
        time.sleep(1)
