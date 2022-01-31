import adafruit_dht
import argparse
import numpy as np
import time
import tensorflow as tf
from board import D4


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()


interpreter = tf.lite.Interpreter(model_path='./models/{}.tflite'.format(args.model))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

window = np.zeros([1, 6, 2], dtype=np.float32)
expected = np.zeros(2, dtype=np.float32)

MEAN = np.array([9.107597, 75.904076], dtype=np.float32)
STD = np.array([ 8.654227, 16.557089], dtype=np.float32)

dht_device = adafruit_dht.DHT11(D4)

while True:
    for i in range(7):
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        if i < 6:
            window[0, i, 0] = np.float32(temperature)
            window[0, i, 1] = np.float32(humidity)
        if i == 6:
            expected[0] = np.float32(temperature)
            expected[1] = np.float32(humidity)

            window = (window - MEAN) / STD
            interpreter.set_tensor(input_details[0]['index'], window)
            interpreter.invoke()
            predicted = interpreter.get_tensor(output_details[0]['index'])

            print('\nMeasured: {:.1f},{:.1f}'.format(expected[0], expected[1]))
            print('Predicted: {:.1f},{:.1f}'.format(predicted[0, 0],
                predicted[0, 1]))

        time.sleep(0.2)
