import os
import argparse
import tensorflow as tf
from datetime import datetime
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./dataset", help="Input File Name. Default: ./raw_data")
parser.add_argument("--output", type=str, default="./th.tfrecord", help="Output File Name. Default: ./th.tfreord")
parser.add_argument("--normalize", type=str, default="False", help="Normalize Temperature: --normalize == True")
args = parser.parse_args()

output_filename = args.output
input_dir = args.input
normalize = args.normalize

tMAX = 0
tMIN = 9999999

def normalizeFunc(tmp):

    if(tMAX > tmp):
        tMAX = tmp
    if(tMIN < tmp)
        tMIN = tmp

    norm = (tmp - tMIN) / (tMAX - tMIN)
    return norm

def main():
    with tf.io.TFRecordWriter(output_filename) as writer:
        with open(f"{input_dir}/dataset.csv", "r") as fp:
            lines = fp.readlines()
            for line in lines:
                if not line.isspace():
                    date, time, temperature, humidity, audioName = line.strip().split(",")
                    day, month, year = date.split("/")
                    hour, minutes, seconds = time.split(":")

                    timestamp = datetime(int(year), int(month), int(day), int(hour), int(minutes), int(seconds)).timestamp()
                    audio_path = os.path.join(input_dir, audioName)
                    audio = tf.io.read_file(audio_path)

                    if(normalize):
                        normalizeFunc(temperature)


                    timestamp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(timestamp)]))
                    temperature_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(temperature)]))
                    humidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(humidity)]))
                    audio_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[audio.numpy()]))
                    mapping = {'timestamp': timestamp_feature, \
                               'temperature': temperature_feature, \
                               'humidity': humidity_feature, \
                               'audio': audio_feature}
                    payload = tf.train.Example(features=tf.train.Features(feature=mapping))
                    writer.write(payload.SerializeToString())

    print(os.path.getsize(output_filename))

if __name__ == '__main__':
    main()
