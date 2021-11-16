# command line:python3 HW1_ex1_Group8.py --input ../datasets --output th.tfrecord --normalize

import os
import argparse
import tensorflow as tf
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./datasets/dataset.csv",
                    help="Input File Name. Default: ./dataset.csv")
parser.add_argument("--output", type=str, default="./th.tfrecord", help="Output File Name. Default: ./th.tfrecord")
parser.add_argument("--normalize", action='store_true', help="Normalize Temperature? Insert --normalize")
args = parser.parse_args()

output_filename = args.output
input_dir = args.input
normalize = args.normalize


def normalize_func(temp, humi):
    normTemp = (int(temp) - 0) / (50 - 0)
    normHumi = (int(humi) - 20) / (90 - 20)

    return normTemp, normHumi


def main():
    with tf.io.TFRecordWriter(output_filename) as writer:

        with open(f"{input_dir}/dataset.csv", "r") as fp:
            lines = fp.readlines()

            for line in lines:
                if not line.isspace() and not line.startswith("date"):
                    date, time, temperature, humidity = line.strip().split(",")
                    day, month, year = date.split("/")
                    hour, minutes, seconds = time.split(":")

                    timestamp = datetime(int(year), int(month), int(day), int(hour), int(minutes),
                                         int(seconds)).timestamp()

                    # Normalization of temperature if required
                    if normalize:
                        # print("Normalizing...")
                        temperature, humidity = normalize_func(temperature, humidity)

                        temperature_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(temperature)]))
                        humidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(humidity)]))
                        timestamp_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[timestamp]))
                    else:
                        # Conversion to best format for saving HDD space
                        # timestamp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(timestamp)]))
                        temperature_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(temperature)]))
                        humidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(humidity)]))
                        timestamp_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[timestamp]))
                        # temperature_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[temperature]))
                        # humidity_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[humidity]))
                        # humidity_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(humidity)]))
                        # temperature_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                        # str.encode(temperature)]))

                    # print(type(timestamp)) #float
                    # print(type(humidity)) #float
                    # print(type(temperature)) #float

                    # best normalizzazione (timestamp - float)(temperature - int64)(humidity - int64)

                    mapping = {'timestamp': timestamp_feature, \
                               'temperature': temperature_feature, \
                               'humidity': humidity_feature
                               }
                    payload = tf.train.Example(features=tf.train.Features(feature=mapping))
                    writer.write(payload.SerializeToString())

    print(f"{os.path.getsize(output_filename)}B")


if __name__ == '__main__':
    main()
