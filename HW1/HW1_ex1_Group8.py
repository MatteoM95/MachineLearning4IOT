# command line:python3 HW1_ex1_Group8.py --input ../datasets --output th.tfrecord --normalize

import os
import argparse
import tensorflow as tf
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./datasets/tempHumi.csv",
                    help="Input File Name. Default: ./tempHumi.csv")
parser.add_argument("--output", type=str, default="./th.tfrecord", help="Output File Name. Default: ./th.tfrecord")
parser.add_argument("--normalize", action='store_true', help="Normalize Temperature? Insert --normalize")
args = parser.parse_args()

output_filename = args.output
input_dir = args.input
normalize = args.normalize

# sensor range temperature and humidity
t_Min = 0
t_Max = 50
h_Min = 20
h_Max = 90

#normalization function
def normalize_func(temp, humi):
    global t_Min, t_Max, h_Min, h_Max
    normTemp = (int(temp) - t_Min) / (t_Max - t_Min)
    normHumi = (int(humi) - h_Min) / (h_Max - h_Min)

    return normTemp, normHumi


def main():
    with tf.io.TFRecordWriter(output_filename) as writer:

        with open(f"{input_dir}/tempHumiHW1.csv", "r") as fp:
            lines = fp.readlines()

            for line in lines:
                if not line.isspace() and not line.startswith("date"):
                    date, time, temperature, humidity = line.strip().split(",")
                    day, month, year = date.split("/")
                    hour, minutes, seconds = time.split(":")

                    timestamp = datetime(int(year), int(month), int(day), int(hour), int(minutes),
                                         int(seconds)).timestamp()
                    # Normalize temperature and humidity if required
                    if normalize:
                        temperature, humidity = normalize_func(temperature, humidity)

                        # Conversion to best format for saving HDD space
                        temperature_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(temperature)]))
                        humidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(humidity)]))
                        timestamp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(timestamp)]))
                        # temperature_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[temperature]))
                        # humidity_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[humidity]))
                        # timestamp_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[timestamp]))
                    # without normalization
                    else:
                        # Conversion to best format for saving HDD space
                        timestamp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(timestamp)]))
                        temperature_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(temperature)]))
                        humidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(humidity)]))
                        # timestamp_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[timestamp]))
                        # temperature_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[temperature]))
                        # humidity_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[humidity]))
                        # humidity_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str.encode(humidity)]))
                        # temperature_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[
                        # str.encode(temperature)]))

                    mapping = {'timestamp': timestamp_feature,
                               'temperature': temperature_feature,
                               'humidity': humidity_feature
                               }
                    payload = tf.train.Example(features=tf.train.Features(feature=mapping))
                    writer.write(payload.SerializeToString())

    print(f"{os.path.getsize(output_filename)}B")


if __name__ == '__main__':
    main()
