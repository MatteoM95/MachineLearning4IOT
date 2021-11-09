import os
import argparse
import tensorflow as tf
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, default="./datasets/dataset.csv", help="Input File Name. Default: ./dataset.csv")
parser.add_argument("--output", type=str, default="./th.tfrecord", help="Output File Name. Default: ./th.tfrecord")
parser.add_argument("--normalize", default="False", help="Normalize Temperature? Insert --normalize")
args = parser.parse_args()

output_filename = args.output
input_dir = args.input
normalize = args.normalize

tMAX = 0
tMIN = 9999999

def normalize_func(tmp):
    if tMAX > tmp:
        tMAX = tmp
    if tMIN < tmp:
        tMIN = tmp
    norm = (tmp - tMIN) / (tMAX - tMIN)
    return norm

def main():
    with tf.io.TFRecordWriter(output_filename) as writer:
        with open(f"{input_dir}/dataset.csv", "r") as fp:
            lines = fp.readlines()
            for line in lines:
                if not line.isspace():
                    date, time, temperature, humidity = line.strip().split(",")
                    day, month, year = date.split("/")
                    hour, minutes, seconds = time.split(":")

                    timestamp = datetime(int(year), int(month), int(day), int(hour), int(minutes), int(seconds)).timestamp()

                    #Normalization of temperature if required
                    if normalize:
                        temperature = normalize_func(temperature)

                    #Conversion to best format for saving HDD space
                    timestamp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(timestamp)]))
                    temperature_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(temperature)]))
                    humidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(humidity)]))

                    mapping = {'timestamp': timestamp_feature, \
                               'humidity': humidity_feature
                                }
                    payload = tf.train.Example(features=tf.train.Features(feature=mapping))
                    writer.write(payload.SerializeToString())

    print(os.path.getsize(output_filename))

if __name__ == '__main__':
    main()
