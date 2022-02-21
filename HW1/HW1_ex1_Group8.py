import os
import argparse
import tensorflow as tf
from datetime import datetime


# Normalization function
def normalize_func(temp, humi):
    t_Min = 0
    t_Max = 50
    h_Min = 20
    h_Max = 90
    normTemp = (int(temp) - t_Min) / (t_Max - t_Min)
    normHumi = (int(humi) - h_Min) / (h_Max - h_Min)
    return normTemp, normHumi


def main(args):
    input_file = args.input
    output_file = args.output
    normalize = args.normalize

    with open(f"{input_file}", "r") as fp:
        lines = fp.readlines()

    with tf.io.TFRecordWriter(output_file) as writer:
        for line in lines:
            if not line.isspace() and not line.startswith("date"):
                date, time, temperature, humidity = line.strip().split(",")
                day, month, year = date.split("/")
                hour, minutes, seconds = time.split(":")

                timestamp = datetime(int(year), int(month), int(day), int(hour), int(minutes),
                                     int(seconds)).timestamp()

                timestamp_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(timestamp)]))

                if normalize:
                    temperature, humidity = normalize_func(temperature, humidity)

                    # Conversion to best format for saving HDD space and maintain data quality
                    temperature_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[float(temperature)]))
                    humidity_feature = tf.train.Feature(float_list=tf.train.FloatList(value=[float(humidity)]))

                else:
                    # Conversion to best format for saving HDD space and maintain data quality
                    temperature_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(temperature)]))
                    humidity_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(humidity)]))

                mapping = {'timestamp': timestamp_feature,
                           'temperature': temperature_feature,
                           'humidity': humidity_feature
                           }
                payload = tf.train.Example(features=tf.train.Features(feature=mapping))
                writer.write(payload.SerializeToString())

    print(f"{os.path.getsize(output_file)}B")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="tempHumi.csv",
                        help="Input File Name. Default: tempHumi.csv")
    parser.add_argument("--output", type=str, default="./th.tfrecord", help="Output File Name. Default: ./th.tfrecord")
    parser.add_argument("--normalize", action='store_true', help="Normalize Temperature? Insert --normalize")
    args = parser.parse_args()

    main(args)
