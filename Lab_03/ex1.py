from time import time
import sys
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class WindowGenerator:
    def __init__(self, input_width, label_options, mean, std, verbose=False):
        self.input_width = input_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])
        self.verbose = verbose

    def split_window(self, features):
        # input_indeces = np.arange(self.input_width)
        inputs = features[:, :-1, :]

        if self.label_options < 2:
            labels = features[:, -1, self.label_options]
            labels = tf.expand_dims(labels, -1)
            num_labels = 1
        else:
            labels = features[:, -1, :]
            num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, num_labels])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        # It returns a tf.data.Dataset instance
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.input_width+1,
                sequence_stride=1,
                batch_size=32)
        # Maps map_func across the elements of this dataset.
        dataset = dataset.map(map_func=self.preprocess)
        if self.verbose:
            print(f"Dataset element: {dataset.element_spec}")
        # Caches the elements in this dataset.
        # The first time the dataset is iterated over, its elements will be cached either 
        # in the specified file or in memory. Subsequent iterations will use the cached data.
        dataset = dataset.cache() 
        if train is True:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)

        return dataset


class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name="mean_absolute_error", **kwargs):
        super().__init__(name, **kwargs)
        self.total = self.add_weight("total", initializer='zeros', shape=[2])
        self.count = self.add_weight("count", initializer="zeros")

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))

        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=0)
        self.total.assign_add(error)
        self.count.assign_add(1.)

        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)

        return result


def get_data(args):

    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True,
        cache_dir='.', 
        cache_subdir='data'
    )

    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    
    column_indices = [2, 5]
    columns = df.columns[column_indices]
    
    data = df[columns].values.astype(np.float32)
    n = len(data)
    
    train_data = data[0:int(n*0.7)]
    val_data = data[int(n*0.7):int(n*0.9)]
    test_data = data[int(n*0.9):]

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    generator = WindowGenerator(args.window_length, args.labels, mean, std)
    train_ds = generator.make_dataset(train_data, True)
    val_ds = generator.make_dataset(val_data, False)
    test_ds = generator.make_dataset(test_data, False)

    return train_ds, val_ds, test_ds

def initialize_model(model_name, label_options):
    
    # label_options can assume only values 0,1,2
    units = max(1,label_options)

    if model_name == "MLP":
        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=units)
        ])
        
    elif model_name == "CNN":
        model = keras.Sequential([
            keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(units=64, activation='relu'),
            keras.layers.Dense(units=units)
        ])
        
    elif model_name == "LSTM":
        model = keras.Sequential([
            keras.layers.LSTM(units=64),
            keras.layers.Flatten(),
            keras.layers.Dense(units=units)
        ])
        
        

    if label_options <= 1:
      metrics = [tf.keras.metrics.MeanAbsoluteError()]
    else: 
      metrics = [MultiOutputMAE()]
      
      
    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.MeanSquaredError(), 
                  metrics=[metrics])

    return model


def main(args):
    
    # generate the window and define the datasets
    train_ds, val_ds, test_ds = get_data(args)

    # get the requested model ['LSTM','CNN','MLP']
    model = initialize_model(args.model, args.labels)

    # fit the model
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)         

    # path for the saved file
    path_to_dir = 'models/' + args.model + "_" + str(args.labels)

    if os.path.exists(path_to_dir) is False:
        os.makedirs(path_to_dir)

    model.save(path_to_dir) 

    loss, mae = model.evaluate(test_ds)
    
    print(f"Loss = {loss} , MAE = {mae}")

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-m','--model', type=str, required=True, help='model name', choices=['MLP','LSTM','CNN'])
    parser.add_argument('-l','--labels', type=int, help='model output', default=1, choices=[0,1,2])
    parser.add_argument('-e','--epochs', type=int, help='training epochs', default=1)
    parser.add_argument('-w','--window_length', type=int, help='length of the sliding window', default=6)
    
    args = parser.parse_args()

    main(args)

  