# command line: python3 HW2_ex1_Group8.py -v a
import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import zlib


# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def main(args):
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # data and windows generator parameters
    version = args.version
    batch_size = 32
    input_width = 6
    if version == 'a':
        label_width = 3
    else:
        label_width = 9
    num_features = 2
    data_path = 'data'

    # folder creation and saving dataset
    model_path = os.path.join("models", "ex1_" + version)
    model_tflite_name = f"Group8_th_{args.version}.tflite"
    tflite_model_path = os.path.join("models", "ex1_" + version, model_tflite_name)

    if os.path.exists(os.path.dirname(tflite_model_path)) is False:
        os.makedirs(os.path.dirname(tflite_model_path))

    # train, test, val dataset
    train_dataset, val_dataset, test_dataset, _ = fetch_datasets(data_path, batch_size, input_width, label_width,
                                                                 num_features)

    # options & params
    model_type = 'mlp'  # cnn or mlp
    width_scaling = 0.1
    pruning_final_sparsity = 0.85
    epochs = 2  # 20

    model = build_model(model_type, num_features=num_features, label_width=label_width, width_scaling=width_scaling)

    # pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,
    #                                                                            final_sparsity=pruning_final_sparsity,
    #                                                                            begin_step=len(train_dataset) * 5,
    #                                                                            end_step=len(train_dataset) * 15)}
    # prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    # model = prune_low_magnitude(model, **pruning_params)
    # callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

    # # input_shape = [batch_size, 6, 2]
    # # model.build(input_shape)

    model.compile(optimizer='adam', loss='mse', metrics=[MultiOutputMAE()])

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset) #, callbacks=callbacks)

    # print(model.summary())

    loss, error = model.evaluate(test_dataset, return_dict=True)
    # model = tfmot.sparsity.keras.strip_pruning(model) -------------------------
    print("Model loss: ", loss, " - Error: ", error)

    model.save(model_path)

    # convert to tflite and save it
    converter_optimisations = [tf.lite.Optimize.DEFAULT]
    tflite_size = to_tflite(model_path, tflite_model_path,
                            converter_optimisations=converter_optimisations,
                            compressed=False)
    print(f"tflite size: {round(tflite_size, 3)} KB", )

    # test tflite model
    error_temp, error_hum = test_tflite(tflite_model_path, test_dataset)

    print('T MAE: ', error_temp)
    print('Rh MAE:', error_hum)


def fetch_datasets(data_path,
                   batch_size=32,
                   input_width=6,
                   label_width=3,  # 3 or 9
                   num_features=2):
    csv_path = os.path.join(data_path, "jena_climate_2009_2016.csv")

    if not os.path.exists(csv_path):
        tf.keras.utils.get_file(
            origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
            fname='jena_climate_2009_2016.csv.zip',
            extract=True,
            cache_dir='.', cache_subdir='data')

    jena_dataframe = pd.read_csv(csv_path)

    column_indices = [2, 5]  # temperature and humidity data columns
    columns = jena_dataframe.columns[column_indices]
    data = jena_dataframe[columns].values.astype(np.float32)

    # splitting dataframe in train/validation/test
    n = len(data)
    train_data = data[0:int(n * 0.7)]
    val_data = data[int(n * 0.7):int(n * 0.9)]
    test_data = data[int(n * 0.9):]

    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    # build datasets
    generator = WindowGenerator(batch_size, input_width, label_width, num_features, mean=mean, std=std)
    train_dataset = generator.make_dataset(train_data, True)
    val_dataset = generator.make_dataset(val_data, False)
    test_dataset = generator.make_dataset(test_data, False)

    return train_dataset, val_dataset, test_dataset, n

# build models network
def build_model(model='mlp', *args, **kwargs):
    if model == 'mlp':
        return make_mlp(*args, **kwargs)
    else:
        return make_cnn(*args, **kwargs)


def make_mlp(input_shape=(6, 2), units=128, num_features=2, label_width=3, width_scaling=1):
    model = keras.Sequential([keras.layers.Flatten(),  # input_shape=input_shape
                              keras.layers.Dense(units=units * width_scaling, activation='relu'),
                              keras.layers.Dense(units=units * width_scaling, activation='relu'),
                              keras.layers.Dense(units=num_features * label_width),
                              keras.layers.Reshape([label_width, num_features])])

    return model


def make_cnn(input_shape=(6, 2), filters=64, units=64, output_units=12, width_scaling=1):
    model = keras.Sequential(
        [keras.layers.Conv1D(filters=filters * width_scaling, kernel_size=3, input_shape=input_shape),
         keras.layers.ReLU(),
         keras.layers.Flatten(),
         keras.layers.Dense(units=units * width_scaling, activation='relu'),
         keras.layers.Dense(units=output_units),
         keras.layers.Reshape([6, 2])])

    return model

# vedere qui per il windows generator, spiega il " timeseries_dataset_from_array "
# https://mobiarch.wordpress.com/2020/11/13/preparing-time-series-data-for-rnn-in-tensorflow/
class WindowGenerator:
    def __init__(self, batch_size, input_width, label_width, num_features, mean, std):
        self.batch_size = batch_size
        self.input_width = input_width
        self.label_width = label_width
        self.num_features = num_features
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        # features -> set of sequences made of input_width + label_width values each. [#batch, (input+label)_width, 2]

        inputs = features[:, : -self.input_width, :]
        labels = features[:, -self.label_width:, :]

        inputs.set_shape([None, self.input_width, self.num_features])
        labels.set_shape([None, self.label_width, self.num_features])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train=True):
        # Creates a dataset of sliding windows over a timeseries provided as array
        dataset = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,  # consecutive data points
            targets=None,  # None -> the dataset will only yield the input data
            sequence_length=self.input_width + self.label_width,  # Length of the output sequences
            sequence_stride=1,  # Period between successive output sequences
            batch_size=self.batch_size)  # Number of timeseries samples in each batch

        # from each set of sequences it splits data to get input and labels and then normalize
        dataset = dataset.map(self.preprocess)

        dataset = dataset.cache()
        if train is True:
            dataset = dataset.shuffle(100, reshuffle_each_iteration=True)

        return dataset


class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='multi_output_mean_absolute_error', **kwargs):
        super().__init__(name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=[2])
        self.count = self.add_weight('count', initializer='zeros')

    def reset_state(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0, 1])
        self.total.assign_add(error)
        self.count.assign_add(1.)
        return

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        return result


def to_tflite(source_model_path, tflite_model_path, converter_optimisations=None, compressed=False):
    converter = tf.lite.TFLiteConverter.from_saved_model(source_model_path)

    if converter_optimisations is not None:
        converter.optimizations = converter_optimisations
    tflite_model = converter.convert()

    if not os.path.exists(os.path.dirname(tflite_model_path)):
        os.makedirs(os.path.dirname(tflite_model_path))

    with open(tflite_model_path, 'wb') as fp:
        fp.write(tflite_model)

    if compressed:
        compressed_tflite_model_path = tflite_model_path + ".zlib"
        with open(compressed_tflite_model_path, 'wb') as fp:
            compressed_tflite_model = zlib.compress(tflite_model, level=9)
            fp.write(compressed_tflite_model)
        return os.path.getsize(compressed_tflite_model_path) / 1024

    return os.path.getsize(tflite_model_path) / 1024


def test_tflite(tflite_model_path, test_dataset):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    error, count = 0, 0
    test_dataset = test_dataset.unbatch().batch(1)
    for features, labels in test_dataset:
        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        mae = np.mean( np.abs(prediction - labels), axis=1)
        error = error + mae
        count += 1

    error_temp = error[0, 0] / float(count + 1)
    error_hum = error[0, 1] / float(count + 1)

    return error_temp, error_hum


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=str, choices=['a', 'b'], required=True,
                        help='Model version to build: a or b')
    args = parser.parse_args()
    main(args)
