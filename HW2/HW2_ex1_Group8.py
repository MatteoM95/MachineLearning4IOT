# command line: python3 HW2_ex1_Group8.py -v a
import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import zlib

# Make sure we don't get any GPU errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
        inputs = features[:, :-self.label_width, :]
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

    def make_dataset(self, data, reshuffle):
        # Creates a dataset of sliding windows over a timeseries provided as array
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,  # consecutive data points
            targets=None,  # None -> the dataset will only yield the input data
            sequence_length=self.input_width + self.label_width,  # Length of the output sequences
            sequence_stride=1,  # Period between successive output sequences
            batch_size=self.batch_size)  # Number of timeseries samples in each batch

        # from each set of sequences it splits data to get input and labels and then normalize
        ds = ds.map(self.preprocess)

        # so the mapping is done only once
        ds = ds.cache()
        if reshuffle:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


class MultiOutputMAE(tf.keras.metrics.Metric):

    def __init__(self, name='multi_output_mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=(2,))
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0, 1])
        self.total.assign_add(error)
        self.count.assign_add(1.)
        return

    def reset_state(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)
        return result


class MyModel:
    def __init__(self, model_name, alpha, version, batch_size=32, final_sparsity=None, label_width=3, num_features=2):

        if model_name.lower() == 'model_a':
            input_shape = [6, 2]
            model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=input_shape),
                tf.keras.layers.Dense(units=int(128 * alpha), activation='relu'),
                # tf.keras.layers.Dense(units=int(128*alpha), activation='relu'),
                tf.keras.layers.Dense(units=label_width * num_features),
                tf.keras.layers.Reshape([label_width, num_features])
            ])

            # model = keras.Sequential([
            #     keras.layers.Conv1D(input_shape=input_shape, filters=int(64 * alpha), kernel_size=3),
            #     keras.layers.ReLU(),
            #     keras.layers.Flatten(),
            #     # keras.layers.Dense(units=int(64 * alpha)),
            #     # keras.layers.ReLU(),
            #     keras.layers.Dense(units=label_width * num_features),
            #     keras.layers.Reshape([label_width, num_features])
            # ])

        elif model_name.lower() == 'model_b':
            model = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=int(128 * alpha), activation='relu'),
                tf.keras.layers.Dense(units=int(128 * alpha), activation='relu'),
                tf.keras.layers.Dense(units=label_width * num_features),
                tf.keras.layers.Reshape([label_width, num_features])
            ])

        self.model = model
        self.alpha = alpha
        self.batch_size = batch_size
        self.final_sparsity = final_sparsity
        self.model_name = model_name.lower()
        self.version = version.lower()
        if alpha != 1:
            self.model_name += '_ws' + str(alpha).split('.')[1]
        if final_sparsity is not None and 'lstm' not in self.model_name:
            self.model_name += '_mb' + str(final_sparsity).split('.')[1]
            self.magnitude_pruning = True
        else:
            self.magnitude_pruning = False

        self.final_sparsity = final_sparsity

    def compile_model(self, optimizer, loss_function, eval_metric, train_ds):

        if self.magnitude_pruning:
            # sparsity scheduler
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.30,
                    final_sparsity=self.final_sparsity,
                    begin_step=len(train_ds) * 5,
                    end_step=len(train_ds) * 15)
            }

            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            self.model = prune_low_magnitude(self.model, **pruning_params)

            input_shape = [self.batch_size, 6, 2]
            self.model.build(input_shape)
            self.model.summary()

        self.model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=eval_metric
        )

    def train_model(self, train_dataset, val_dataset, epochs, callbacks=[]):

        if self.magnitude_pruning:
            callbacks.append(tfmot.sparsity.keras.UpdatePruningStep())

        print('\tTraining... ')
        print('\t', end='')

        self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            verbose=1,
            callbacks=callbacks,
        )

    def evaluate_model(self, test_dataset, return_dict=False):
        return self.model.evaluate(test_dataset, return_dict=return_dict)

    def prune_model(self, tflite_model_path, compressed=False, weights_only=True):

        self.model = tfmot.sparsity.keras.strip_pruning(self.model)
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # convert to tflite and save it
        converter_optimisations = [tf.lite.Optimize.DEFAULT,  # Enables quantization at conversion.
                                   tf.lite.Optimize.OPTIMIZE_FOR_SIZE  # Enables size reduction optimization.
                                   ]
        if weights_only:
            converter.optimizations = converter_optimisations
            converter.target_spec.supported_types = [tf.float16]  # post training quantization to float16 on the weights
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
        # with open(f'./Group10_th_{self.version}.tflite.zlib', 'wb') as fp:
        #     tflite_compressed = zlib.compress(tflite_model)
        #     fp.write(tflite_compressed)

    # test error MAE on temperature and humidity
    def test_tflite(self, tflite_model_path, test_dataset):
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
            mae = np.mean(np.abs(prediction - labels), axis=1)
            error = error + mae
            count += 1

        error_temp = error[0, 0] / float(count + 1)
        error_hum = error[0, 1] / float(count + 1)

        return error_temp, error_hum

    def save_model(self, model_path):
        self.model.save(model_path)


def main(args):
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # data and windows generator parameters
    version = args.version
    dir_path = '../datasets/'

    if version == 'a':
        model_name = 'model_a'

        input_width = 6
        label_width = 3
        num_features = 2

        epochs = 100  # 100
        alpha = 0.2 # 0.2
        learning_rate = 0.1 # 0.1
        batch_size = 512 # 512
        pruning_final_sparsity = 0.93 # 0.93

        MILESTONE = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        def scheduler(epoch, lr):
            if epoch in MILESTONE:
                return lr * 0.5
            else:
                return lr

    else:
        model_name = 'model_b'

        input_width = 6  # attenzione deve essere piú di 9 se addestro con version b
        label_width = 9
        num_features = 2

        epochs = 50  # 20
        alpha = 0.1
        learning_rate = 0.01
        batch_size = 512
        pruning_final_sparsity = 0.9

        def scheduler(epoch, lr):
            return lr

    # folder creation and saving dataset
    model_path = os.path.join("models", "ex1_" + version)
    model_tflite_name = f"Group8_th_{args.version}.tflite"
    tflite_model_path = os.path.join("models", "ex1_" + version, model_tflite_name)

    if os.path.exists(os.path.dirname(tflite_model_path)) is False:
        os.makedirs(os.path.dirname(tflite_model_path))

    # train, test, val dataset
    train_dataset, val_dataset, test_dataset, _ = make_tf_datasets(dir_path, batch_size, input_width, label_width,
                                                                 num_features)

    eval_metric = [MultiOutputMAE()]
    loss_function = [tf.keras.losses.MeanSquaredError()]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model = MyModel(model_name, alpha, version, batch_size, pruning_final_sparsity, label_width, num_features)
    model.compile_model(optimizer, loss_function, eval_metric, train_dataset)
    model.train_model(train_dataset, val_dataset, epochs,
                      callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])
    model.save_model(model_path)

    loss, error = model.evaluate_model(test_dataset, return_dict=True)
    print("Model loss: ", loss[0], " - Error: ", error[0])

    # magnitude based pruning
    tflite_size = model.prune_model(tflite_model_path, compressed=True, weights_only=True)
    print(f"tflite size: {round(tflite_size, 3)} KB", )

    # test tflite model
    error_temp, error_hum = model.test_tflite(tflite_model_path, test_dataset)
    print('T MAE: ', error_temp)
    print('Rh MAE:', error_hum)


def make_tf_datasets(dir_path,
                   batch_size=32,
                   input_width=6,
                   label_width=3,  # 3 or 9
                   num_features=2):

    csv_path = os.path.join(dir_path, "jena_climate_2009_2016.csv")

    if not os.path.exists(csv_path):
        tf.keras.utils.get_file(
            origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
            fname='jena_climate_2009_2016.csv.zip',
            extract=True,
            cache_dir='.',
            cache_subdir='../datasets/')

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=str,
                        choices=['a', 'b'], required=True,
                        help='Model version to build: a or b')
    args = parser.parse_args()
    main(args)

