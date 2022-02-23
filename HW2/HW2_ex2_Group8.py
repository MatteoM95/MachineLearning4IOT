# command line: python3 HW2_ex2_Group8.py -v a
# latency calculation:  B) python3 kws_latency.py --mfcc --model ./models/ex2_b/Group8_kws_b.tflite --rate 8000 --length 240 --stride 120
#                       C) python3 kws_latency.py --mfcc --model ./models/ex2_c/Group8_kws_c.tflite --rate 8000 --length 240 --stride 120
import argparse
import numpy as np
import os
import tensorflow as tf
import zlib
from scipy import signal

RANDOM_SEED = 42
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
                 num_mel_bins=None, lower_frequency=None, upper_frequency=None,
                 num_coefficients=None, mfcc=False, resampling_rate=None):

        self.labels = labels

        self.frame_length = frame_length
        self.frame_step = frame_step

        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency

        self.num_mel_bins = num_mel_bins
        self.num_coefficients = num_coefficients

        self.resampling_rate = resampling_rate
        self.sampling_rate = sampling_rate

        num_spectrogram_bins = (frame_length) // 2 + 1
        rate = self.resampling_rate if self.resampling_rate else self.sampling_rate

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def apply_resampling(self, audio):
        audio = signal.resample_poly(audio, 1, self.sampling_rate // self.resampling_rate)
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        return audio

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        if self.resampling_rate:
            audio = tf.numpy_function(self.apply_resampling, [audio], tf.float32)

        return audio, label_id

    def pad(self, audio):
        if self.resampling_rate is not None:
            rate = self.resampling_rate
        else:
            rate = self.sampling_rate
        zero_padding = tf.zeros([rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                              frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                                       self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)
        return ds


class MyModel:
    def __init__(self, alpha, input_shape, output_shape, version):

        strides = [2, 1]

        model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * alpha),
                                                            kernel_size=[3, 3], strides=strides, use_bias=False,
                                                            name='first_conv1d'),
                                     tf.keras.layers.BatchNormalization(momentum=0.1),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                     use_bias=False),
                                     tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * alpha),
                                                            kernel_size=[1, 1], strides=[1, 1], use_bias=False,
                                                            name='second_conv1d'),
                                     tf.keras.layers.BatchNormalization(momentum=0.1),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1],
                                                                     use_bias=False, ),
                                     tf.keras.layers.Conv2D(input_shape=input_shape, filters=int(256 * alpha),
                                                            kernel_size=[1, 1], strides=[1, 1], use_bias=False,
                                                            name='third_conv1d'),
                                     tf.keras.layers.BatchNormalization(momentum=0.1),
                                     tf.keras.layers.ReLU(),
                                     tf.keras.layers.GlobalAvgPool2D(),
                                     tf.keras.layers.Dense(output_shape, name='fc')])

        self.model = model
        self.alpha = alpha
        self.version = version.lower()

        self.input_shape = input_shape  # need to append batch size

    def compile_model(self, train_ds, optimizer, loss_function, eval_metric):

        input_shape = [32] + self.input_shape
        self.model.build(input_shape)
        print(f"Input shape: {input_shape}")
        self.model.summary()  # model info

        self.model.compile( optimizer=optimizer, loss=loss_function, metrics=eval_metric )

    def train_model(self, train_dataset, val_dataset, epochs):

        print('\tTraining... ', '\t', end='')
        self.model.fit( train_dataset, epochs=epochs, validation_data=val_dataset, verbose=1 )

        return

    def prune_model(self, tflite_model_path, compressed=False):

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # PTQ and convert to tflite model
        converter_optimisations = [tf.lite.Optimize.DEFAULT] # standard (8-bit) weights-only
        converter.optimizations = converter_optimisations
        tflite_model = converter.convert()

        if not os.path.exists(os.path.dirname(tflite_model_path)):
            os.makedirs(os.path.dirname(tflite_model_path))

        # save tflite model
        with open(tflite_model_path, 'wb') as fp:
            fp.write(tflite_model)

        # compress the tflite model and save it
        if compressed:
            print("Compressed: ...")
            compressed_tflite_model_path = tflite_model_path + ".zlib"
            with open(compressed_tflite_model_path, 'wb') as fp:
                compressed_tflite_model = zlib.compress(tflite_model, level=9)
                fp.write(compressed_tflite_model)
            return os.path.getsize(compressed_tflite_model_path) / 1024

        return os.path.getsize(tflite_model_path) / 1024

    # test accuracy of the model
    def evaluate_tflite(self, tflite_model_path, test_dataset):
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        test_dataset = test_dataset.unbatch().batch(1)

        accuracy, count = 0, 0

        for features, labels in test_dataset:
            # give the input
            interpreter.set_tensor(input_details[0]['index'], features)
            interpreter.invoke()

            # predict and get the current ground truth
            prediction_logits = interpreter.get_tensor(output_details[0]['index']).squeeze()
            curr_label = labels.numpy().squeeze()

            curr_prediction = np.argmax(prediction_logits)

            if curr_prediction == curr_label:
                accuracy += 1
            count += 1

        return accuracy / float(count)

    #save model Tflite
    def save_model(self, model_path):
        self.model.save(model_path)


def main(args):

    version = args.version.lower()
    print("Training version: ", version)
    dir_path = 'data'

    tflite_model_name = f"Group8_kws_{version}.tflite"
    tflite_model_path = os.path.join("models", "ex2_" + version, tflite_model_name)

    if os.path.exists(os.path.dirname(tflite_model_path)) is False:
        os.makedirs(os.path.dirname(tflite_model_path))

    # version A
    if version == 'a':
        sampling_rate = 16000
        resampling = None
        signal_parameters = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
                             'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
                             'num_coefficients': 10}
        epochs = 25
        # learning_rate = 0.01
        alpha = 0.85

        input_shape = [49, 10, 1]
        output_shape = 8

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9
        )

    #version B and C
    elif version == 'b' or version == 'c':
        sampling_rate = 16000
        resampling = 8000
        signal_parameters = {'frame_length': 240, 'frame_step': 120, 'mfcc': True,
                             'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
                             'num_coefficients': 10}
        epochs = 25
        alpha = 0.25

        input_shape = [65, 10, 1]
        output_shape = 8

        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2,
            decay_steps=10000,
            decay_rate=0.9
        )

    train_dataset, val_dataset, test_dataset = make_tf_datasets(dir_path, sampling_rate=sampling_rate,
                                                                resampling_rate=resampling, **signal_parameters)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    eval_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    model = MyModel(alpha, input_shape, output_shape, version)
    model.compile_model(train_dataset, optimizer, loss_function, eval_metric)
    model.train_model(train_dataset, val_dataset, epochs)

    tflite_model_size = model.prune_model(tflite_model_path, compressed=True)
    print(f"tflite size: {round(tflite_model_size, 3)} KB", )

    tflite_performance = model.evaluate_tflite(tflite_model_path, test_dataset)
    print("tflite performance: ", tflite_performance)


def make_tf_datasets(dir_path, sampling_rate=16000, resampling_rate=None, **signal_parameters):
    dataset_path = os.path.join(dir_path, "mini_speech_commands")
    print(dataset_path)

    mini_speech_command_zip = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.',
        cache_subdir='data'
    )

    # it will download a folder and its subfolders
    # data > mini_speech_commands > [down/, go/, left/, no/, right/, stop/, us/, yes/]
    csv_path, _ = os.path.splitext(mini_speech_command_zip)

    data_dir = os.path.join('.', 'data', 'mini_speech_commands')

    with open("./kws_train_split.txt", "r") as fp:
        train_files = [line.rstrip() for line in fp.readlines()]  # len 6400
    with open("./kws_val_split.txt", "r") as fp:
        val_files = [line.rstrip() for line in fp.readlines()]  # len 800
    with open("./kws_test_split.txt", "r") as fp:
        test_files = [line.rstrip() for line in fp.readlines()]  # len 800

    train_files = tf.convert_to_tensor(train_files)
    val_files = tf.convert_to_tensor(val_files)
    test_files = tf.convert_to_tensor(test_files)

    labels = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']

    generator = SignalGenerator(labels, sampling_rate=sampling_rate, resampling_rate=resampling_rate,
                                **signal_parameters)
    train_dataset = generator.make_dataset(train_files, True)
    val_dataset = generator.make_dataset(val_files, False)
    test_dataset = generator.make_dataset(test_files, False)

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=str,
                        choices=['a', 'b', 'c'], required=True,
                        help='Model version to build: a - b -c')
    args = parser.parse_args()
    main(args)
