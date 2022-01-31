"""
The Mini Speech Command dataset collects 8000 samples of eight keywords 
('down', 'no', 'go', 'yes', 'stop', 'up', 'right', 'left'), 1000 samples 
per label. Each sample is recorded at 16kHz with an Int16 resolution and 
has a variable duration (1s the longest).

Write a Python script to train and evaluate different models for keyword 
spotting on the Mini Speech Command dataset.
"""

import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

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



def load_data(args):

    if args.silence is True:
      data_dir = os.path.join('.', 'data', 'mini_speech_commands_silence')
    else:
      mini_speech_command_zip = tf.keras.utils.get_file(
                            origin = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                            fname = "mini_speech_commands.zip",
                            extract=True,
                            cache_dir='.',
                            cache_subdir='data'
                        )
      # it will download a folder and its subfolders
      # data > mini_speech_commands > [down/, go/, left/, no/, right/, stop/, us/, yes/]
      csv_path, _ = os.path.splitext(mini_speech_command_zip)

      data_dir = os.path.join('.', 'data', 'mini_speech_commands')

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)
    num_samples = len(filenames)

    if args.silence is True:
        total = 9000
    else:
        total = 8000

    train_files = filenames[:int(total*0.8)]
    val_files = filenames[int(total*0.8): int(total*0.9)]
    test_files = filenames[int(total*0.9):]


    LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))
    LABELS = LABELS[LABELS != 'README.md']

    STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}
    MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
            'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
            'num_coefficients': 10}


    if args.mfcc is True:
        options = MFCC_OPTIONS
    else:
        options = STFT_OPTIONS

    generator = SignalGenerator(LABELS, 16000, **options)
    train_ds = generator.make_dataset(train_files, True)
    val_ds = generator.make_dataset(val_files, False)
    test_ds = generator.make_dataset(test_files, False)

    return train_ds, val_ds, test_ds


def get_model(model_name, mfcc, silence):

    strides = [2,1] if mfcc else [2,2]
    units = 9 if silence else 8

    if model_name == "MLP":
        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(units=units)
        ])

    elif model_name == "CNN":

        model = keras.Sequential([
          keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=strides, use_bias=False),
          keras.layers.BatchNormalization(momentum=0.1),
          keras.layers.ReLU(),
          keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=strides, use_bias=False),
          keras.layers.BatchNormalization(momentum=0.1),
          keras.layers.ReLU(),
          keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=strides, use_bias=False),
          keras.layers.BatchNormalization(momentum=0.1),
          keras.layers.ReLU(),
          keras.layers.GlobalAveragePooling2D(),
          keras.layers.Dense(units=units)
      ])

    elif model_name == "DS-CNN":
        model = keras.Sequential([
                  keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=strides, use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                  keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
                  keras.layers.Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
                  keras.layers.BatchNormalization(momentum=0.1),
                  keras.layers.ReLU(),

                  keras.layers.GlobalAveragePooling2D(),
                  keras.layers.Dense(units=units)
              ])

    return model

def train_model(model, train_data, val_data, optimizer, loss, metrics, bs, epochs):
    
    model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)

    model.fit(train_data, validation_data = val_data, batch_size=bs, epochs=epochs)

def main(args):
    
    train_ds, val_ds, test_ds = load_data(args)

    # set the output path
    modality = 'STFT' if not args.mfcc else 'MFCCS'
    output_path = f'{args.model}_{modality}/'

    model = get_model(args.model, args.mfcc, args.silence)

    model.compile(optimizer=tf.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=keras.metrics.SparseCategoricalAccuracy())

    save_model = keras.callbacks.ModelCheckpoint(filepath=output_path, 
                                                monitor="val_sparse_categorical_accuracy",
                                                save_best_only=True,
                                                save_weights_only=False,
                                                save_freq='epoch')

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=save_model)
    test_loss, test_acc = model.evaluate(test_ds)
    
    print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default="MLP", choices=['MLP','CNN','DS-CNN'])
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-M','--mfcc', action='store_true')
    parser.add_argument("-s", '--silence', action='store_true')
    
    args = parser.parse_args()

    main(args)