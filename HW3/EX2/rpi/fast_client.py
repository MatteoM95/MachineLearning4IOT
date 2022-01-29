# import subprocess
#
# performance = ['sudo', 'sh', '-c', 'echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
# powersave = ['sudo', 'sh', '-c', 'echo powersave > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor']
#
# subprocess.check_call(performance)

import tensorflow as tf
import numpy as np
import os
import base64
import json
import datetime
import requests
from scipy.signal import resample_poly


class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
                 num_mel_bins=None, lower_frequency=None, upper_frequency=None,
                 num_coefficients=None, resampling=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        self.resampling = resampling
        num_spectrogram_bins = (frame_length) // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
            self.lower_frequency, self.upper_frequency)
        self.preprocess = self.preprocess_with_mfcc

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
        mfccs = mfccs[:, :self.num_coefficients]

        return mfccs

    def preprocess_with_mfcc(self, audio):
        if self.resampling:
            audio = resample_poly(audio, self.sampling_rate, 16000)

        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        # print(tf.shape(spectrogram))
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)
        # print("MFCC ------", mfccs.shape)
        return mfccs


def success_checker(pred):
    y_pred = tf.nn.softmax(pred)

    list_sm = tf.sort(y_pred, direction='DESCENDING')[:2]
    sm = list_sm[0] - list_sm[1]

    return sm


def make_tf_datasets(dir_path, sampling_rate=16000):
    dataset_path = os.path.join(dir_path, "mini_speech_commands")

    if not os.path.exists(dataset_path):
        tf.keras.utils.get_file(
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            fname='mini_speech_commands.zip',
            extract=True,
            cache_dir='.', cache_subdir='data')

    test_files = open('./kws_test_split.txt', 'r').read().splitlines()
    test_files = tf.convert_to_tensor(test_files)

    labels = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']

    sampling_rate = 16000
    lower_frequency = 20
    upper_frequency = 4000
    frame_length = 480  # 40
    frame_step = 320  # 20
    num_mel_bins = 40
    num_coefficients = 10

    sg = SignalGenerator(labels=labels, sampling_rate=sampling_rate, frame_length=frame_length, frame_step=frame_step,
                         num_mel_bins=num_mel_bins, lower_frequency=lower_frequency, upper_frequency=upper_frequency,
                         num_coefficients=num_coefficients, resampling=False)

    interpreter = tf.lite.Interpreter("./kws_dscnn_True.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    communication_cost = 0.0
    threshold = 0.15
    accuracy = 0
    total_test_size = len(test_files)
    for it, file_path in enumerate(test_files):
        print(f'Progress: {i + 1}/{total_test_size}', end='\r')
        audio, label_true = sg.read(file_path)
        mfcc = sg.preprocess_with_mfcc(audio)

        mfcc = tf.expand_dims(mfcc, axis=0)

        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        softmax_difference = success_checker(prediction)

        if softmax_difference < threshold:
            print(f"Slow service calling.... [{softmax_difference}]")
            ENCODING = 'utf-8'
            audio_bytes = base64.b64encode(audio)
            audio_string = audio_bytes.decode(ENCODING)

            request = {
                "bn": "fast_service@127.0.0.1",
                "bt": int(datetime.datetime.now().timestamp()),
                "e": [
                    {"n": "a", "u": "/", "t": 0, "v": audio_string}
                ]
            }
            request = json.dumps(request)
            if (communication_cost + len(request)) / (1024 * 1024) < 4.5:
                communication_cost += len(request)
                r = requests.post('http://127.0.0.1:8080/slow_model', request)
                if r.status_code == 200:
                    label_pred = r.json()['e'][0]['v']
                else:
                    print('Error with the slow model prediction')
            else:
                label_pred = tf.argmax(prediction)
        else:
            label_pred = tf.argmax(prediction)

        if label_pred == label_true:
            accuracy += 1

    accuracy = round(accuracy / len(test_files) * 100, 3)
    print(f'Collaborative accuracy: {accuracy}%')
    print(f'Communication Cost: {communication_cost / (1024 * 1024)} MB')


if __name__ == '__main__':
    make_tf_datasets("./data")












