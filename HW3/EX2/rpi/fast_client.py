from subprocess import Popen
import argparse
import tensorflow as tf
import os
import base64
import json
import datetime
import requests
from scipy.signal import resample_poly

Popen('sudo sh -c "echo performance >'
      '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
      shell=True).wait()


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

        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs


def success_checker(pred):
    y_pred = tf.nn.softmax(pred)

    list_sm = tf.sort(y_pred, direction='DESCENDING')[:2]
    sm = list_sm[0] - list_sm[1]

    return sm


def main(args):
    dataset_path = os.path.join(args.path, "mini_speech_commands")

    text_split_path = './kws_test_split.txt'
    model_tflite_path = "./kws_dscnn_True.tflite"
    IP = args.ip
    PORT = args.port

    if not os.path.exists(dataset_path):
        tf.keras.utils.get_file(
            origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
            fname='mini_speech_commands.zip',
            extract=True,
            cache_dir='.', cache_subdir='data')

    test_files = open(text_split_path, 'r').read().splitlines()
    test_files = tf.convert_to_tensor(test_files)

    labels = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']

    sampling_rate = 16000
    lower_frequency = 20
    upper_frequency = 4000
    frame_length = 480
    frame_step = 320
    num_mel_bins = 32
    num_coefficients = 10

    sg = SignalGenerator(labels=labels, sampling_rate=sampling_rate, frame_length=frame_length, frame_step=frame_step,
                         num_mel_bins=num_mel_bins, lower_frequency=lower_frequency, upper_frequency=upper_frequency,
                         num_coefficients=num_coefficients, resampling=False)

    interpreter = tf.lite.Interpreter(model_tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    communication_cost = 0.0
    threshold = 0.1
    accuracy = 0

    total_test_size = len(test_files)
    for it, file_path in enumerate(test_files):
        print(f'Progress: {it + 1}/{total_test_size}', end='\r')
        audio, label_true = sg.read(file_path)
        mfcc = sg.preprocess_with_mfcc(audio)

        mfcc = tf.expand_dims(mfcc, axis=0)

        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]

        softmax_difference = success_checker(prediction)

        if softmax_difference < threshold:
            audio_bytes = base64.b64encode(audio)
            audio_string = audio_bytes.decode('utf-8')

            request = {
                "bn": f"fast_service@{IP}",
                "bt": int(datetime.datetime.now().timestamp()),
                "e": [
                    {"n": "a", "u": "/", "t": 0, "v": audio_string}
                ]
            }
            request = json.dumps(request)
            communication_cost += len(request)

            response = requests.put(f'http://{IP}:{PORT}/slow_model', request)
            if response.status_code == 200:
                label_pred = response.json()['e'][0]['v']
            else:
                print('Error with the slow model prediction')
        else:
            label_pred = tf.argmax(prediction)

        if label_pred == label_true:
            accuracy += 1

    accuracy = round(accuracy / len(test_files) * 100, 3)
    print(f'Accuracy: {accuracy}%')
    print(f'Communication Cost: {communication_cost / (1024 * 1024)} MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=False, default="./data", help='Dataset path')
    parser.add_argument('--ip', type=str, required=False, default="127.0.0.1", help='IP of slow_service')
    parser.add_argument('--port', type=str, required=False, default="8080", help='Port of slow_service')
    args = parser.parse_args()

    main(args)
