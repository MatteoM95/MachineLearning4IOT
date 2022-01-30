import argparse

import tensorflow as tf
import numpy as np
import os
import base64
import json
import datetime
import requests
from scipy.signal import resample_poly
import cherrypy


class SignalGenerator:
    def __init__(self, sampling_rate, frame_length, frame_step,
                 num_mel_bins=None, lower_frequency=None, upper_frequency=None,
                 num_coefficients=None, resampling=False):
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        self.resampling = resampling
        num_spectrogram_bins = frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
            self.lower_frequency, self.upper_frequency)

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


class SlowService:
    exposed = True

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):
        model_tflite_path = "./kws_dscnn_True.tflite"

        body = cherrypy.request.body.read()
        body = json.loads(body)

        encoded_audio = body['e'][0]['v']
        audio_bytes = base64.b64decode(encoded_audio.encode('utf-8'))
        audio = np.frombuffer(audio_bytes, dtype=np.float32)

        sampling_rate = 16000
        lower_frequency = 20
        upper_frequency = 4000
        frame_length = 640  # 16 * 40
        frame_step = 320  # 16 * 20
        num_mel_bins = 40
        num_coefficients = 10

        sg = SignalGenerator(sampling_rate=sampling_rate, frame_length=frame_length,
                             frame_step=frame_step,
                             num_mel_bins=num_mel_bins, lower_frequency=lower_frequency,
                             upper_frequency=upper_frequency,
                             num_coefficients=num_coefficients, resampling=False)

        interpreter = tf.lite.Interpreter(model_tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        mfcc = sg.preprocess_with_mfcc(audio)
        mfcc = tf.expand_dims(mfcc, axis=0)

        interpreter.set_tensor(input_details[0]['index'], mfcc)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        label_pred = np.argmax(prediction).tolist()

        response = {
            "bn": "slow_service",
            "bt": int(datetime.datetime.now().timestamp()),
            "e": [
                {"n": "a", "u": "/", "t": 0, "v": label_pred}
            ]
        }

        return json.dumps(response)

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', type=str, required=False, default="127.0.0.1", help='IP of slow_service')
    parser.add_argument('--port', type=int, required=False, default="8080", help='Port of slow_service')
    args = parser.parse_args()

    conf = {
                '/': {
                    'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
                    'tools.sessions.on': True,
                }
            }
    cherrypy.tree.mount(SlowService(), '/slow_model', conf)
    cherrypy.config.update({'server.socket_host': f'{args.ip}'})
    cherrypy.config.update({'server.socket_port': args.port})
    cherrypy.engine.start()
    cherrypy.engine.block()
