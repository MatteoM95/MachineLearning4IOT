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
        num_spectrogram_bins = frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
            self.lower_frequency, self.upper_frequency)
        self.preprocess = self.preprocess_with_mfcc

    # def read(self, file_path):
    #     parts = tf.strings.split(file_path, os.path.sep)
    #     label = parts[-2]
    #     label_id = tf.argmax(label == self.labels)
    #     audio_binary = tf.io.read_file(file_path)
    #     audio, _ = tf.audio.decode_wav(audio_binary)
    #     audio = tf.squeeze(audio, axis=1)
    #
    #     return audio, label_id

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


class SlowService:
    exposed = True

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):
        # if len(path) != 1 or path[0] != "slow_model":
        #     raise cherrypy.HTTPError(400, 'Wrong path')

        body = cherrypy.request.body.read()
        body = json.loads(body)

        encoded_audio = body['e'][0]['v']

        # https://stackabuse.com/encoding-and-decoding-base64-strings-in-python/
        audio_bytes = base64.b64decode(encoded_audio.encode('utf-8'))

        audio = np.frombuffer(audio_bytes, dtype=np.float32)

        sampling_rate = 16000
        lower_frequency = 20
        upper_frequency = 4000
        frame_length = 640  # 40
        frame_step = 320  # 20
        num_mel_bins = 40
        num_coefficients = 10

        sg = SignalGenerator(labels=None, sampling_rate=sampling_rate, frame_length=frame_length,
                             frame_step=frame_step,
                             num_mel_bins=num_mel_bins, lower_frequency=lower_frequency,
                             upper_frequency=upper_frequency,
                             num_coefficients=num_coefficients, resampling=False)

        interpreter = tf.lite.Interpreter("./kws_dscnn_True.tflite")
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
            "bn": "slow_service@127.0.0.1",
            "bt": int(datetime.datetime.now().timestamp()),
            "e": [
                {"n": "a", "u": "/", "t": 0, "v": label_pred}
            ]
        }

        return json.dumps(response)

    def PUT(self, *path, **query):
        pass

    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    conf = {
                '/': {
                    'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
                    'tools.sessions.on': True,
                }
            }
    cherrypy.tree.mount(SlowService(), '/slow_model', conf)
    cherrypy.config.update({'server.socket_host': '127.0.0.1'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()




    #
    # import base64
    # import io
    # import json
    # import cherrypy
    # import datetime
    # import tensorflow as tf
    # import numpy as np
    # from scipy import signal
    # import wave
    # import os
    #
    #
    # class BigModelGenerator(object):
    #     exposed = True
    #
    #     def POST(self):
    #         body = cherrypy.request.body.read()
    #         body = json.loads(body)
    #
    #         audio = body["e"][0]["vd"]
    #         audio = base64.b64decode(audio.encode())
    #
    #         audio = np.frombuffer(audio, dtype=np.float32)
    #
    #         sampling_rate = 16000
    #         frame_length, frame_step = 640, 320
    #         lower_frequency, upper_frequency = 20, 4000
    #
    #         num_coefficients = 30
    #         num_mel_bins = 40
    #
    #         num_spectrogram_bins = (frame_length) // 2 + 1
    #         linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins,
    #                                                                             sampling_rate, lower_frequency,
    #                                                                             upper_frequency)
    #
    #         # creating mfccs
    #
    #         zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
    #         audio = tf.concat([audio, zero_padding], 0)
    #         audio.set_shape([sampling_rate])
    #
    #         stft = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    #         spectrogram = tf.abs(stft)
    #
    #         mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    #         log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    #         mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    #         mfccs = mfccs[..., :num_coefficients]
    #         mfccs = tf.expand_dims(mfccs, -1)
    #
    #         interpreter = tf.lite.Interpreter(model_path=f'./big.tflite')
    #         interpreter.allocate_tensors()
    #         input_details = interpreter.get_input_details()
    #         output_details = interpreter.get_output_details()
    #
    #         interpreter.set_tensor(input_details[0]['index'], [mfccs])
    #         interpreter.invoke()
    #         y_pred = np.argmax(interpreter.get_tensor(output_details[0]['index'])[0]).tolist()
    #
    #         response = {
    #             "bn": "big_model@127.0.0.1",
    #             "bt": int(datetime.datetime.now().timestamp()),
    #             "e": [
    #                 {"n": "label", "u": "/", "t": 0, "v": y_pred}
    #             ]
    #         }
    #         return json.dumps(response)
    #
    #
    # if __name__ == '__main__':
    #
    #     if 'data' not in os.listdir():
    #         zip_path = tf.keras.utils.get_file(
    #             origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    #             fname='mini_speech_commands.zip',
    #             extract=True,
    #             cache_dir='.', cache_subdir='data')
    #
    #     conf = {
    #         '/': {
    #             'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
    #             'tools.sessions.on': True,
    #         }
    #     }
    #     cherrypy.tree.mount(BigModelGenerator(), '/big_model', conf)
    #     cherrypy.config.update({'server.socket_host': '127.0.0.1'})
    #     cherrypy.config.update({'server.socket_port': 8080})
    #     cherrypy.engine.start()
    #     cherrypy.engine.block()