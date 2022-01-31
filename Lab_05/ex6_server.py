import cherrypy 
import json
import base64
from cherrypy.process.wspbus import ChannelFailures
import numpy as np
import tensorflow as tf
import sys



class KWS(object):
	exposed = True

	def __init__(self):
		dscnn = tf.keras.models.load_model("./Lab05/dscnn_kws/")
		self.labels = ['right', 'go', 'no', 'left', 'stop', 'up', 'down', 'yes']
		self.models = {'dscnn': dscnn}

        # audio setup (taken from the dataset)
		self.length = 640
		self.stride = 320
		self.bins = 40
		self.coeff = 10
		self.rate = 16000
		self.resize = 32

		num_spectrogram_bins = self.length // 2 + 1

		self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
						self.bins, num_spectrogram_bins, self.rate, 20, 4000)

	def preprocess(self, audio_bytes):
		# decode and normalize
		audio, _ = tf.audio.decode_wav(audio_bytes)
		audio = tf.squeeze(audio, axis=1)

		stft = tf.signal.stft(audio, frame_length=self.length,
								frame_step=self.stride, fft_length=self.length)
		spectrogram = tf.abs(stft)

		mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
		log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
		mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
		mfccs = mfccs[..., :self.coeff]

		mfccs = tf.expand_dims(mfccs, -1)
		mfccs = tf.expand_dims(mfccs, 0)

		return mfccs



	def PUT(self, *path, **query):

		input_body = cherrypy.request.body.read()
		input_body = json.loads(input_body)
		events = input_body['e']

        # I am waiting for a single audio
		audio_string = events[0]['vd']

		if audio_string is None:
			raise cherrypy.HTTPError(400, "No event")
		
		audio_bytes = base64.b64decode(audio_string)
		mfccs = self.preprocess(audio_bytes)
		model = self.models.get(path[0]) # get the selected model

		if model is None:
			raise cherrypy.HTTPError(400, "No valid model")

		logits = model.predict(mfccs)
		probabilities = tf.nn.softmax(logits)
		probability = tf.reduce_max(probabilities).numpy() * 100
		predicted_label = self.labels[tf.argmax(probabilities, 1).numpy()[0]]

		output_body = {
            'label': predicted_label,
            'probability': probability
		}

		return json.dumps(output_body)


if __name__ == '__main__':
	conf = {
		'/': {
			'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
		}
	}
	cherrypy.tree.mount(KWS(), '/', conf)
	cherrypy.engine.start()
	cherrypy.engine.block()