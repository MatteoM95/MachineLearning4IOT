import tensorflow as tf

import time

from scipy.io import wavfile
from scipy import signal

import numpy as np

from argparse import ArgumentParser

import os



def preprocess_audio(args):

	# read the audio signal
	rate, audio = wavfile.read(args.base_path + args.filename)

	print(f'--- {args.filename} has been read --- ')
	print(f'--- Start poly resampling --- ')

	time_poly = time.time()

	# apply poly_phase filtering
	audio = signal.resample_poly(audio, 1, int(rate/args.poly_rate))
	print(f'--- Time spent for the poly_filtering phase is {time.time() - time_poly} ---')

	print('--- Start STFT conversion ---')
	time_stft = time.time()
	# cast the signal to the original data type
	audio = audio.astype(np.int16)

	# save the pre processed file
	preprocess_path = args.base_path + "res_" + args.filename
	wavfile.write(preprocess_path, args.poly_rate, audio)
	print('--- Saved the preprocessed audio file on disk ---')

	# get the audio in a string format
	audio = tf.io.read_file(preprocess_path)
	tf_audio, rate = tf.audio.decode_wav(contents=audio, desired_channels=1)

	# remove dimension of size 1
	tf_audio = tf.squeeze(tf_audio,1)

	# convert the waveform in a spectroram applying the STFT
	# frame_length = An integer scalar Tensor. The window length in samples. | l=40ms (frame_length=f x l)
	frame_length = tf.constant(int(rate.numpy() * args.window_length / 1000))
	# frame_step = An integer scalar Tensor. The number of samples to step.  | s = 2 ms
	frame_step = tf.constant(int(rate.numpy() * args.step_length / 1000))
	# fft_length = An integer scalar Tensor. The size of the FFT to apply.   |
	#              If not provided, uses the smallest power of 2 enclosing frame_length.
	fft_length = tf.constant(int(rate.numpy() * args.window_length / 1000))

	stft = tf.signal.stft(tf_audio,
			      frame_length = fft_length,
			      frame_step = frame_length,
		      	      fft_length = fft_length
		     	     )

	print(f'--- Time spent for the STFT is {time_stft - time.time()} ---')
	# extract the spectrogram
	spectrogram = tf.abs(stft)
	print(f'--- Extracted the spectrogram ---')
	# convert the spectrogram in byte tensor and write it on disk
	spectrogram_byte_tensor = tf.io.serialize_tensor(spectrogram)

	# write it on disk
	stft_filename = args.base_path + "sftf_res_" + args.filename
	tf.io.write_file(filename=stft_filename, contents=spectrogram_byte_tensor)
	print('--- Saved the spectrogram on disk ---')

def main():

	parser = ArgumentParser()

	parser.add_argument('--base_path', type=str, default='audio/')
	parser.add_argument('--filename', type=str, default='yes_01.wav')
	parser.add_argument('--poly_rate', type=int, default=16000)
	parser.add_argument('--window_length', type=int, default=40)
	parser.add_argument('--step_length', type=int, default=20)

	args = parser.parse_args()

	preprocess_audio(args)


if __name__ == '__main__':

	main()
