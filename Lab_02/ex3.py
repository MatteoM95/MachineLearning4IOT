import tensorflow as tf

import time

from scipy.io import wavfile
from scipy import signal

import numpy as np

from argparse import ArgumentParser

import os


def get_mfcc(args):

	# load the spectrogram
	print('--- Loading the spectrogram ---')
	byte_spectrogram = tf.io.read_file(args.base_path + args.filename)
	# transform the serialized tensor into a tensor
	print('--- Transforming the serialized tensor into a tensor ---')
	spectrogram = tf.io.parse_tensor(byte_spectrogram, out_type=tf.float32)

	# compute the log scaled Mel spectrogram with 40mel bins, 20Hz as lower freq and 4kHz as upper freq
	print('--- Extracting the log scaled Mel spectrogram from the stft ---')
	time_mfcc = time.time()

	num_spectrogram_bins = spectrogram.shape[-1]
	linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
						args.num_mel_bins,
						num_spectrogram_bins,
						args.sampling_rate,
						args.lower_frequency,
						args.upper_frequency
					)
	mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix,1)
	mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

	log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
	print(f'--- Log scaled Mel spectrogram extracted in {time.time()-time_mfcc} seconds ---')

	# compute the MFCCs from the log scaled mel spectrogram and take the first 10 coefficients
	print('--- Computing the MFCCs ---')
	time_mfcc = time.time()
	mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[...,:10]
	print(f'--- Computed the MFCCs in {time.time() - time_mfcc} seconds --- ')

	# convert the MFCCs in byte tensor and write it on disk
	mfccs_byte_tensor = tf.io.serialize_tensor(spectrogram)
	print('--- Convert it in bytes ---')

	# save it on disk
	mfccs_filename = args.base_path + "mfccs_res_" + args.filename
	tf.io.write_file(filename=mfccs_filename, contents=mfccs_byte_tensor)
	print('--- Saved the MFCCs on disk ---')

	# get the sizes
	print(f'--- The size of the MFCCs is {os.path.getsize(mfccs_filename)} ---')
	print(f'--- The size of the spectrogra is {os.path.getsize(args.base_path + args.filename)} ---')

def main():

	parser = ArgumentParser()

	parser.add_argument('--base_path', type=str, default='audio/')
	parser.add_argument('--filename', type=str, default='sftf_res_yes_01.wav')

	parser.add_argument('--num_mel_bins', type=int, default=40)
	parser.add_argument('--sampling_rate', type=int, default=16000)
	parser.add_argument('--lower_frequency', type=int, default=20)
	parser.add_argument('--upper_frequency', type=int, default=4000)

	args = parser.parse_args()

	get_mfcc(args)

if __name__ == '__main__':

	main()
