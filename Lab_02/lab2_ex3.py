import argparse
import os
import tensorflow as tf
import time


parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='input name',
        required=True)
parser.add_argument('--mel-bins', type=int, help='Mel bins',
        required=True)
parser.add_argument('--coefficients', type=int, help='MFCCs number',
        required=True)
args = parser.parse_args()


spectrogram = tf.io.read_file(args.filename)
spectrogram = tf.io.parse_tensor(spectrogram, out_type=tf.float32)
print('Spectrogram shape:', spectrogram.shape)

num_spectrogram_bins = spectrogram.shape[-1]
num_mel_bins = args.mel_bins
sampling_rate = 16000

start = time.time()
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sampling_rate, 20, 4000)
mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
mfccs = mfccs[..., :args.coefficients]
end = time.time()
print('Execution Time: {:.3f}s'.format(end-start))
print('MFCCs shape:', mfccs.shape)

mfccs_byte = tf.io.serialize_tensor(mfccs)
filename_byte = '{}_mfccs.tf'.format(os.path.splitext(args.filename)[0])
tf.io.write_file(filename_byte, mfccs_byte)
input_size = os.path.getsize(args.filename) / 2.**10
mfccs_size = os.path.getsize(filename_byte) / 2.**10
print('Input Size: {:.2f}KB'.format(input_size))
print('MFCCs Size: {:.2f}KB'.format(mfccs_size))

image = tf.transpose(mfccs)
image = tf.expand_dims(image, -1)
min_ = tf.reduce_min(image)
max_ = tf.reduce_max(image)
image = (image - min_) / (max_ - min_)
image = image * 255.
image = tf.cast(image, tf.uint8)
image = tf.io.encode_png(image)
filename_image = '{}_mfccs.png'.format(os.path.splitext(args.filename)[0])
tf.io.write_file(filename_image, image)
