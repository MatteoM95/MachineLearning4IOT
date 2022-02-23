import argparse
import os
import tensorflow as tf
import time


parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='input name',
        required=True)
parser.add_argument('--length', type=float, help='frame length in s',
        required=True)
parser.add_argument('--stride', type=float, help='stride in s', required=True)
args = parser.parse_args()


audio = tf.io.read_file(args.filename)
tf_audio, rate = tf.audio.decode_wav(audio)
tf_audio = tf.squeeze(tf_audio, 1)

frame_length = int(args.length * rate.numpy())
frame_step = int(args.stride * rate.numpy())
print('Frame length:', frame_length)
print('Frame step:', frame_step)

start = time.time()
stft = tf.signal.stft(tf_audio, frame_length, frame_step,
        fft_length=frame_length)
end = time.time()
print('Execution Time: {:.4f}s'.format(end-start))
spectrogram = tf.abs(stft)
print('Spectrogram shape:', spectrogram.shape)

spectrogram_byte = tf.io.serialize_tensor(spectrogram)
filename_byte = '{}.tf'.format(os.path.splitext(args.filename)[0])
tf.io.write_file(filename_byte, spectrogram_byte)
input_size = os.path.getsize(args.filename) / 2.**10
spectrogram_size = os.path.getsize(filename_byte) / 2.**10
print('Input Size: {:.2f}KB'.format(input_size))
print('Spectrogram Size: {:.2f}KB'.format(spectrogram_size))

image = tf.transpose(spectrogram)
image = tf.expand_dims(image, -1)
image = tf.math.log(image + 1.e-6)
min_ = tf.reduce_min(image)
max_ = tf.reduce_max(image)
image = (image - min_) / (max_ - min_)
image = image * 255.
image = tf.cast(image, tf.uint8)
image = tf.io.encode_png(image)
filename_image = '{}_stft.png'.format(os.path.splitext(args.filename)[0])
tf.io.write_file(filename_image, image)
