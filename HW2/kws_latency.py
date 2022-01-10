import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=False,
        help='model full path')
parser.add_argument('--rate', type=int, default=16000,
        help='sampling rate after resampling')
parser.add_argument('--mfcc', action='store_true',
        help='use mfcc')
parser.add_argument('--resize', type=int, default=32,
        help='input size after resize')
parser.add_argument('--length', type=int, default=640,
        help='stft window legnth in number of samples')
parser.add_argument('--stride', type=int, default=320,
        help='stft window stride in number of samples')
parser.add_argument('--bins', type=int, default=40,
        help='number of mel bins')
parser.add_argument('--lower-frequency', type=int, default=20,
        help='lower frequency of the mel-scale filters')
parser.add_argument('--upper-frequency', type=int, default=4000,
        help='upper frequency of the mel-scale filters')
parser.add_argument('--coeff', type=int, default=10,
        help='number of MFCCs')
args = parser.parse_args()


import tensorflow as tf
import time
from scipy import signal
import numpy as np
from subprocess import call

call('sudo sh -c "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
            shell=True)

rate = args.rate
length = args.length
stride = args.stride
resize = args.resize
num_mel_bins = args.bins
num_coefficients = args.coeff

num_frames = (rate - length) // stride + 1
num_spectrogram_bins = length // 2 + 1

linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, rate, args.lower_frequency,
        args.upper_frequency)

if args.model is not None:
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


inf_latency = []
tot_latency = []
for i in range(100):
    sample = np.array(np.random.random_sample(48000), dtype=np.float32)

    start = time.time()

    # Resampling
    sample = signal.resample_poly(sample, 1, 48000 // rate)

    sample = tf.convert_to_tensor(sample, dtype=tf.float32)

    # STFT
    stft = tf.signal.stft(sample, length, stride,
            fft_length=length)
    spectrogram = tf.abs(stft)

    if args.mfcc is False and args.resize > 0:
        # Resize (optional)
        spectrogram = tf.reshape(spectrogram, [1, num_frames, num_spectrogram_bins, 1])
        spectrogram = tf.image.resize(spectrogram, [resize, resize])
        input_tensor = spectrogram
    else:
        # MFCC (optional)
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :num_coefficients]
        mfccs = tf.reshape(mfccs, [1, num_frames, num_coefficients, 1])
        input_tensor = mfccs

    if args.model is not None:
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        start_inf = time.time()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

    end = time.time()
    tot_latency.append(end - start)

    if args.model is None:
        start_inf = end

    inf_latency.append(end - start_inf)
    time.sleep(0.1)

print('Inference Latency {:.2f}ms'.format(np.mean(inf_latency)*1000.))
print('Total Latency {:.2f}ms'.format(np.mean(tot_latency)*1000.))
