import argparse
import numpy as np
import os
import time
from scipy.io import wavfile
from scipy import signal


parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='input name',
        required=True)
parser.add_argument('--rate', type=int, help='output rate',
        required=True)
args = parser.parse_args()


input_rate, audio = wavfile.read(args.filename)
sampling_ratio = input_rate / args.rate

start = time.time()
audio = signal.resample_poly(audio, 1, sampling_ratio)
end = time.time()
print('Execution Time: {:.3f}s'.format(end-start))

audio = audio.astype(np.int16)
filename = '{}_16.wav'.format(os.path.splitext(args.filename)[0])
wavfile.write(filename, args.rate, audio)

input_size = os.path.getsize(args.filename) / 2.**10
output_size = os.path.getsize(filename) / 2.**10
print('Input Size: {:.2f}KB'.format(input_size))
print('Output Size: {:.2f}KB'.format(output_size))
