import os
import zipfile
import tensorflow as tf
import time
from subprocess import Popen
import math
import numpy as np
from scipy.io import wavfile
from scipy import signal


def audio_processing(filename, stftParams, mfccParams, num_coefficients, resample=False, new_resample_rate=16000):
    frame_length = stftParams['frame_length'] * stftParams['frame_step']

    if resample:
        input_rate, audio = wavfile.read(filename)
        sampling_ratio = input_rate / new_resample_rate

        audio = signal.resample_poly(audio, 1, sampling_ratio)

        tf_audio = tf.convert_to_tensor(audio, dtype=tf.float32)

        # stftParams = {'frame_length': 1,
        #                'frame_step': 2}

        stft = tf.signal.stft(tf_audio,
                              frame_length=stftParams['frame_length'],
                              frame_step=stftParams['frame_step'],
                              fft_length=frame_length)
        spectrogram = tf.abs(stft)

    else:
        audio = tf.io.read_file(filename)
        tf_audio, _ = tf.audio.decode_wav(audio)
        tf_audio = tf.squeeze(tf_audio, 1)

        stft = tf.signal.stft(tf_audio,
                              frame_length=stftParams['frame_length'],
                              frame_step=stftParams['frame_step'],
                              fft_length=frame_length)
        spectrogram = tf.abs(stft)

    num_spectrogram_bins = spectrogram.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=mfccParams['num_mel_bins'],
                                                                        num_spectrogram_bins=num_spectrogram_bins,
                                                                        sample_rate=mfccParams['sampling_rate'],
                                                                        lower_edge_hertz=mfccParams['lower_frequency'],
                                                                        upper_edge_hertz=mfccParams['upper_frequency'])

    mel_spectrogram = tf.tensordot(spectrogram,
                                   linear_to_mel_weight_matrix,
                                   1)

    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coefficients]

    return mfccs


Popen('sudo sh -c "echo performance >'
      '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
      shell=True).wait()

path = "../datasets/yes_no"
# path = "/content/yesNo_"

# STFT parameters slow
sftf_param_slow = {'frame_length': 8,
                   'frame_step': 16}

# STFT parameters fast
sftf_param_fast = {'frame_length': 2,
                   'frame_step': 4}

# MFCC_slow parameters
mfccSlow_param = {'num_mel_bins': 40,
                  'lower_frequency': 20,
                  'upper_frequency': 4000,
                  'sampling_rate': 16000}

# MFCC_fast parameters
mfccFast_param = {'num_mel_bins': 40,
                  'lower_frequency': 690,
                  'upper_frequency': 1800,
                  'sampling_rate': 3600}

num_coefficients = 10

mfccSlow_execTime = 0
mfccFast_execTime = 0
SNR = 0
num_file = 0

for file in os.listdir(path):
    start = time.time()
    mfccSlow = audio_processing(os.path.join(path, file), sftf_param_slow, mfccSlow_param, num_coefficients)
    end = time.time()
    mfccSlow_execTime += end - start

    start = time.time()
    mfccFast = audio_processing(os.path.join(path, file), sftf_param_fast, mfccFast_param, num_coefficients, True, 4000)
    end = time.time()
    mfccFast_execTime += end - start

    SNR += 20 * math.log10(np.linalg.norm(mfccSlow)/np.linalg.norm(mfccSlow - mfccFast + 1e-6))

    num_file += 1

print("MFCC shape: ", mfccSlow.shape, mfccFast.shape)

# SNR = 20 * math.log10(np.linalg.norm(mfccSlow) / np.linalg.norm(tf.subtract(mfccSlow - mfccFast) + 1e-6))

print(f"MFCC slow = {mfccSlow_execTime / num_file * 1000} ms")
print(f"MFCC fast = {mfccFast_execTime / num_file * 1000} ms")
print(f"SNR = {SNR / num_file} dB")

assert mfccSlow.shape == mfccFast.shape, "The shape of MFCCslow != shape of MFCCfast"
assert SNR / num_file > 10.40, "SNR is < 10.40!"
assert mfccFast_execTime / num_file * 1000 > 18, "Attention, MFCCfast is not so fast"

# assert tf.shape(mfccSlow) == tf.shape(mfccFast), "The shape of MFCCslow != shape of MFCCfast"
