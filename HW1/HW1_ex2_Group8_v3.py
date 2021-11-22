# command line:python3 HW1_ex2_Group8.py
import os
import tensorflow as tf
import time
from subprocess import Popen
import numpy as np
from scipy.io import wavfile
from scipy import signal


def audio_processing(path_dir, stftParams, mfccParams, num_coefficients, factor=1, reading_method="tf"):
    Popen('sudo sh -c "echo performance >'
          '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
          shell=True).wait()

    totalTime = []
    mfccs = []

    for i, filename in enumerate(os.listdir(path_dir)):
        if not os.path.isdir(filename):

            start = time.time()

            # scipy library
            if reading_method == 'scipy':
                _, audio = wavfile.read(f'{path_dir}{filename}')
                audio = signal.resample_poly(audio, 1, factor)
                tf_audio = tf.convert_to_tensor(audio, dtype=tf.float32)

            # tensorflow library
            elif reading_method == 'tf':
                audio = tf.io.read_file(f'{path_dir}{filename}')
                tf_audio, _ = tf.audio.decode_wav(audio)
                if factor > 1:
                    tf_audio = tf.reshape(tf_audio, [int(mfccParams['sampling_rate'] / factor), factor])[:, 0]
                else:
                    tf_audio = tf.squeeze(tf_audio, 1)  # shape: (16000,)
            else:
                print("select reading method: tf or scipy")
                return None

            # Convert the waveform in a spectrogram applying the STFT
            stft = tf.signal.stft(tf_audio,
                                  frame_length=int(stftParams['frame_length'] / factor),
                                  frame_step=int(stftParams['frame_step'] / factor),
                                  fft_length=int(stftParams['frame_length'] / factor)
                                  )
            spectrogram = tf.abs(stft)

            #  #Compute the log-scaled Mel spectrogram
            if i == 0:
                num_spectrogram_bins = spectrogram.shape[-1]
                linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    num_mel_bins=mfccParams['num_mel_bins'],
                    num_spectrogram_bins=num_spectrogram_bins,
                    sample_rate=int(mfccParams['sampling_rate'] / factor),
                    lower_edge_hertz=mfccParams['lower_frequency'],
                    upper_edge_hertz=mfccParams['upper_frequency'])

            mel_spectrogram = tf.tensordot(spectrogram,
                                           linear_to_mel_weight_matrix,
                                           1)

            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

            #Compute the MFCC
            mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coefficients]

            finish = time.time()
            totalTime.append(finish - start)
            mfccs.append(mfcc)

    return np.mean(totalTime, axis=0), mfccs


def compute_average_SNR(mfcc_slow, mfcc_fast):
    SNRs = []

    for mfccS, mfccF in zip(mfcc_slow, mfcc_fast):
        tmp_SNR = 20 * np.log10((np.linalg.norm(mfccS)) / (np.linalg.norm(mfccS - mfccF + 10e-6)))
        SNRs.append(tmp_SNR)

    return np.mean(SNRs)


if __name__ == "__main__":
    dir_path = "../datasets/yes_no/"

    rate = 16  # [sample/ms]
    factor = 2

    # STFT parameters
    sftf_param = {'frame_length': 16 * rate, # rate [samples/ms] * 16 [ms]
                  'frame_step': 8 * rate} # rate [samples/ms] * 8 [ms]

    # MFCC_slow parameters
    mfccSlow_param = {'num_mel_bins': 40,
                      'lower_frequency': 20,
                      'upper_frequency': 4000,
                      'sampling_rate': rate * 1000}

    # MFCC_fast parameters
    mfccFast_param = {'num_mel_bins': 32,
                      'lower_frequency': 20,
                      'upper_frequency': 4000,
                      'sampling_rate': rate * 1000}

    num_coefficients = 10

    mfccSlow_execTime = 0
    mfccFast_execTime = 0

    avg_Execution_Slow, mfccSlow = audio_processing(dir_path, sftf_param, mfccSlow_param, num_coefficients, 1, "tf")

    avg_Execution_Fast, mfccFast = audio_processing(dir_path, sftf_param, mfccFast_param, num_coefficients, factor, "tf")

    SNR_mean = compute_average_SNR(mfccSlow, mfccFast)

    print(f"MFCC slow = {avg_Execution_Slow * 1000:.2f} ms")
    print(f"MFCC fast = {avg_Execution_Fast * 1000:.2f} ms")
    print(f"SNR = {SNR_mean:.2f} dB")
