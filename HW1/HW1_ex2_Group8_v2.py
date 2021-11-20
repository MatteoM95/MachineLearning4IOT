import os
import zipfile
import tensorflow as tf
import time
from subprocess import Popen
import math
import numpy as np
from scipy.io import wavfile
from scipy import signal

EARLY_STOP = False
ITER_NUM = 50


def audio_processing(path_dir, stftParams, mfccParams, num_coefficients, factor=1):
    Popen('sudo sh -c "echo performance >'
          '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
          shell=True).wait()

    time_results = []
    mfccs_results = []

    for i, filename in enumerate(os.listdir(path_dir)):
        if not os.path.isdir(filename):

            time0 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            _, audio = wavfile.read(f'{path_dir}{filename}')
            audio = signal.resample_poly(audio, 1, factor)
            tf_audio = tf.convert_to_tensor(audio, np.float32)

            time1 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            stft = tf.signal.stft(tf_audio,
                                  frame_length=int(stftParams['frame_length'] / factor),
                                  frame_step=int(stftParams['frame_step'] / factor),
                                  fft_length=int(stftParams['frame_length'] / factor)
                                  )
            spectrogram = tf.abs(stft)

            time2 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

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

            # mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

            time3 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coefficients]

            time4 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<
            time_list = [(time4 - time0), (time1 - time0), (time2 - time1), (time3 - time2), (time4 - time3)]
            time_results.append(time_list)
            mfccs_results.append(mfccs)

            if i == ITER_NUM - 1 and EARLY_STOP:
                return time_results, mfccs_results

    return time_results, mfccs_results


def compute_average_SNR(mfcc_list_slow, mfcc_list_fast):
    SNRs = []

    for mfccS, mfccF in zip(mfcc_list_slow, mfcc_list_fast):
        current_SNR = 20 * np.log10((np.linalg.norm(mfccS)) / (np.linalg.norm(mfccS - mfccF + 10e-6)))
        SNRs.append(current_SNR)

    return np.mean(SNRs)


if __name__ == "__main__":
    dir_path = "../datasets/yes_no/"
    # path = "/content/yesNo_"

    rate = 16  # [sample/ms]
    factor = 2

    # STFT parameters slow
    sftf_param = {'frame_length': 16 * rate,
                  'frame_step': 8 * rate}

    # MFCC_slow parameters
    mfccSlow_param = {'num_mel_bins': 40,
                      'lower_frequency': 20,
                      'upper_frequency': 4000,
                      'sampling_rate': rate * 1000}

    # MFCC_slow parameters
    mfccFast_param = {'num_mel_bins': 32,
                      'lower_frequency': 20,
                      'upper_frequency': 1000,
                      'sampling_rate': rate * 1000 / 4}  # 2000

    num_coefficients = 10

    mfccSlow_execTime = 0
    mfccFast_execTime = 0

    start = time.time()
    times_Slow, mfccSlow = audio_processing(dir_path, sftf_param, mfccSlow_param, num_coefficients, 1)
    end = time.time()
    mfccSlow_execTime = end - start

    tags = ['reading', 'sftf', 'mel', 'mfccs']
    print("|   |--- Average execution time: ", np.mean(times_Slow, axis=0)[0] * 1000, ' ms')
    print("|   |      |--- tags: ", tags)
    print("|   |      |--- ms: ", np.mean(times_Slow, axis=0)[1:] * 1000)
    print("|   |      |--- %: ", np.mean(times_Slow, axis=0)[1:] * 100 / np.mean(times_Slow, axis=0)[0])
    print("|   |--- Total execution time: ", mfccSlow_execTime / 60.0, ' min')
    print("|   |      |--- Effective min: ", np.sum(times_Slow, axis=0)[0] / 60.0)
    print("|   |      |--- Effective %: ", np.sum(times_Slow, axis=0)[0] * 100 / mfccSlow_execTime)
    print("|   |--- Executed files: ", len(times_Slow))
    print("|")

    start = time.time()
    times_Fast, mfccFast = audio_processing(dir_path, sftf_param, mfccFast_param, num_coefficients, factor)
    end = time.time()
    mfccFast_execTime = end - start

    tags = ['reading', 'sftf', 'mel', 'mfccs']
    print("|   |--- Average execution time: ", np.mean(times_Fast, axis=0)[0] * 1000, ' ms')
    print("|   |      |--- tags: ", tags)
    print("|   |      |--- ms: ", np.mean(times_Fast, axis=0)[1:] * 1000)
    print("|   |      |--- %: ", np.mean(times_Fast, axis=0)[1:] * 100 / np.mean(times_Fast, axis=0)[0])
    print("|   |--- Total execution time: ", mfccFast_execTime / 60.0, ' min')
    print("|   |      |--- Effective min: ", np.sum(times_Fast, axis=0)[0] / 60.0)
    print("|   |      |--- Effective %: ", np.sum(times_Fast, axis=0)[0] * 100 / mfccFast_execTime)
    print("|   |--- Executed files: ", len(times_Fast))
    print("|")

    print(f'|--- SNR...')
    SNR_mean = compute_average_SNR(mfccSlow, mfccFast)
    print("|   |--- dB: ", SNR_mean)

    # print(f"MFCC slow = {np.mean(durationSlow) * 1000} ms")
    # print(f"MFCC fast = {np.mean(durationFast) * 1000} ms")
    # print(f"SNR = {SNR_mean} dB")

    # assert mfccSlow.shape == mfccFast.shape, "The shape of MFCCslow != shape of MFCCfast"
    # assert SNR / num_file > 10.40, "SNR is < 10.40!"
    # assert mfccFast_execTime / num_file * 1000 > 18, "Attention, MFCCfast is not so fast"

    # assert tf.shape(mfccSlow) == tf.shape(mfccFast), "The shape of MFCCslow != shape of MFCCfast"
