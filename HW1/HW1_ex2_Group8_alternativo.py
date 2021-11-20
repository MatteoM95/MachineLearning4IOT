import os
import sys
import zipfile
import tensorflow as tf
import time
import numpy as np

from subprocess import Popen

from scipy import signal
import wave
from scipy.io import wavfile

EARLY_STOP = False
ITER_NUM = 20
VERBOSE = True


def MFCC_slow(file_path, frame_length, frame_step, num_mel_bins, sampling_rate, lower_frequency, upper_frequency,
              mel_coefs):
    Popen('sudo sh -c "echo performance >'
          '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
          shell=True).wait()

    time_results = []
    mfccs_results = []

    for i, filename in enumerate(os.listdir(file_path)):
        if not os.path.isdir(filename):

            time0 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            # #Read the audio signal
            # audio = tf.io.read_file(f'{file_path}{filename}')
            # #Convert the signal in a TensorFlow
            # tf_audio, rate = tf.audio.decode_wav(audio)
            # tf_audio = tf.squeeze(tf_audio, 1) #shape: (16000,)
            # =======================================================

            rate, audio = wavfile.read(f'{file_path}{filename}')
            tf_audio = tf.convert_to_tensor(audio, np.float32)

            if VERBOSE: print(f'>>audio: {tf_audio.shape}')

            time1 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            # Convert the waveform in a spectrogram applying the STFT
            stft = tf.signal.stft(tf_audio,
                                  frame_length=frame_length,
                                  frame_step=frame_step,
                                  fft_length=frame_length)
            spectrogram = tf.abs(stft)  # shape: (49,321)

            if VERBOSE: print(f'>>spectrogram: {spectrogram.shape}')

            time2 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            if i == 0:
                # Compute the log-scaled Mel spectrogram
                num_spectrogram_bins = spectrogram.shape[-1]

                linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    num_mel_bins,
                    num_spectrogram_bins,
                    sampling_rate,
                    lower_frequency,
                    upper_frequency)

            mel_spectrogram = tf.tensordot(
                spectrogram,
                linear_to_mel_weight_matrix,
                1)

            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)  # shape: (49,40)

            if VERBOSE: print(f'>>mel: {log_mel_spectrogram.shape}')

            time3 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            # Compute the MFCCs  #shape:(49,10)
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
                log_mel_spectrogram)[..., :mel_coefs]

            if VERBOSE: print(f'>>mfcc: {mfccs.shape}')

            time4 = time.time()

            duration = time4 - time0

            time_list = [duration, (time1 - time0), (time2 - time1), (time3 - time2), (time4 - time3)]

            time_results.append(time_list)
            mfccs_results.append(mfccs)

            if i == ITER_NUM - 1 and EARLY_STOP:
                return time_results, mfccs_results

    return time_results, mfccs_results


def MFCC_fast(file_path, frame_length, frame_step, num_mel_bins, sampling_rate, lower_frequency, upper_frequency,
              mel_coefs, factor):
    Popen('sudo sh -c "echo performance >'
          '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
          shell=True).wait()

    time_results = []
    mfccs_results = []

    for i, filename in enumerate(os.listdir(file_path)):
        if not os.path.isdir(filename):

            time0 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            rate, audio = wavfile.read(f'{file_path}{filename}')
            audio = signal.resample_poly(audio, 1, factor)
            tf_audio = tf.convert_to_tensor(audio, np.float32)

            if VERBOSE: print(f'>>audio: {tf_audio.shape}')

            time1 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            # Convert the waveform in a spectrogram applying the STFT
            stft = tf.signal.stft(tf_audio,
                                  frame_length=int(frame_length / factor),
                                  frame_step=int(frame_step / factor),
                                  fft_length=int(frame_length / factor))
            spectrogram = tf.abs(stft)

            if VERBOSE: print(f'>>spectrogram: {spectrogram.shape}')  # shape: (49,321)

            time2 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            if i == 0:
                # Compute the log-scaled Mel spectrogram
                num_spectrogram_bins = spectrogram.shape[-1]

                linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    num_mel_bins,
                    num_spectrogram_bins,
                    int(sampling_rate / factor),
                    lower_frequency,
                    int(upper_frequency / factor))

            mel_spectrogram = tf.tensordot(
                spectrogram,
                linear_to_mel_weight_matrix,
                1)

            log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

            if VERBOSE: print(f'>>mel: {log_mel_spectrogram.shape}')  # shape: (49,40)

            time3 = time.time()  # <<<<<<<<<<<<<<<<<<<<<<<<

            # Compute the MFCCs
            mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
                log_mel_spectrogram)[:, :mel_coefs]

            if VERBOSE: print(f'>>mfcc: {mfccs.shape}')  # shape:(49,10)

            time4 = time.time()

            duration = time4 - time0

            time_list = [duration, (time1 - time0), (time2 - time1), (time3 - time2), (time4 - time3)]

            time_results.append(time_list)
            mfccs_results.append(mfccs)

            if i == ITER_NUM - 1 and EARLY_STOP:
                return time_results, mfccs_results

    return time_results, mfccs_results


def getSNR(mfcc_listS, mfcc_listF):
    SNRs = []

    for mfccS, mfccF in zip(mfcc_listS, mfcc_listF):
        current_SNR = 20 * np.log10((np.linalg.norm(mfccS)) / (np.linalg.norm(mfccS - mfccF + 10e-6)))
        SNRs.append(current_SNR)

    return np.mean(SNRs)


if __name__ == "__main__":
    VERSION = 1.6
    print(f'--- V.{VERSION} ---')

    file_path = '../datasets/yes_no/'  # path to unziped files

    L = 1000  # ms
    l = 16  # ms
    s = 8  # ms

    rate = 16  # [samples/ms]

    frame_length = rate * l  # rate [samples/ms] * 16 [ms]
    frame_step = rate * s  # rate [samples/ms] * 8 [ms]

    num_mel_bins = 40
    sampling_rate = rate * 1000
    lower_frequency = 20  # Hz
    upper_frequency = 4000  # Hz
    mel_coefs = 10

    print('|--- MFCC_slow...')
    start_time = time.time()
    times_MFCC_slow, mfccs_MFCC_slow = MFCC_slow(file_path, frame_length, frame_step, num_mel_bins, sampling_rate,
                                                 lower_frequency, upper_frequency, mel_coefs)
    end_time = time.time()
    tags = ['reading', 'sftf', 'mel', 'mfccs']
    print("|   |--- Average execution time: ", np.mean(times_MFCC_slow, axis=0)[0] * 1000, ' ms')
    print("|   |      |--- tags: ", tags)
    print("|   |      |--- ms: ", np.mean(times_MFCC_slow, axis=0)[1:] * 1000)
    print("|   |      |--- %: ", np.mean(times_MFCC_slow, axis=0)[1:] * 100 / np.mean(times_MFCC_slow, axis=0)[0])
    print("|   |--- Total execution time: ", (end_time - start_time) / 60.0, ' min')
    print("|   |      |--- Effective min: ", np.sum(times_MFCC_slow, axis=0)[0] / 60.0)
    print("|   |      |--- Effective %: ", np.sum(times_MFCC_slow, axis=0)[0] * 100 / (end_time - start_time))
    print("|   |--- Executed files: ", len(times_MFCC_slow))
    print("|")

    factor = 2

    print(f'|--- MFCC_fast ({factor})...')
    start_time = time.time()
    times_MFCC_fast, mfccs_MFCC_fast = MFCC_fast(file_path, frame_length, frame_step, num_mel_bins, sampling_rate,
                                                 lower_frequency, upper_frequency, mel_coefs, factor)
    end_time = time.time()
    tags = ['reading', 'sftf', 'mel', 'mfccs']
    print("|   |--- Average execution time: ", np.mean(times_MFCC_fast, axis=0)[0] * 1000, ' ms')
    print("|   |      |--- tags: ", tags)
    print("|   |      |--- ms: ", np.mean(times_MFCC_fast, axis=0)[1:] * 1000)
    print("|   |      |--- %: ", np.mean(times_MFCC_fast, axis=0)[1:] * 100 / np.mean(times_MFCC_fast, axis=0)[0])
    print("|   |--- Total execution time: ", (end_time - start_time) / 60.0, ' min')
    print("|   |      |--- Effective min: ", np.sum(times_MFCC_fast, axis=0)[0] / 60.0)
    print("|   |      |--- Effective %: ", np.sum(times_MFCC_fast, axis=0)[0] * 100 / (end_time - start_time))
    print("|   |--- Executed files: ", len(times_MFCC_fast))
    print("|")

    print(f'|--- SNR...')
    mean_SNR = getSNR(mfccs_MFCC_slow, mfccs_MFCC_fast)
    print("|   |--- dB: ", mean_SNR)
