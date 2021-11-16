import os
import zipfile
import tensorflow as tf
import time
from subprocess import Popen
import math
import numpy as np


def audio_processing(stftParams, mfccParams, num_coefficients):
    exec_time = 0
    num_file = 0

    frame_length = stftParams['frame_length'] * stftParams['frame_step']

    for file in os.listdir("../datasets/yes_no"):
        start = time.time()
        filename = os.path.join("../datasets/yes_no", file)
        #print("filename: ", filename)

        audio = tf.io.read_file(filename)
        tf_audio, rate = tf.audio.decode_wav(audio)

        tf_audio = tf.squeeze(tf_audio, 1)

        stft = tf.signal.stft(tf_audio,  
                            frame_length=stftParams['frame_length'],  
                            frame_step=stftParams['frame_step'], 
                            fft_length=frame_length) 
        spectrogram = tf.abs(stft)
        
        num_spectrogram_bins = spectrogram.shape[-1] 
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix( 
                                                                            num_mel_bins=mfccParams['num_mel_bins'], 
                                                                            num_spectrogram_bins=num_spectrogram_bins, 
                                                                            sample_rate=mfccParams['sampling_rate'], 
                                                                            lower_edge_hertz=mfccParams['lower_frequency'], 
                                                                            upper_edge_hertz=mfccParams['upper_frequency']) 
        mel_spectrogram = tf.tensordot( 
                                       spectrogram,  
                                       linear_to_mel_weight_matrix, 
                                       1) 

        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])) 

        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :num_coefficients]

        end = time.time() 
        exec_time += (end - start)
        num_file += 1

    return exec_time/num_file, mfccs

  
##  COLAB: make folder yesNo, upload the zip and then run this code 
#!mkdir -p yesNo_
#with zipfile.ZipFile("/content/yes_no.zip", 'r') as zip_ref:
#    zip_ref.extractall("/content/yesNo_")

Popen('sudo sh -c "echo performance >'  
      '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',    
      shell=True).wait()

# STFT parameters
sftf_param = {'frame_length':8, 
              'frame_step':16}

# MFCC_slow parameters
mfccSlow_param = {'num_mel_bins':40, 
                  'lower_frequency':20, 
                  'upper_frequency':4000, 
                  'sampling_rate':16000}

# MFCC_fast parameters
mfccFast_param = {'num_mel_bins':40, 
                  'lower_frequency':20, 
                  'upper_frequency':4000, 
                  'sampling_rate':16000}

num_coefficients = 10

mfccSlow_execTime, mfccSlow = audio_processing(sftf_param, mfccSlow_param, num_coefficients)
mfccFast_execTime, mfccFast = audio_processing(sftf_param, mfccFast_param, num_coefficients)

assert mfccSlow.shape == mfccFast.shape, "The shape of MFCCslow != shape of MFCCfast"
# assert tf.shape(mfccSlow) == tf.shape(mfccFast), "The shape of MFCCslow != shape of MFCCfast"

SNR = 20 * math.log10(np.linalg.norm(mfccSlow)/np.linalg.norm(mfccSlow - mfccFast + 1e-6))
# SNR = 20 * math.log10(np.linalg.norm(mfccSlow)/np.linalg.norm( tf.subtract(mfccSlow - mfccFast) + 1e-6))

print(f"MFCC slow = {mfccSlow_execTime} ms")
print(f"MFCC fast = {mfccFast_execTime} ms")
print(f"SNR = {SNR} ms")

assert SNR > 10.40, "SNR is < 10.40!"

assert mfccSlow_execTime - mfccFast_execTime > 18, "Attention, MFCCfast is not so fast"
