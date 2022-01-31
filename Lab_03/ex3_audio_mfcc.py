import argparse
import numpy as np
import pyaudio
import tensorflow as tf
import time
import wave
from io import BytesIO
from scipy.io import wavfile
from scipy import signal


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
args = parser.parse_args()

interpreter = tf.lite.Interpreter(model_path='./models/{}.tflite'.format(args.model))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

chunk = 4800
resolution = pyaudio.paInt16
samp_rate = 48000
record_secs = 1 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii)
chunks = int((samp_rate / chunk) * record_secs)

length = int(0.040*16000)
stride = int(0.020*16000)
num_mel_bins = 40
spectrogram_width = (16000 - length) // stride + 1
num_spectrogram_bins = length // 2 + 1
num_coefficients = 10
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, 16000, 20, 4000)

buf = BytesIO()

audio = pyaudio.PyAudio() # create pyaudio instantiation
stream = audio.open(format=resolution, rate=samp_rate, channels=1,
                    input_device_index=dev_index, input=True,
                    frames_per_buffer=chunk)
stream.stop_stream()

COMMANDS = ['stop', 'up', 'yes', 'right', 'left', 'no', 'silence', 'down', 'go']


while True:
    frames = []
    buf.seek(0)

    print('record')
    time.sleep(0.5)

    stream.start_stream()
    for ii in range(chunks):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()

    wavefile = wave.open(buf ,'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(audio.get_sample_size(resolution))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()
    buf.seek(0)

    
    sample, _ = tf.audio.decode_wav(buf.read())
    sample = tf.squeeze(sample, 1)
    start = time.time()
    sample = signal.resample_poly(sample, 1, 3)
    sample = tf.convert_to_tensor(sample, dtype=tf.float32)
    stft = tf.signal.stft(sample, length, stride,
            fft_length=length)
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :num_coefficients]
    mfccs = tf.reshape(mfccs, [1, spectrogram_width, num_coefficients, 1])
    end = time.time()
    preprocessing = (end-start)*1e3
    print('Preprocessing {:.3f}ms'.format(preprocessing))

    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], mfccs)
    interpreter.invoke()
    predicted = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    inference = (end-start)*1e3
    print('Inference {:.3f}ms'.format(inference))
    print('Total {:.3f}ms'.format(preprocessing+inference))
    index = np.argmax(predicted[0])
    print('Command:', COMMANDS[index])
    print()
    time.sleep(0.5)
