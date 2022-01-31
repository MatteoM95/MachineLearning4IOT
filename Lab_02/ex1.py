import pyaudio
import wave

import os

from argparse import ArgumentParser

from scipy import signal
from scipy.io import wavfile

import time



def print_statistics(start, end, filename):

	# get time interval
	interval = end - start

	# get size informations
	len_audio = os.path.getsize(filename)
	len_audio_res = os.path.getsize(f'res_' + filename)

	# print statistics
	print(f'\n\n ----- {interval} seconds  ----- ')
	print(f'Length of the original audio signal = {len_audio}')
	print(f'Length of the resampled audio signal = {len_audio_res}\n')



def record_audio(args):

	# define pyaudio object
	p = pyaudio.PyAudio()

	# define the format
	if args.format == 'Int8':
		format = pyaudio.paInt8
	elif args.format == 'Int16':
		format = pyaudio.paInt16
	elif args.format == 'Int32':
		format = pyaudio.paInt32

	# open stream
	stream = p.open(format=format, channels=args.channels, rate=args.rate, input=True, frames_per_buffer=args.chunk)


	# record for the given amount of time
	print("Start recording")

	frames = []
	for i in range(0,int(args.rate / args.chunk * args.seconds)):
		data = stream.read(args.chunk)
		frames.append(data)

	print("End recording")

	# stop the stream and close it
	stream.stop_stream()
	stream.close()

	# terminate pyaudio's object
	p.terminate()

	# set the filename
	if args.name == None:
		# concatenate the values
		FILENAME = "{}_{}Hz_{}s.wav".format(args.format, args.rate, args.seconds)


	# overwrite the file if it is already present
	# - workaround : delete it
	if FILENAME in os.listdir():
		os.remove(FILENAME)


	# setup the final audio file and save it
	wf = wave.open(FILENAME, 'wb')
	wf.setnchannels(args.channels)
	wf.setsampwidth(p.get_sample_size(format))
	wf.setframerate(args.rate)
	wf.writeframes(b''.join(frames))

	# close write_file's object
	wf.close()

	print("File salved")

	return FILENAME


def main():

	# keep track of the execution time
	start = time.time()


	parser = ArgumentParser()

	parser.add_argument('--chunk', type=int, default=1024, help='Set number of chunks')
	parser.add_argument('--format', type=str, default='Int16', help='Set the format of the audio track [Int8,Int16,Int32]')
	parser.add_argument('--channels', type=int, default=2, help='Set the number of channels')
	parser.add_argument('--seconds', type=int, default=1, help='Set the length of the recording (seconds)')
	parser.add_argument('--rate', type=int, default=48000, help='Set the rate')
	parser.add_argument('--name', type=str, default=None, help='Set the name of the audio track')

	args = parser.parse_args()

	# Step 1 : Record an audio signal with L=1s, 48kHz sampling rate and 16-bit resolution
	filename = record_audio(args)


	# Step 2 : Read the audio signal
	rate, audio = wavfile.read(filename)


	# Step 3 : Resample the signal with poly-phase filtering at a frequency f=16kHz
	# resample_poly(audio, 1, sampling_ratio) -> sampling_ratio = 48000/16000 = 3
	sampling_ratio = args.rate / 16000
	audio = signal.resample_poly(audio, 1, sampling_ratio)


	# Step 4 : Measure the execution time, store the output on disk and measure the file size
	p = pyaudio.PyAudio()

	wf = wave.open("res_" + filename, 'wb')
	wf.setnchannels(args.channels)
	wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
	wf.setframerate(16000)
	wf.writeframes(b''.join(audio))

	# close write_file's object
	wf.close()


	# print statistics
	print_statistics(start, time.time(), filename)


if __name__  == '__main__':

	main()
