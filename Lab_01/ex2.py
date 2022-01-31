# Use the pyaudio package to drive the microphone 
# (with the blocking mode) and the wave package to 
# store the samples on disk.

import pyaudio
import wave

import os

from argparse import ArgumentParser


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
	FILENAME = args.name
	if FILENAME == None:
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


def main():

	parser = ArgumentParser()

	parser.add_argument('--chunk', type=int, default=1024, help='Set number of chunks')
	parser.add_argument('--format', type=str, default='Int16', help='Set the format of the audio track [Int8,Int16,Int32]')
	parser.add_argument('--channels', type=int, default=2, help='Set the number of channels')
	parser.add_argument('--seconds', type=int, default=3, help='Set the length of the recording (seconds)')
	parser.add_argument('--rate', type=int, default=44100, help='Set the rate')
	parser.add_argument('--name', type=str, default=None, help='Set the name of the audio track')

	args = parser.parse_args()

	record_audio(args)

if __name__ == '__main__':
    main()



"""Benchmark of the sizes between different combinations of rate and format
-rw-r--r-- 1 pi pi  528428 Oct 22 16:45 Int16_44100Hz_3s.wav
-rw-r--r-- 1 pi pi  573484 Oct 22 16:47 Int16_48000Hz_3s.wav
-rw-r--r-- 1 pi pi 1056812 Oct 22 16:46 Int32_44100Hz_3s.wav
-rw-r--r-- 1 pi pi 1146924 Oct 22 16:47 Int32_48000Hz_3s.wav
-rw-r--r-- 1 pi pi  264236 Oct 22 16:46 Int8_44100Hz_3s.wav
-rw-r--r-- 1 pi pi  286764 Oct 22 16:47 Int8_48000Hz_3s.wav
"""