import argparse
import requests
import base64
import wave
import time


def main(args):


    url = "http://127.0.0.1:8080/"

    for _ in range(args.n_recordings):
    
        r = requests.get(url)

        if r.status_code == 200:
            body = r.json()

            timestamp = body['timestamp']

            output = ''

            for measure in body['measures']:

                if measure['what'] == 'audio':
                    audio = measure['value']
                else:
                    output += '{} \t {} {}\n'.format(timestamp, measure['value'], measure['unit'])

            # save audio 
            audio_bytes = base64.b64decode(audio)
            output_audio = "{}.wav".format(timestamp)
            wavefile = wave.open(output_audio, "wb")
            wavefile.setnchannels(1)
            wavefile.setsampwidth(2)
            wavefile.setframerate(48000)
            wavefile.writeframes(audio_bytes)
            wavefile.close()
                    
        else:
            output = 'Error {} : {}'.format(r.status_code, r.text)

        print(output)

        time.sleep(args.delay)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--n_recordings', type=int, default=1)
    parser.add_argument('-d', '--delay', type=float, default=3.0, help='Interval between two recordings')
    args = parser.parse_args()

    main(args)