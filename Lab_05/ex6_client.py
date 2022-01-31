import numpy as np
import pyaudio
import datetime
import requests
from io import BytesIO
from scipy import signal
import base64
import wave
import json



def main():
    chunk = 2400
    resolution = pyaudio.paInt16
    samp_rate = 48000
    record_secs = 1
    chunks = (samp_rate// chunk) * record_secs

    audio = pyaudio.PyAudio()

    now = datetime.datetime.now()
    timestamp = int(now.timestamp())

    print("Start streaming...")

    stream = audio.open(format=resolution, rate= samp_rate, channels=1,
                        input_device_index=0, input=True,
                        frames_per_buffer=chunk)
    frames = []
    for _ in range(chunks): 
        data = stream.read(chunk)
        frames.append(data)    
    stream.stop_stream()

    print("End streaming...")

    audio = np.frombuffer(b''.join(frames), dtype=np.int16)
    audio = signal.resample_poly(audio, 1, 48000/16000)
    audio = audio.astype(np.int16)
    buf = BytesIO()

    wavefile = wave.open(buf, 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(16000)
    wavefile.writeframes(audio.tobytes())
    wavefile.close()
    buf.seek(0)

    audio_b64bytes = base64.b64encode(buf.read())
    audio_string = audio_b64bytes.decode()

    audio_b64bytes = base64.b64encode(buf.read())
    audio_string = audio_b64bytes.decode()

    body = {
        # my url
        "bn": "http://192.168.1.92/",
        "bt": timestamp,
        "e": [
            {
                "n": "audio",
                "u": "/",
                "t": 0,
                "vd": audio_string
            }
        ]
    }

    url = "http://192.168.1.232:8080/dscnn"

    # I don't need to manually convert body in a json if I use the
    # json parameter in the put request
    r = requests.put(url, json=body)

    if r.status_code == 200:
        rbody = r.json()
        prob = rbody['probability']
        label = rbody['label']
        print("{} ({}%)".format(label, prob))
    else:
        print("Error")
        print(r.text)



if __name__ == '__main__':
    main()