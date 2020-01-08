import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import speech_recognition as sr
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 120)

def convert_to_text():
    r = sr.Recognizer()
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording
    print("SPEAK NOW")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file
    write("newoutput.wav", fs, np.int16(myrecording / np.max(np.abs(myrecording)) * 32767))
    print("recording completed")

    hellow = sr.AudioFile('/home/anuj/PycharmProjects/RETINA/newoutput.wav')
    with hellow as source:
        audio = r.record(source)
        r.adjust_for_ambient_noise(source, duration=0.5)

    try:
        result = r.recognize_google(audio).lower()
        if result == 'hello':
            engine.say("yes, my lord")
            engine.runAndWait()
            engine.stop()
            print("result is :", result)

    except:
        print("Exception:")
        return

    return result
