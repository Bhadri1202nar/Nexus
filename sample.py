import sounddevice as sd
from scipy.io.wavfile import write
def record_audio(filename="recorded.wav",duration=10 , fs=44100):
    print("Recording")
    audio=sd.rec(int(duration*fs),samplerate=fs,channels=1)
    sd.wait()
    write(filename,fs,audio)
    print("Recording saved:" , filename)
record_audio()