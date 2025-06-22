# audio_utils.py
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import tempfile

# Load model once and reuse
_whisper_model = whisper.load_model("base")

def record_audio(duration=10, sample_rate=16000):
    """Records audio from the microphone and returns the file path."""
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()

    # Use a temp file to store the recording
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(temp_wav.name, sample_rate, audio)
    return temp_wav.name

def transcribe_audio(file_path):
    """Transcribes audio using Whisper model and returns the text."""
    result = _whisper_model.transcribe(file_path)
    return result["text"]
