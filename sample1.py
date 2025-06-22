import whisper
def transcribe_audio(filename="recorded.wav"):
    model=whisper.load_model("base")
    print("🧠 Transcribing...")
    result=model.transcribe(filename)
    print("📄 Transcription")
    print(result["text"])
transcribe_audio()