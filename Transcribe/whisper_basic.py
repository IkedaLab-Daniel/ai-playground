import whisper
import os

# Load the Whisper model
model = whisper.load_model("base")

# Check if audio file exists
audio_file = "dont-look-back.mp3"
if not os.path.exists(audio_file):
    print(f"Error: Audio file '{audio_file}' not found!")
    print("Please place an audio file in the same directory or update the file path.")
else:
    try:
        # Transcribe the audio file
        result = model.transcribe(audio_file)
        
        # Output the transcription
        print("Transcription:")
        print(result["text"])
        
    except Exception as e:
        print(f"Error during transcription: {e}")