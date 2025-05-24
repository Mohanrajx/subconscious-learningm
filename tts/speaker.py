import pyttsx3
import os

def text_to_speech(text, filename='output.wav'):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # words per minute
    engine.setProperty('volume', 1.0)  # max volume

    # Save speech to file
    engine.save_to_file(text, filename)
    engine.runAndWait()
    print(f"[INFO] Saved: {filename}")
