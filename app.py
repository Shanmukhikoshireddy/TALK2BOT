from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import librosa
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import google.generativeai as genai
from keras.models import load_model
import uuid
from langdetect import detect

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("static", exist_ok=True)

print("🔄 Loading emotion model...")
model = load_model('emotion_model.h5')
print("✅ Emotion model loaded.")

label_map_inv = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'ps', 6: 'sad'}

genai.configure(api_key="your API key here")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def convert_webm_to_wav(webm_path, wav_path):
    print(f"🔄 Converting {webm_path} to WAV...")
    audio = AudioSegment.from_file(webm_path, format="webm")
    audio.export(wav_path, format="wav")
    print(f"✅ Converted to {wav_path}")

def detect_language(text):
    try:
        lang = detect(text)
        print(f"🌐 Detected language: {lang}")
        return lang
    except:
        print("⚠️ Language detection failed. Defaulting to English.")
        return "en"

def transcribe_audio(file_path):
    print(f"🧠 Transcribing audio from {file_path}...")
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.record(source)
        try:
            text = r.recognize_google(audio, language="te-IN")
            print(f"✅ Transcription (Telugu try): {text}")
            lang = detect_language(text)
            if lang != 'te':
                print("🔄 Retrying in English...")
                text = r.recognize_google(audio, language="en-US")
                print(f"✅ Transcription (English): {text}")
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            text = "క్షమించండి, నేను అర్థం చేసుకోలేకపోయాను."
        return text

def predict_emotion(file_path):
    print(f"🧠 Predicting emotion from {file_path}...")
    y, sr_ = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr_, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)
    pred = model.predict(mfcc)
    emotion = label_map_inv[np.argmax(pred)]
    print(f"✅ Detected Emotion: {emotion}")
    return emotion

def generate_response(text, emotion):
    lang = detect_language(text)
    print(f"💬 Generating Gemini response in {'Telugu' if lang == 'te' else 'English'}...")

    if lang == 'te':
        prompt = f"""
        మీరు ఒక అనుకూలమైన వాయిస్ అసిస్టెంట్. వినియోగదారి సందేశం మరియు భావోద్వేగం ఆధారంగా తెలుగులో స్పందించండి.
        వినియోగదారి సందేశం: {text}
        వినియోగదారి భావోద్వేగం: {emotion}
        """
    else:
        prompt = f"""
        You are a friendly voice assistant. Based on the user's message and detected emotion, respond in English in a kind and helpful way.
        User message: {text}
        User emotion: {emotion}
        """

    config = genai.types.GenerationConfig(max_output_tokens=150, temperature=0.3)
    response = gemini_model.generate_content(prompt, generation_config=config)
    print(f"✅ Gemini response generated.")
    return response.text

def generate_voice(text, output_path="static/response.mp3"):
    lang = detect_language(text)
    tts_lang = "te" if lang == "te" else "en"
    print(f"🎤 Converting response to speech in {tts_lang}...")
    tts = gTTS(text=text, lang=tts_lang)
    tts.save(output_path)
    print(f"✅ TTS audio saved at: {output_path}")
    return output_path

@app.route("/")
def index():
    return render_template("index.html")

from datetime import datetime
response_counter = 0

@app.route("/process_audio", methods=["POST"])
def process_audio():
    global response_counter

    if "audio_data" not in request.files:
        print("❌ No audio_data received in request.")
        return jsonify({"error": "No audio data received"}), 400

    file = request.files["audio_data"]
    print("✅ Audio file received.")

    webm_path = os.path.join(app.config['UPLOAD_FOLDER'], "input.webm")
    wav_path = os.path.join(app.config['UPLOAD_FOLDER'], "input.wav")

    file.save(webm_path)
    print(f"✅ Saved file to {webm_path}")

    convert_webm_to_wav(webm_path, wav_path)

    text = transcribe_audio(wav_path)
    emotion = predict_emotion(wav_path)
    response = generate_response(text, emotion)

    response_counter += 1
    response_audio_filename = f"response{response_counter}.mp3"
    response_audio_path = os.path.join("static", response_audio_filename)

    generate_voice(response, response_audio_path)

    return jsonify({
        "text": text,
        "response": response,
        "audio_url": f"/static/{response_audio_filename}"
    })

@app.route("/regenerate", methods=["POST"])
def regenerate():
    data = request.get_json()
    user_text = data.get("text", "")
    if not user_text:
        return jsonify({"error": "No text received"}), 400

    emotion = "neutral"
    response = generate_response(user_text, emotion)

    response_id = str(uuid.uuid4())[:8]
    audio_path = f"static/response_{response_id}.mp3"
    generate_voice(response, audio_path)

    
    return jsonify({
    "text": text,
    "response": response,
    "emotion": emotion,
    "language": detect_language(text),
    "audio_url": f"/static/{response_audio_filename}"
     })


import atexit
import glob

def cleanup_audio_files():
    print("🧹 Cleaning up audio files...")
    for file in glob.glob("static/response*.mp3"):
        try:
            os.remove(file)
            print(f"🗑️ Deleted: {file}")
        except Exception as e:
            print(f"⚠️ Failed to delete {file}: {e}")

atexit.register(cleanup_audio_files)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
