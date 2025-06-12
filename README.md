# 🎤 Voice Chatbot with Emotion Detection (English + తెలుగు)

A web-based AI voice chatbot that can:
- 🎙️ Accept voice input
- 🧠 Detect the user's emotion from speech
- 🗣️ Transcribe audio to text (Telugu & English)
- 🤖 Generate intelligent responses using Gemini AI
- 🔁 Let the user edit and regenerate responses
- 🔊 Speak the response back to the user

---

## 📁 Project Structure

```
├── app.py                      # Main Flask backend
├── templates/
│   └── index.html              # Frontend UI
├── static/
│   └── response*.mp3           # Auto-generated response audios
├── uploads/
│   └── input.webm/.wav         # Uploaded and converted audio
├── emotion_model.h5           # Pre-trained emotion classification model
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Features

- 🎙️ **Voice Recording**: Record your speech directly from the browser.
- 🔁 **Speech-to-Text**: Transcribes using Google Speech Recognition (supports Telugu & English).
- 🧠 **Emotion Detection**: Detects emotion from audio using a CNN model.
- 🤖 **Gemini AI Integration**: Generates responses using Google Gemini API (multilingual).
- ✏️ **Editable Messages**: Edit your question and regenerate a new response.
- 🔊 **Voice Output**: Converts the AI response into speech using gTTS.
- 🌐 **Language Detection**: Automatically detects whether you're speaking Telugu or English.

---

## 🧪 Technologies Used

- Python (Flask)
- HTML, CSS, JavaScript
- [Google Gemini API](https://ai.google.dev/)
- Google Speech Recognition API
- gTTS (Google Text-to-Speech)
- Pydub + FFmpeg for audio conversion
- Librosa for audio feature extraction
- Keras/TensorFlow for emotion classification

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/voice-chatbot-emotion.git
cd voice-chatbot-emotion
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Install system dependencies**

Make sure `ffmpeg` is installed:

```bash
# On Windows: install via ffmpeg.org or add to PATH
# On Ubuntu:
sudo apt update
sudo apt install ffmpeg
```

4. **Add Gemini API Key**

Update this line in `app.py` with your key:

```python
genai.configure(api_key="YOUR_API_KEY")
```

5. **Run the Flask app**

```bash
python app.py
```

Open in browser: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Emotion Detection Model

- Input: Audio clip (3 seconds)
- Extracts MFCC features using `librosa`
- CNN model trained on labeled emotion audio dataset
- Output: Predicted emotion (e.g., happy, sad, angry...)

Model file: `emotion_model.h5`

---

## 🖼️ Screenshot

![screenshot](Output.png)

---

## 🛠️ Troubleshooting

| Problem | Solution |
|--------|----------|
| ❌ Audio not recording | Check browser permissions |
| ❌ "Couldn't detect language" | Ensure you're speaking clearly; fallback is English |
| ❌ Regeneration error | Ensure API key is set and fix typo in `/regenerate` |
| ❌ FFmpeg errors | Install `ffmpeg` and add it to your system PATH |

---

## 📌 TODO (Optional Improvements)

- Add Whisper model for better transcription
- Emotion-based avatar reactions (3D/2D)
- Save chat history
- Real-time audio streaming
- Dark/light theme switch

---

## 📄 License

This project is for educational and personal use. Contact the author for commercial use.

---

## 🙌 Acknowledgements

- Google Gemini & gTTS
- SpeechRecognition & Librosa
- TensorFlow/Keras
- Inspiration from emotion-aware interfaces

---
