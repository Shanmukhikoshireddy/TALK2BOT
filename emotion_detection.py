# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
import kagglehub
ejlok1_toronto_emotional_speech_set_tess_path = kagglehub.dataset_download('ejlok1/toronto-emotional-speech-set-tess')

print('Data source import complete.')


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
warnings.filterwarnings('ignore')

paths = []
labels = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break
print('Dataset is Loaded')

len(paths)

paths[:5]

labels[2795:2800]

df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()

df['label'].value_counts()

sns.countplot(data=df, x='label')

def waveplot(data, sr, emotion):
    plt.figure(figsize=(10,4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()

def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11,4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()

for i in range(7):
  label_map_inv= {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'ps', 6: 'sad'}
  emotion = label_map_inv[i]
  path = np.array(df['speech'][df['label']==emotion])[0]
  data, sampling_rate = librosa.load(path)
  waveplot(data, sampling_rate, emotion)
  spectogram(data, sampling_rate, emotion)
  Audio(path)

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

extract_mfcc(df['speech'][0])

X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))

X_mfcc

X = [x for x in X_mfcc]
X = np.array(X)
X.shape

## input split
X = np.expand_dims(X, -1)
X.shape

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()
y.shape

# Get actual label order used in training
label_order = enc.categories_[0].tolist()
label_map_inv = {i: label for i, label in enumerate(label_order)}
print("Correct label_map_inv:", label_map_inv)


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from keras.layers import BatchNormalization

model = Sequential([
    LSTM(32, return_sequences=False, input_shape=(40,1)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=64, callbacks=[early_stop])


# Get the exact order of labels from encoder
label_order = enc.categories_[0].tolist()

# Invert it to map prediction index to emotion label
label_map_inv = {i: label for i, label in enumerate(label_order)}

print("Label map used by model:", label_map_inv)


epochs = list(range(37))
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, label='train accuracy')
plt.plot(epochs, val_acc, label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, label='train loss')
plt.plot(epochs, val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


model.save('emotion_model.h5')


def predict_emotion_from_file(model_path, file_path, label_map_inv, sr=22050):
    """
    Predicts emotion from a single .wav file.

    Args:
        model_path: Path to the saved Keras model (.h5).
        file_path: Path to the input .wav audio file.
        label_map_inv: Dictionary mapping class index to emotion label.
        sr: Sample rate to load audio.

    Returns:
        Predicted emotion string.
    """
    from keras.models import load_model
    import librosa
    import numpy as np

    # Load model
    model = load_model(model_path)

    # Extract MFCCs (same as in training)
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    # Reshape for model input (40,1)
    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)

    # Predict

    prediction = model.predict(mfcc)
    predicted_index = np.argmax(prediction)
    print(predicted_index)
    predicted_emotion = label_map_inv[predicted_index]
    confidence = prediction[0][predicted_index]
    #print(f"Predicted Emotion: {predicted_emotion} (Confidence: {confidence:.2f})")
    return predicted_emotion


predict_emotion_from_file('emotion_model.h5', 'young_happy.wav', label_map_inv)

