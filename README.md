This project implements an Emotion-Aware Speech Recognition System that processes an MP4 video, extracts its audio, transcribes the speech using OpenAI's Whisper model, and detects the speaker's emotional state using audio features like MFCCs and pitch. It uses basic machine learning (Support Vector Machine) for classifying emotions and is intended as a prototype for integrating emotion understanding in speech interfaces or media analytics.

Working Principle
The workflow of the code can be broken down into the following main steps:

1. Load Whisper Model
python
Copy
Edit
model = whisper.load_model("base")
The Whisper model is loaded to transcribe speech from audio. This model is pre-trained and can detect multiple languages and transcribe audio accurately.

2. Convert MP4 to WAV
python
Copy
Edit
convert_mp4_to_wav(mp4_path, wav_path)
The MP4 video file is converted to a WAV audio file using moviepy. This format is more suitable for audio processing and feature extraction.

3. Transcribe Audio
python
Copy
Edit
transcribe_audio(wav_path)
The audio is transcribed using Whisper, returning both the recognized text and the detected language. Whisper handles language detection automatically.

4. Extract Audio Features
python
Copy
Edit
extract_audio_features(audio_path)
Two types of features are extracted using librosa:

MFCCs (Mel Frequency Cepstral Coefficients): Capture timbral (tone quality) features of speech.

Pitch: Related to the perceived frequency of the voice (prosody/emotion clues).
These are combined into a 26-dimensional feature vector (13 MFCCs + 13 pitch values).

5. Train Emotion Classifier
python
Copy
Edit
train_emotion_classifier()
A simple Support Vector Machine (SVM) classifier is trained on randomly generated feature vectors labeled with mock emotions: ['happy', 'sad', 'angry', 'neutral']. This simulates how a classifier would work if trained on real labeled emotion data.

6. Predict Emotion
python
Copy
Edit
classify_emotion(features, classifier, label_encoder)
The previously extracted audio features are fed into the classifier to predict the speaker's emotion. The label encoder is used to convert between numeric labels and string class names.

7. Display Results
Finally, the system prints:

The transcribed speech

The detected emotion

The language of the audio

This gives a holistic interpretation of the audio both in terms of content (what is said) and emotion (how it is said).
