import streamlit as st
import librosa
import numpy as np
import pickle
import os
from PIL import Image
import time

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Custom mapping function
def map_score(pred):
    j = int(pred)
    if (pred - j) > 0.7:
        return j + 1
    elif (pred - j) > 0.2 and (pred - j) <= 0.5:
        return float(j + 0.5)
    elif (pred - j) > 0.5:
        return float(j + 0.5)
    else:
        return j

# Feature extraction function
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    features = np.hstack([mfcc_mean, mfcc_std, zcr, spectral_centroid, spectral_rolloff, chroma_stft])
    return features.reshape(1, -1)

# Streamlit UI
st.set_page_config(page_title="Grammar Scoring Engine", page_icon="🎧", layout="centered")

st.title("Grammar Scoring Engine 🎤")
st.write("Welcome to the Grammar Scoring Engine! Upload an audio file to get a grammar score.")


uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"], label_visibility="collapsed")

# Progress bar while processing
if uploaded_file is not None:
    with st.spinner("Processing audio... Please wait."):
        # Temporary save the uploaded file
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())
        
        # Display the audio file
        st.audio("temp.wav", format="audio/wav")
        
        # Extract features and make a prediction
        features = extract_features("temp.wav")
        
        # Show a progress bar for the model prediction
        progress_bar = st.progress(0)
        time.sleep(1) 
        
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.02)
        
        raw_pred = model.predict(features)[0]
        final_score = map_score(raw_pred)
        
        # Show the result with an icon and color
        st.markdown(f"### **Predicted Grammar Score:** {final_score:.1f}", unsafe_allow_html=True)

        # Remove the temporary audio file
        os.remove("temp.wav")
