import streamlit as st
import librosa
import numpy as np
import google.generativeai as genai

st.title("🎨 EchoCanvas")

key = st.text_input("1. Paste Gemini API Key here:", type="password")
song = st.file_uploader("2. Upload an MP3/WAV file", type=["mp3", "wav"])

if st.button("Run Analysis"):
    if not key or not song:
        st.error("Missing Key or Song!")
    else:
        try:
            with st.spinner("Analyzing your music and calling the AI..."):
                # 1. Load song
                y, sr = librosa.load(song)
                
                # 2. Get Tempo (Handled for all library versions)
                tempo_data, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo_data[0]) if isinstance(tempo_data, (np.ndarray, list)) else float(tempo_data)
                
                # 3. Configure AI with the more stable model name
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-pro')
                
                # 4. Get the Vision
                prompt = f"The music has a tempo of {int(tempo)} BPM. Describe a beautiful, abstract digital painting that matches this speed. Mention colors and art style."
                response = model.generate_content(prompt)
                
                st.success(f"Success! Tempo detected: {int(tempo)} BPM")
                st.subheader("The AI's Vision:")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Error Details: {e}")
