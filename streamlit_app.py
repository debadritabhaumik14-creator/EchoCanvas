import streamlit as st
import librosa
import numpy as np
import google.generativeai as genai

st.title("🎨 EchoCanvas: Debug Mode")

key = st.text_input("1. Paste Gemini API Key here:", type="password")
song = st.file_uploader("2. Upload an MP3/WAV file", type=["mp3", "wav"])

if st.button("Run Analysis"):
    if not key or not song:
        st.error("Missing Key or Song!")
    else:
        try:
            with st.spinner("Analyzing your music..."):
                # Load song
                y, sr = librosa.load(song)
                
                # FIX: Handle tempo as a single number
                tempo_data, _ = librosa.beat.beat_track(y=y, sr=sr)
                if isinstance(tempo_data, np.ndarray):
                    tempo = float(tempo_data[0])
                else:
                    tempo = float(tempo_data)
                
                # Configure AI
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Get the Vision
                response = model.generate_content(f"Describe a painting for a song with {int(tempo)} BPM.")
                
                st.success(f"Success! Tempo: {int(tempo)} BPM")
                st.subheader("The AI's Vision:")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Error Details: {e}")
