import streamlit as st
import librosa
import numpy as np
import google.generativeai as genai

st.set_page_config(page_title="EchoCanvas Debug")
st.title("🎨 EchoCanvas: Debug Mode")

# Simple input for the key
key = st.text_input("1. Paste Gemini API Key here:", type="password")

# Simple file uploader
song = st.file_uploader("2. Upload an MP3/WAV file", type=["mp3", "wav"])

if st.button("Run Analysis"):
    if not key:
        st.error("Missing API Key!")
    elif not song:
        st.error("Please upload a song first!")
    else:
        try:
            with st.spinner("Talking to Google AI..."):
                # Configure AI
                genai.configure(api_key=key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                # Test Analysis
                y, sr = librosa.load(song)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                
                # Test AI Response
                response = model.generate_content(f"Describe a painting for a song with {int(tempo)} BPM.")
                
                st.success(f"It worked! Tempo detected: {int(tempo)} BPM")
                st.write("**AI's Visual Idea:**")
                st.info(response.text)
                
        except Exception as e:
            # THIS WILL TELL US THE REAL ERROR
            st.error(f"Error Details: {e}")
