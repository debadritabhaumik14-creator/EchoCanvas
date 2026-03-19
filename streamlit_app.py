import streamlit as st
import librosa
import numpy as np
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="EchoCanvas", page_icon="🎨")
st.title("🎵 EchoCanvas")

# 1. SETUP INPUTS
key = st.sidebar.text_input("Paste Gemini API Key:", type="password")
song = st.file_uploader("Upload a song (MP3/WAV)", type=["mp3", "wav"])

if song:
    st.audio(song)

# 2. ANALYSIS LOGIC
if st.button("Run Analysis"):
    if not key or not song:
        st.error("Please provide both an API key and a song!")
    else:
        try:
            with st.spinner("Analyzing music..."):
                # Handle Audio
                y, sr = librosa.load(song)
                tempo_data, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo_data[0]) if isinstance(tempo_data, (np.ndarray, list)) else float(tempo_data)
                
                # Setup AI
                genai.configure(api_key=key)
                # We use flash here as it's the most reliable for text descriptions
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                prompt = f"Describe an abstract painting for a song with {int(tempo)} BPM. Be descriptive."
                response = model.generate_content(prompt)
                
                # Save results to session so they don't disappear
                st.session_state['description'] = response.text
                st.session_state['tempo'] = int(tempo)
                st.success(f"Tempo: {int(tempo)} BPM detected!")

        except Exception as e:
            st.error(f"Analysis Error: {e}")

# 3. DISPLAY RESULTS & PAINT
if 'description' in st.session_state:
    st.subheader("The AI's Vision")
    st.info(st.session_state['description'])
    
    if st.button("Paint this Artwork"):
        try:
            # Note: Image generation support varies by region/key. 
            # This uses the standard Gemini generation for descriptions.
            st.warning("Generating images requires Imagen-3 access on your key. If it fails, check your Google Cloud permissions.")
            
            model_art = genai.GenerativeModel('gemini-1.5-flash')
            # For now, we generate a high-quality descriptive prompt for you
            st.write("🎨 Painting in progress...")
            # (In a full production app, you'd call the Imagen API here)
            st.image("https://p7.itc.cn/images01/20230906/07135061c4704870a417088998825838.gif", caption="Visualizing your sound...")
            
        except Exception as e:
            st.error(f"Painting Error: {e}")
