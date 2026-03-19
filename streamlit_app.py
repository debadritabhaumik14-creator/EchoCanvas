import streamlit as st
import librosa
import numpy as np
import google.generativeai as genai

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
            # --- API CONFIGURATION ---
            # This line forces the app to use the stable 'v1' instead of 'v1beta'
            genai.configure(api_key=key, transport='rest')
            
            with st.spinner("Finding best available model..."):
                # Automatically find the newest Flash model available to you
                available_models = [m.name for m in genai.list_models() 
                                   if 'generateContent' in m.supported_generation_methods 
                                   and 'flash' in m.name.lower()]
                
                # Use the newest one found (likely gemini-2.0-flash or 2.5)
                # If list is empty, it falls back to a safe 2.0 name
                model_id = available_models[0] if available_models else 'models/gemini-2.0-flash'
                model = genai.GenerativeModel(model_id)

            with st.spinner(f"Analyzing music with {model_id.split('/')[-1]}..."):
                # Handle Audio
                y, sr = librosa.load(song)
                tempo_data, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo_data[0]) if isinstance(tempo_data, (np.ndarray, list)) else float(tempo_data)
                
                prompt = f"The music has a tempo of {int(tempo)} BPM. Describe a beautiful abstract painting for it."
                response = model.generate_content(prompt)
                
                st.success(f"Success! Model used: {model_id}")
                st.subheader("The AI's Vision")
                st.info(response.text)

        except Exception as e:
            st.error(f"Error Details: {e}")
            st.info("Tip: If you still see 404, check that 'Generative Language API' is enabled in your Google Cloud Console.")
