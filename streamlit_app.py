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
            with st.spinner("Analyzing and finding available AI models..."):
                # 1. Setup API
                genai.configure(api_key=key)
                
                # 2. Find a model that works for YOU
                available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                if not available_models:
                    st.error("No models found for this API key. Check your Google AI Studio project.")
                    st.stop()
                
                # Use the first one found (usually gemini-pro or gemini-1.5-flash)
                model_name = available_models[0]
                model = genai.GenerativeModel(model_name)
                
                # 3. Audio Analysis
                y, sr = librosa.load(song)
                tempo_data, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo_data[0]) if isinstance(tempo_data, (np.ndarray, list)) else float(tempo_data)
                
                # 4. Generate Vision
                prompt = f"Describe an abstract painting for a song with {int(tempo)} BPM."
                response = model.generate_content(prompt)
                
                st.success(f"Using Model: {model_name}")
                st.subheader("The AI's Vision:")
                st.write(response.text)
                
        except Exception as e:
            st.error(f"Error Details: {e}")
