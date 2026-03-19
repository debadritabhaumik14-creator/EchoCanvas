import streamlit as st
import librosa
import numpy as np
from google import genai # Note: This is the new 2026 client
from google.genai import types
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="EchoCanvas", page_icon="🎨")
st.title("🎵 EchoCanvas")

# 1. SETUP
key = st.sidebar.text_input("Paste Gemini API Key:", type="password")
song = st.file_uploader("Upload a song (MP3/WAV)", type=["mp3", "wav"])

if song:
    st.audio(song)

# 2. ANALYSIS 
if st.button("Run Analysis"):
    if not key or not song:
        st.error("Missing Key or Song!")
    else:
        try:
            # We use the new 'genai.Client' for everything now
            client = genai.Client(api_key=key)
            
            with st.spinner("Analyzing music & generating vision..."):
                # Audio Processing
                y, sr = librosa.load(song)
                tempo_data, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo_data[0]) if isinstance(tempo_data, (np.ndarray, list)) else float(tempo_data)
                
                # Text Generation (using your successful 2.5 Flash model)
                prompt = f"The music has a tempo of {int(tempo)} BPM. Describe a beautiful abstract painting for it."
                response = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=prompt
                )
                
                st.session_state['vision_text'] = response.text
                st.session_state['bpm'] = int(tempo)
                st.success(f"99 BPM Detected!")

        except Exception as e:
            st.error(f"Analysis Error: {e}")

# 3. THE PAINTER (2026 Nano Banana 2 Unified Version)
if 'vision_text' in st.session_state:
    st.subheader("The AI's Vision")
    st.info(st.session_state['vision_text'])
    
    if st.button("🎨 Paint this Image"):
        try:
            # We use the standard Client you already have
            client = genai.Client(api_key=key)
            
            with st.spinner("Nano Banana 2 is painting your music..."):
                # 2026 logic: Just ask the image model for content!
                # Note the model name: 'gemini-3.1-flash-image-preview'
                response = client.models.generate_content(
                    model='gemini-3.1-flash-image-preview', 
                    contents=f"Create a 4K abstract painting based on this description: {st.session_state['vision_text']}"
                )

                # Find the image in the response parts
                found_image = False
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        img_data = part.inline_data.data
                        image = Image.open(BytesIO(img_data))
                        st.image(image, caption=f"Visualized at {st.session_state.get('bpm', 99)} BPM", use_container_width=True)
                        st.success("Masterpiece Complete via Nano Banana 2!")
                        found_image = True
                        break
                
                if not found_image:
                    st.warning("The AI described a painting but didn't 'draw' it. Try clicking again!")
                
        except Exception as e:
            st.error(f"Painting Error: {e}")
            st.info("Tip: Double-check your API Key in Google AI Studio. Make sure 'Gemini 3.1 Flash Image' is in your model list.")
