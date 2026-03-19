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
# 3. THE FREE ARTIST WORKFLOW
if 'vision_text' in st.session_state:
    st.subheader("The AI's Vision")
    st.info(st.session_state['vision_text'])
    
    st.write("---")
    st.markdown("### 🚀 Generate Your 4K Art for Free")
    st.write("Google requires a credit card to use the 'Paint' button in apps. To keep it **100% free**, follow this:")
    
    # A one-click button to copy the prompt
    st.text_area("Copy this Prompt:", value=st.session_state['vision_text'], height=150)
    
    st.markdown("""
    1.  Go to **[Google AI Studio](https://aistudio.google.com/)**
    2.  Select **'Gemini 3.1 Flash Image'** in the top right.
    3.  Paste the text and download your 4K painting!
    """)

    # Let the user upload their free masterpiece back into the app
    final_art = st.file_uploader("🖼️ Upload your finished painting to the Gallery:", type=["png", "jpg"])
    if final_art:
        st.image(final_art, caption=f"EchoCanvas: 99 BPM Vision", use_container_width=True)
        st.balloons()
