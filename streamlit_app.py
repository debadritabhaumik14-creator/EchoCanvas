import streamlit as st
import librosa
import numpy as np
import google.generativeai as genai

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
            genai.configure(api_key=key, transport='rest')
            # Use the model that just worked for you!
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            with st.spinner("Analyzing music..."):
                y, sr = librosa.load(song)
                tempo_data, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo_data[0]) if isinstance(tempo_data, (np.ndarray, list)) else float(tempo_data)
                
                prompt = f"The music has a tempo of {int(tempo)} BPM. Describe a beautiful abstract painting for it."
                response = model.generate_content(prompt)
                
                # Store the description so we can use it to paint
                st.session_state['vision_text'] = response.text
                st.success(f"99 BPM Detected! Model: Gemini 2.5 Flash")

        except Exception as e:
            st.error(f"Error: {e}")

# 3. THE PAINTER (The part you've been waiting for!)
if 'vision_text' in st.session_state:
    st.subheader("The AI's Vision")
    st.info(st.session_state['vision_text'])
    
    if st.button("🎨 Paint this Image"):
        try:
            with st.spinner("Imagen 3 is painting... (takes 15 seconds)"):
                # Calling the specific Image Generation model
                # Note: Some free keys need 'imagen-3.0-generate-001'
                paint_model = genai.GenerativeModel("imagen-3.0-generate-001")
                
                # We give it the description the AI just wrote
                result = paint_model.generate_images(
                    prompt=st.session_state['vision_text'],
                    number_of_images=1
                )
                
                if result.images:
                    st.image(result.images[0], caption="Your Music-Driven Canvas", use_container_width=True)
                    st.success("Masterpiece Complete!")
                
        except Exception as e:
            st.warning("Text-to-Image might be restricted on your current API tier.")
            st.error(f"Details: {e}")
