import streamlit as st
import librosa
import numpy as np
import google.generativeai as genai
import time

# Set Page Config
st.set_page_config(page_title="EchoCanvas", page_icon="🎨")
st.title("🎵 EchoCanvas")
st.write("Turn your music into 4K artwork with Imagen 3.")

# --- inputs ---
key = st.sidebar.text_input("1. Paste Gemini API Key here:", type="password")
song = st.file_uploader("2. Upload an MP3/WAV file", type=["mp3", "wav"])

if song:
    st.audio(song)

# --- analysis ---
if st.button("Listen to Music"):
    if not key or not song:
        st.error("Missing Key or Song!")
    else:
        try:
            with st.spinner("Analyzing audio..."):
                # Load song
                y, sr = librosa.load(song)
                
                # Get Tempo
                tempo_data, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo = float(tempo_data[0]) if isinstance(tempo_data, (np.ndarray, list)) else float(tempo_data)
                
                # Get Energy (new!)
                energy = np.mean(librosa.feature.rms(y=y))
                
                st.success(f"BPM: {int(tempo)} | Energy: {energy:.2f}")
                st.session_state['visual_details'] = f"{int(tempo)} BPM, abstract painting, vibrant colors, 4k"

        except Exception as e:
            st.error(f"Error: {e}")

# --- image generation ---
if 'visual_details' in st.session_state and key:
    if st.button("Paint this Image with Imagen 3"):
        try:
            genai.configure(api_key=key)
            
            with st.spinner("Talking to Imagen 3... this takes 10-20 seconds..."):
                # This is the magic line that calls the Imagen 3 model!
                model = genai.GenerativeModel("imagen-3.0-generate-001")
                
                # Generate the image
                result = model.generate_images(
                    prompt=f"{st.session_state['visual_details']}. highly detailed, artistic.",
                    number_of_images=1
                )
                
                if result.images:
                    st.subheader("Your Music-Driven Canvas:")
                    # Show the first generated image
                    st.image(result.images[0], use_container_width=True)
                    st.success("Painted!")
                else:
                    st.warning("No image generated. This model might be restricted on your free key.")
                    
        except Exception as e:
            # THIS IS A LIKELY PLACE FOR A 404/PERMISSION ERROR
            st.error(f"Imagen Error (Check Model Permissions): {e}")
