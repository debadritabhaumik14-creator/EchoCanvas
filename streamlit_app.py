import streamlit as st
import librosa
import numpy as np
import google.generativeai as genai

# Page Config
st.set_page_config(page_title="EchoCanvas", page_icon="🎨")

st.title("🎵 EchoCanvas")
st.write("Upload a song and let AI paint the music.")

# --- SIDEBAR FOR API KEY ---
# Note: In the future, we will move this to 'Secrets'
api_key = st.sidebar.text_input("Enter Google Gemini API Key", type="password")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose an audio file (mp3, wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if st.button("Analyze & Generate"):
        if not api_key:
            st.error("Please enter your API Key in the sidebar first!")
        else:
            with st.spinner('Listening to the music...'):
                # 1. Analyze Audio
                y, sr = librosa.load(uploaded_file)
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                energy = np.mean(librosa.feature.rms(y=y))
                
                st.write(f"**Tempo:** {int(tempo)} BPM")
                st.write(f"**Energy:** {energy:.2f}")

                # 2. Generate Prompt with Gemini
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                vibe_query = f"The music has a tempo of {tempo} BPM and an energy of {energy}. Describe a beautiful, artistic, abstract image that represents this sound. Be very descriptive."
                
                response = model.generate_content(vibe_query)
                
                st.subheader("The AI's Vision:")
                st.info(response.text)
                
                st.success("Analysis complete! Next, we'll add actual image generation.")
