import librosa
import numpy as np

def get_audio_features(file):
    # 1. Load the song
    y, sr = librosa.load(file)
    
    # 2. Extract Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # 3. Extract Energy (Loudness/Intensity)
    rms = librosa.feature.rms(y=y)
    energy = np.mean(rms)
    
    # 4. Extract "Chroma" (The musical notes/key)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chroma)

    return {
        "tempo": float(tempo),
        "energy": float(energy),
        "mood_score": float(mean_chroma)
    }
