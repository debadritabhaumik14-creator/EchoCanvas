import google.generativeai as genai
import streamlit as st

def generate_visual_prompt(features):
    # This function turns BPM/Energy into a poetic description
    bpm = features['tempo']
    energy = features['energy']
    
    if bpm > 120:
        vibe = "energetic, sharp geometric shapes, neon colors, fast motion blur"
    else:
        vibe = "calm, flowing watercolor, soft pastels, ethereal lighting"
        
    prompt = f"A beautiful abstract digital painting representing music with {bpm} BPM. Style: {vibe}. Highly detailed, 4k, artistic."
    return prompt

def get_ai_image(prompt, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash') # Or 'imagen-3.0-generate-001' if available
    
    # For now, let's have Gemini write a very detailed prompt for the user
    # In the full version, we'd call the Image Generation model here
    response = model.generate_content(f"Describe a beautiful piece of art based on this: {prompt}")
    return response.text
