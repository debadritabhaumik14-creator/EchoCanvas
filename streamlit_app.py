# --- image generation ---
if 'visual_details' in st.session_state and key:
    if st.button("Paint this Image"):
        try:
            # We are using a more universal "Client" method here
            from google import genai
            from google.genai import types
            from PIL import Image
            from io import BytesIO

            client = genai.Client(api_key=key)
            
            with st.spinner("Imagen 3 is painting your music..."):
                # Call the model directly through the client
                response = client.models.generate_images(
                    model='imagen-3.0-generate-001',
                    prompt=st.session_state['visual_details'],
                    config=types.GenerateImagesConfig(number_of_images=1)
                )

                if response.generated_images:
                    # Convert the raw data into a viewable image
                    img_data = response.generated_images[0].image.image_bytes
                    image = Image.open(BytesIO(img_data))
                    
                    st.subheader("Your Music-Driven Canvas:")
                    st.image(image, use_container_width=True)
                    st.success("Masterpiece Complete!")
                else:
                    st.warning("Google didn't return an image. Try a different song!")

        except Exception as e:
            st.error(f"Almost there! Just a small setup error: {e}")
            st.info("Tip: If you see 'Model Not Found', go to AI Studio and create a NEW key in a NEW project.")
