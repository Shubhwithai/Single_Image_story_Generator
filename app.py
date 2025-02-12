import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO

# Initialize OpenAI client with Together.ai base URL
client = OpenAI(
    api_key=st.secrets['together_api_key'],
    base_url="https://api.together.xyz/v1"
)

def generate_image(prompt: str):
    """Generate an image using FLUX model"""
    try:
        response = client.images.generate(
            model="black-forest-labs/FLUX.1-schnell-Free",
            prompt=prompt,
        )
        # Get image URL from response
        image_url = response.data[0].url
        
        # Load and return the image
        response = requests.get(image_url)
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Failed to generate image: {str(e)}")
        return None

def generate_story(image_url: str, topic: str):
    """Generate a story using Llama model"""
    try:
        prompt = f"Look at this image: {image_url}. Write a short story about it related to the topic: {topic}."
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Failed to generate story: {str(e)}")
        return None

# Main app
st.title("🎨 AI Story Generator")
st.write("Generate an image and story from your topic!")

# Get user input
topic = st.text_input("What's your story about?", placeholder="e.g., A cat playing the piano")

# Generate button
if st.button("Generate", type="primary"):
    if topic:
        with st.spinner("Creating your story..."):
            # Generate and display image
            image = generate_image(f"An image related to {topic}")
            if image:
                st.image(image, caption="Generated Image")
                
                # Generate and display story
                story = generate_story(image, topic)
                if story:
                    st.write("### Your Story")
                    st.write(story)
    else:
        st.warning("Please enter a topic first!")
