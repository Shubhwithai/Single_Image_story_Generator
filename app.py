import streamlit as st
import time
from typing import Tuple
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

def initialize_together_client() -> OpenAI:
    return OpenAI(
        api_key=st.secrets["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1"
    )

def safe_api_call(func):
    """Handle rate limits with simple retry logic"""
    max_retries = 3
    base_delay = 2
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                st.warning(f"Rate limit hit. Waiting {delay} seconds before retry...")
                time.sleep(delay)
                continue
            raise e
    return None

def generate_image(client: OpenAI, topic: str) -> str:
    """
    Generate an image using FLUX.1-schnell-Free model.
    
    Args:
        client: OpenAI client
        topic (str): The topic for image generation.
    
    Returns:
        str: URL of the generated image.
    """
    image_prompt = f"Create a vivid and detailed image about: {topic}"
    
    def make_request():
        response = client.images.generate(
            model="black-forest-labs/FLUX.1-schnell-Free",
            prompt=image_prompt,
        )
        return response.data[0].url
    
    return safe_api_call(make_request)

def generate_story(client: OpenAI, image_url: str, topic: str) -> str:
    """
    Generate a story using Llama-Vision-Free model based on an image and topic.
    
    Args:
        client: OpenAI client
        image_url (str): URL of the image to describe.
        topic (str): The topic for the story.
    
    Returns:
        str: Generated story.
    """
    prompt = f"""Look at this image: {image_url}. 
    Write an engaging and descriptive short story (about 100 words) related to the topic: {topic}.
    Make the story vivid and captivating, incorporating visual elements from the image."""
    
    def make_request():
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    return safe_api_call(make_request)

def create_story_app(client: OpenAI, topic: str) -> Tuple[str, str]:
    """
    Create a story app that generates an image and a story based on a given topic.
    
    Args:
        client: OpenAI client
        topic (str): The topic for the story and image.
    
    Returns:
        Tuple[str, str]: A tuple containing the image URL and the generated story.
    """
    # Generate image
    image_url = generate_image(client, topic)
    if not image_url:
        raise Exception("Failed to generate image")
    
    # Add a small delay to ensure the image is processed
    time.sleep(2)
    
    # Generate story based on the image
    story = generate_story(client, image_url, topic)
    if not story:
        raise Exception("Failed to generate story")
    
    return image_url, story

def display_story(image_url: str, story: str):
    """Display the generated image and story in a two-column layout"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        try:
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Generated Image", use_column_width=True)
        except Exception as e:
            st.error(f"Unable to display image. Error: {str(e)}")
            st.write(f"Image URL: {image_url}")
    
    with col2:
        st.write("**Generated Story:**")
        st.write(story)

def main():
    st.set_page_config(page_title="AI Story Generator", layout="wide")
    
    st.title("AI Story Generator")
    st.write("Generate a unique story with an AI-generated image based on your topic!")

    topic = st.text_input("Enter a topic for your story:", 
                         placeholder="e.g., A cat playing the piano")

    if st.button("Generate Story") and topic:
        try:
            client = initialize_together_client()
            
            with st.spinner('Generating your story and image... This may take a minute.'):
                image_url, story = create_story_app(client, topic)
            
            st.success("Story generated successfully!")
            st.markdown("---")
            display_story(image_url, story)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("\nTroubleshooting tips:")
            st.write("1. Check if the secrets.toml file is properly configured")
            st.write("2. Try again in a few minutes if the service is busy")
    
    st.markdown("---")
    st.markdown("Created with ❤️ By BuildFastWithAI")

if __name__ == "__main__":
    main()
