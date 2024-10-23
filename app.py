import streamlit as st
import time
from typing import Tuple
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

#############################################
# Configuration and Setup
#############################################

def initialize_together_client() -> OpenAI:
    """
    Initialize the Together AI client with API key from Streamlit secrets.
    
    Returns:
        OpenAI: Configured client for making API calls
    """
    try:
        return OpenAI(
            api_key=st.secrets["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1"
        )
    except Exception as e:
        st.error("Failed to initialize API client. Check if your API key is properly set in secrets.toml")
        raise e

#############################################
# Error Handling and Rate Limiting
#############################################

def safe_api_call(func, max_retries: int = 3, base_delay: int = 2):
    """
    Wrapper function to handle API rate limits and retries.
    
    Args:
        func: The API function to call
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries (increases with each attempt)
    
    Returns:
        The result of the API call or None if all retries fail
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            # Handle rate limiting (HTTP 429 error)
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)
                st.warning(f"Rate limit hit. Waiting {delay} seconds before retry #{attempt + 2}...")
                time.sleep(delay)
                continue
            # If it's not a rate limit error or we're out of retries, raise the exception
            if attempt == max_retries - 1:
                st.error(f"Failed after {max_retries} attempts. Error: {str(e)}")
            raise e
    return None

#############################################
# Image Generation
#############################################

def generate_image(client: OpenAI, topic: str) -> str:
    """
    Generate an AI image based on the given topic using FLUX.1-schnell-Free model.
    
    Args:
        client: OpenAI client instance
        topic: User's input topic for image generation
    
    Returns:
        str: URL of the generated image
    """
    # Create a detailed prompt for better image generation
    image_prompt = f"""Create a vivid and detailed image about: {topic}. 
    Make it colorful and engaging, suitable for storytelling."""
    
    def make_request():
        response = client.images.generate(
            model="black-forest-labs/FLUX.1-schnell-Free",
            prompt=image_prompt,
        )
        return response.data[0].url
    
    return safe_api_call(make_request)

#############################################
# Story Generation
#############################################

def generate_story(client: OpenAI, image_url: str, topic: str) -> str:
    """
    Generate a story based on the image and topic using Llama-Vision-Free model.
    
    Args:
        client: OpenAI client instance
        image_url: URL of the generated image
        topic: User's input topic for story generation
    
    Returns:
        str: Generated story text
    """
    # Craft a detailed prompt for story generation
    prompt = f"""Look at this image: {image_url}. 
    Write an engaging and descriptive short story (about 100 words) related to the topic: {topic}.
    Include these elements in your story:
    - Vivid descriptions of what's in the image
    - An interesting beginning, middle, and end
    - Emotional elements to engage the reader
    - Clear connection to the given topic
    Make the story captivating and suitable for all ages."""
    
    def make_request():
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    return safe_api_call(make_request)

#############################################
# Main Application Logic
#############################################

def create_story_app(client: OpenAI, topic: str) -> Tuple[str, str]:
    """
    Coordinate the creation of both image and story.
    
    Args:
        client: OpenAI client instance
        topic: User's input topic
    
    Returns:
        Tuple containing image URL and generated story
    """
    # Step 1: Generate the image
    st.info("Step 1/2: Generating image...")
    image_url = generate_image(client, topic)
    if not image_url:
        raise Exception("Image generation failed")
    
    # Small delay to ensure image processing is complete
    time.sleep(2)
    
    # Step 2: Generate the story
    st.info("Step 2/2: Creating story...")
    story = generate_story(client, image_url, topic)
    if not story:
        raise Exception("Story generation failed")
    
    return image_url, story

#############################################
# User Interface
#############################################

def display_story(image_url: str, story: str):
    """
    Display the generated content in a clean, two-column layout.
    
    Args:
        image_url: URL of the generated image
        story: Generated story text
    """
    col1, col2 = st.columns([1, 1])
    
    with col1:
        try:
            # Display the generated image
            response = requests.get(image_url, timeout=10)  # Added timeout
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Generated Image", use_column_width=True)
        except Exception as e:
            st.error("⚠️ Image display failed. Here's the URL:")
            st.code(image_url)
            st.error(f"Error details: {str(e)}")
    
    with col2:
        # Display the generated story
        st.markdown("### Your Story")
        st.write(story)

def main():
    """Main application entry point with improved error handling and user feedback"""
    # Configure the Streamlit page
    st.set_page_config(
        page_title="AI Story Generator",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Application header
    st.title("🎨 AI Story Generator")
    st.write("Generate a unique story with an AI-generated image based on your topic!")

    # User input section
    topic = st.text_input(
        "Enter a topic for your story:",
        placeholder="e.g., A magical forest at sunset",
        help="Be specific and descriptive for better results!"
    )

    # Generate button and processing
    if st.button("🪄 Generate Story") and topic:
        try:
            # Initialize API client
            client = initialize_together_client()
            
            # Show progress while generating
            with st.spinner('🎨 Creating your story and image... Please wait...'):
                image_url, story = create_story_app(client, topic)
            
            # Success message and display results
            st.success("✨ Your story has been created!")
            st.markdown("---")
            display_story(image_url, story)
                
        except Exception as e:
            # Enhanced error handling with troubleshooting tips
            st.error("❌ Something went wrong!")
            st.error(f"Error details: {str(e)}")
            
            st.markdown("### 🔧 Troubleshooting Tips:")
            st.write("1. Check if your API key is correctly set in secrets.toml")
            st.write("2. Try again in a few minutes if the service is busy")
            st.write("3. Make sure your topic is clear and appropriate")
            st.write("4. Check your internet connection")
    
    # Footer
    st.markdown("---")
    st.markdown("Created with ❤️ By BuildFastWithAI")

if __name__ == "__main__":
    main()
