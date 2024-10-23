import streamlit as st
import json
import time
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO

def initialize_together_client() -> OpenAI:
    return OpenAI(
        api_key=st.secrets["TOGETHER_API_KEY"],
        base_url="https://api.together.xyz/v1"
    )

def generate_story_content(client: OpenAI, topic: str) -> dict:
    """Generate story line, image prompt, and story in a single API call"""
    prompt = {
        "role": "user",
        "content": f"""Given the topic: {topic}
        Create a story package with the following elements:
        1. A story line (one sentence summary)
        2. An image prompt that captures the key visual elements
        3. A vivid, descriptive short story (100 words) based on the story line
        
        Return only a JSON object with this structure:
        {{
            "story_line": "one sentence summary",
            "image_prompt": "detailed visual description for image generation",
            "story": "100-word story"
        }}"""
    }
    
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[prompt]
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                delay = (attempt + 1) * 2
                st.warning(f"Rate limit hit. Waiting {delay} seconds...")
                time.sleep(delay)
                continue
            raise e
    return None

def generate_image(client: OpenAI, prompt: str) -> str:
    """Generate image with retry logic"""
    for attempt in range(3):
        try:
            response = client.images.generate(
                model="black-forest-labs/FLUX.1-schnell-Free",
                prompt=prompt,
            )
            return response.data[0].url
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                delay = (attempt + 1) * 2
                st.warning(f"Rate limit hit. Waiting {delay} seconds...")
                time.sleep(delay)
                continue
            raise e
    return None

def display_story(image_url: str, story_line: str, story: str):
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
        st.write("**Story Line:**")
        st.write(story_line)
        st.write("**Story:**")
        st.write(story)

def main():
    st.set_page_config(page_title="AI Story Generator", layout="wide")
    
    st.title("AI Story Generator")
    st.write("Generate a unique story with an AI-generated image based on your topic!")

    topic = st.text_input("Enter a topic for your story:", 
                         placeholder="e.g., A magical forest where animals play musical instruments")

    if st.button("Generate Story") and topic:
        try:
            client = initialize_together_client()
            
            with st.spinner('Generating your story and image... This may take a minute.'):
                # Generate all story content in one API call
                content = generate_story_content(client, topic)
                if not content:
                    st.error("Failed to generate story content")
                    return
                
                # Generate image
                image_url = generate_image(client, content["image_prompt"])
                if not image_url:
                    st.error("Failed to generate image")
                    return
            
            st.success("Story generated successfully!")
            st.markdown("---")
            display_story(image_url, content["story_line"], content["story"])
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("\nTroubleshooting tips:")
            st.write("1. Check if the secrets.toml file is properly configured")
            st.write("2. Try again in a few minutes if the service is busy")
    
    st.markdown("---")
    st.markdown("Created with ❤️ By BuildFastWithAI")

if __name__ == "__main__":
    main()
