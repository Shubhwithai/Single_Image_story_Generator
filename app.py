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

def handle_api_call(func):
    max_retries = 3
    base_delay = 2  # Start with 2 seconds delay
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                delay = base_delay * (attempt + 1)  # Increase delay with each retry
                st.warning(f"Rate limit hit. Waiting {delay} seconds before retry...")
                time.sleep(delay)
                continue
            raise e
    return None

def generate_story_prompt(client: OpenAI, topic: str) -> dict:
    prompt = {
        "role": "user",
        "content": f"""Create a story line and image prompt about: {topic}
        Return only a JSON object with this structure:
        {{
            "story_line": "your story line here",
            "image_prompt": "your image prompt here"
        }}"""
    }
    
    def make_request():
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[prompt]
        )
        return json.loads(response.choices[0].message.content)
    
    return handle_api_call(make_request)

def generate_image(client: OpenAI, prompt: str) -> str:
    def make_request():
        response = client.images.generate(
            model="black-forest-labs/FLUX.1-schnell-Free",
            prompt=prompt,
        )
        return response.data[0].url
    
    return handle_api_call(make_request)

def generate_story(client: OpenAI, image_prompt: str, story_line: str) -> str:
    prompt = f"""Write a short story (100 words) that combines these elements:
    1. Scene description: {image_prompt}
    2. Story line: {story_line}
    Make the story vivid and descriptive, as if describing a scene from a painting."""

    def make_request():
        response = client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    return handle_api_call(make_request)

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
                # Generate story prompt and image prompt
                prompts = generate_story_prompt(client, topic)
                if not prompts:
                    st.error("Failed to generate story prompts after retries")
                    return
                
                # Generate image
                image_url = generate_image(client, prompts["image_prompt"])
                if not image_url:
                    st.error("Failed to generate image after retries")
                    return
                
                # Generate story
                story = generate_story(client, prompts["image_prompt"], prompts["story_line"])
                if not story:
                    st.error("Failed to generate story after retries")
                    return
            
            st.success("Story generated successfully!")
            st.markdown("---")
            display_story(image_url, prompts["story_line"], story)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("\nTroubleshooting tips:")
            st.write("1. Check if the secrets.toml file is properly configured")
            st.write("2. Try again in a few minutes if the service is busy")
    
    st.markdown("---")
    st.markdown("Created with ❤️ By BuildFastWithAI")


if __name__ == "__main__":
    main()
    
