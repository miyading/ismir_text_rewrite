from openai import OpenAI
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
with open("selections_with_prompt.json", 'r') as f:
    json_data = json.load(f)

# Extracting original prompt and keywords
original_prompt = json_data["original_prompt"]
keywords = set()
for result in json_data["results"]:
    keywords.update(result["keywords"])

# Construct the refined prompt
keywords_list = ", ".join(keywords)
# refined_prompt = f"{original_prompt}, with additional keywords: {keywords_list}."

# OpenAI API call
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an expert musician refining a text prompt for a music generation model."},
        {"role": "user", "content": f"Succinctly rewrite the following prompt '{original_prompt}' with these newly added keywords: {keywords_list}"}
    ]
)

# Print the refined prompt
print("Refined Prompt:", response.choices[0].message.content.strip())