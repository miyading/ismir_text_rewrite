import os
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Set up argument parser
parser = argparse.ArgumentParser(description="Refine a music generation prompt using OpenAI's API.")
parser.add_argument("json_filename", type=str, help="The name of the JSON file containing the original prompt and results.")
args = parser.parse_args()

# Extract base filename (without extension) for use as a dictionary key
base_filename = os.path.splitext(os.path.basename(args.json_filename))[0]

# Load input JSON file
with open(args.json_filename, 'r') as f:
    json_data = json.load(f)

# Extracting original prompt and keywords
original_prompt = json_data["original_prompt"]
keywords = set()
for result in json_data["results"]:
    keywords.update(result["keywords"])

# Construct the refined prompt
keywords_list = ", ".join(keywords)

# OpenAI API call
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are an expert musician refining a text prompt for a music generation model."},
        {"role": "user", "content": f"Succinctly rewrite the following prompt '{original_prompt}' with these newly added keywords: {keywords_list}"}
    ]
)

# Get the refined prompt
refined_prompt = response.choices[0].message.content.strip()

# Save the output to rag_rewrites.json
output_filename = "rag_rewrites.json"
if os.path.exists(output_filename):
    with open(output_filename, 'r') as f:
        output_data = json.load(f)
else:
    output_data = {}

# Update dictionary with new refined prompt
output_data[base_filename] = refined_prompt

# Save updated JSON
with open(output_filename, 'w') as f:
    json.dump(output_data, f, indent=4)

# Print confirmation
print(f"Refined Prompt: {refined_prompt} for {base_filename} saved in {output_filename}.")