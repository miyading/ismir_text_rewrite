import pandas as pd
import subprocess
import sys

file_path = "/Users/huangjuhua/Desktop/NYU/1011_NLP/NLP FInal Project/NLP Final Project Code/data/musiccaps-updated.csv"

df = pd.read_csv(file_path)

# Initialize an empty list to store expert-level prompts
expert_prompts = []

# Iterate through the novice descriptions
for novice_description in df['novice']:
    # Command to run the fine-tuned model using mlx_lm.generate
    command = [
        'mlx_lm.generate',
        '--model', '/Users/huangjuhua/文档文稿/NYU/NLP_Final_Project/models/NEP2.0',
        '--system-prompt', """ou are a helpful assistant that converts novice-friendly music descriptions into expert descriptions.\n\n
        Transform the given input novice-level prompt into a prompt that a user with extensive music training and terminologies would use to prompt music generation models.\n\n
        Keep the instruments, genres, mood, and other information that represents the essence of the music.\n\n
        Write the output succinctly in a coherent sentence.""",
        '--prompt', novice_description,
        '--temp', '0.7',
        '--verbose', 'False'
    ]
    
    # Run the command and capture the output
    result = subprocess.run(command, capture_output=True, text=True)
    expert_prompt = result.stdout.strip()
    expert_prompts.append(expert_prompt)

# Add the expert-level prompts to the DataFrame
df['expert'] = expert_prompts


df.to_csv('/Users/huangjuhua/Desktop/NYU/1011_NLP/NLP FInal Project/NLP Final Project Code/data/paired_dataset_LoRA.csv', index=False)  # Replace with your actual output path