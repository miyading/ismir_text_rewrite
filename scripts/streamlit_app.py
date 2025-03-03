import streamlit as st
import sys
import os
import json
import random
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_retrieval import retrieve_prompts

# Load dataset
musiccaps = pd.read_csv("data/musiccaps.csv")

# Define mapping for prompt categories
prompt_mapping = {
    "R&B, male singer, string, strong bass, drums, suited for an intimate setting": "1",
    "Calming classical music similar to Bach with harp": "2",
    "joyful pop song with passionate male vocal with shiny drum set sounds and wooden percussion": "3",
    "soul, romantic love song with lush saxophone, drum and piano accompaniment": "4",
    "indie song, female vocal, synth bass, industrial sound, medium fast": "5",
    "classic New Orleans jazz, vintage swing feel, jazz orchestra with charming female talking intro": "6"
}

# Initialize session state variables
if "participant_id" not in st.session_state:
    st.session_state.participant_id = ""
if "novice_prompt" not in st.session_state:
    st.session_state.novice_prompt = "" 
if "selections" not in st.session_state:
    st.session_state.selections = {"original_prompt": "", "results": []}
if "retrieved_prompts" not in st.session_state:
    st.session_state.retrieved_prompts = None

# Streamlit UI Components
st.title("Retrieval and Keyword Selection Interface: ")
st.subheader("Enhancing Music Generation with Retrieval-Augmented Prompt Rewrite")
st.markdown('Imagine you are a novice user of a music generation system, you would like to transform the given novice text prompt into a more descriptive text prompt that would help to generate more expert-level music. Please select relevant keywords that would aid this novice-to-expert rewrite.')

# **1. Participant ID Input**
participant_id = st.text_input("Please enter the random number generated:", st.session_state.participant_id)
if participant_id:
    st.session_state.participant_id = participant_id

# **2. Novice Prompt Selection (Dropdown)**
st.session_state.novice_prompt = st.selectbox("Select a novice prompt:", list(prompt_mapping.keys()))

# **3. Retrieve Prompts Button**
if st.button("Retrieve Prompts"):
    if not st.session_state.novice_prompt:
        st.warning("Please select a novice prompt!")
    else:
        # Retrieve top k matches
        top_k_text_matches, top_k_audio_matches = retrieve_prompts(st.session_state.novice_prompt, k=3)
        st.session_state.retrieved_prompts = (top_k_text_matches, top_k_audio_matches)

        # Save the original novice prompt in session state
        st.session_state.selections["original_prompt"] = st.session_state.novice_prompt
        st.session_state.selections["results"] = []  # Clear previous selections

# Function to render matches
def render_matches(matches, section_title, modality):
    st.subheader(section_title)

    for idx, (score, ytid, prompt, keywords) in enumerate(matches):
        # Get YouTube start timestamp
        start_s = musiccaps.loc[musiccaps['ytid'] == ytid, 'start_s'].values[0]
        st.markdown(f"**Prompt {idx + 1}:** {prompt} (Link: https://www.youtube.com/watch?v={ytid}&t={start_s}, Score: {score:.4f})")

        # Get default selections for this specific prompt
        default_entry = next(
            (entry for entry in st.session_state.selections["results"] if entry["prompt"] == prompt and entry["ytid"] == ytid),
            None,
        )
        default_keywords = default_entry["keywords"] if default_entry else []

        # Create a multiselect for selecting keywords
        selected_keywords = st.multiselect(
            f"Select relevant keywords for {section_title[:-2]} {idx + 1}:",
            options=eval(keywords),  # Convert keywords string to a list
            default=default_keywords,
            key=f"{section_title}-{idx}",
        )

        # Save selections to session state
        if selected_keywords != default_keywords:
            # Remove old entry for this prompt and YouTube ID
            st.session_state.selections["results"] = [
                entry
                for entry in st.session_state.selections["results"]
                if not (entry["prompt"] == prompt and entry["ytid"] == ytid)
            ]
            # Add updated entry
            st.session_state.selections["results"].append(
                {
                    "ytid": ytid,
                    "prompt": prompt,
                    "keywords": selected_keywords,
                    "modality": modality,
                }
            )

# Display retrieved prompts if available
if st.session_state.retrieved_prompts:
    top_k_text_matches, top_k_audio_matches = st.session_state.retrieved_prompts
    
    # Render Top K Text Matches
    render_matches(top_k_text_matches, "Top K Text Matches", modality= "text")

    # Render Top K Audio Matches
    render_matches(top_k_audio_matches, "Top K Audio Matches", modality = "audio")

# **4. Export Selections Button**
if st.button("Export Selections"):
    if not st.session_state.selections["results"]:
        st.warning("No selections to export!")
    else:
        # Determine participant ID (random if not provided)
        participant_id = st.session_state.participant_id.strip()

        # Get prompt category ID
        novice_prompt = st.session_state.novice_prompt
        prompt_id = prompt_mapping.get(novice_prompt, "unknown")

        # Construct filename: {participant_id}_{prompt_id}.json
        filename = f"rewrites/{participant_id}_{prompt_id}.json"

        # Save selections to JSON file
        with open(filename, "w") as f:
            json.dump(st.session_state.selections, f, indent=4)

        st.success(f"Selections exported to '{filename}'!")