import streamlit as st
import sys
import os
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_retrieval import retrieve_prompts

musiccaps = pd.read_csv("data/musiccaps.csv")

# Initialize session state to store selections and retrieved prompts
if "selections" not in st.session_state:
    st.session_state.selections = {"original_prompt": "", "results": []}
if "retrieved_prompts" not in st.session_state:
    st.session_state.retrieved_prompts = None
if "input_prompt" not in st.session_state:
    st.session_state.input_prompt = ""

def render_matches(matches, section_title):
    """
    Render the Top K Matches (Text or Audio) with unified selection capabilities.
    """
    st.subheader(section_title)

    for idx, (score, ytid, prompt, keywords) in enumerate(matches):
        # Display prompt details
        # https://youtu.be/ibTVNWeEPF4?t=454
        start_s = musiccaps.loc[musiccaps['ytid'] == ytid,'start_s'] 
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
                }
            )


def main():
    st.title("Retrieve Top K Similar Prompts and Select Keywords")

    # Input text prompt
    new_text_prompt = st.text_input("Enter your text prompt:", st.session_state.input_prompt)

    # Store the input prompt in session state
    if new_text_prompt:
        st.session_state.input_prompt = new_text_prompt

    if st.button("Retrieve Prompts"):
        if not st.session_state.input_prompt.strip():
            st.warning("Please enter a text prompt!")
        else:
            # Retrieve top k matches
            top_k_text_matches, top_k_audio_matches = retrieve_prompts(st.session_state.input_prompt, k=5)
            st.session_state.retrieved_prompts = (top_k_text_matches, top_k_audio_matches)
            # Save the original novice input prompt in session state
            st.session_state.selections["original_prompt"] = st.session_state.input_prompt
            st.session_state.selections["results"] = []  # Clear previous results

    # Display retrieved prompts if available
    if st.session_state.retrieved_prompts:
        top_k_text_matches, top_k_audio_matches = st.session_state.retrieved_prompts

        # Render Top K Text Matches
        render_matches(top_k_text_matches, "Top K Text Matches")

        # Render Top K Audio Matches
        render_matches(top_k_audio_matches, "Top K Audio Matches")

    # Export selections
    if st.button("Export Selections"):
        if not st.session_state.selections["results"]:
            st.warning("No selections to export!")
        else:
            import json
            with open("selections_with_prompt.json", "w") as f:
                json.dump(st.session_state.selections, f, indent=4)
            st.success("Selections exported to 'selections_with_prompt.json'!")


if __name__ == "__main__":
    main()
