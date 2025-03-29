
<center><h2>Enhancing Text-to-Music Generation through Retrieval-Augmented Prompt Rewrite</h2></center>

[**Dataset**](data) | [**Paper**]()

This repository contains the code and dataset used for the paper  
**"Enhancing Text-to-Music Generation through Retrieval-Augmented Prompt Rewrite."**

<p align="center">
  <img src="analysis/figures/Overview.png" alt="Overview" width="700"/>
</p>

This figure shows two novice-to-expert prompt rewrite methods:

1. **RAG** – A retrieval-augmented generation method that uses CLAP-based similarity to retrieve the top-$k=3$ most relevant audio captions. Participants select keywords (highlighted in blue) to guide GPT-3.5 in generating a custom expert-level prompt.

2. **LoRA** – A fine-tuned model for prompt rewriting.

---

## Quick Start

### 1. Environment Setup

Create and activate a conda environment:

```bash
conda create -n ismir_text_rewrite python=3.10
conda activate ismir_text_rewrite
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App

Launch the app locally:

```bash
streamlit run scripts/streamlit_app.py
```

---

## Play with the Streamlit App

1. **Enter a Random Number**

   This is used as an anonymous identifier instead of your real name.

2. **Select a Novice Prompt**

   Choose one of the six novice prompts provided to begin the evaluation.

3. **Choose Relevant Keywords**

   For both the top-k text and audio matches, select the keywords you want to guide the expert prompt rewrite.

<p align="center">
  <img src="analysis/figures/streamlit_app.png" alt="Streamlit App" width="400"/>
</p>

---

## Survey Questions

Please respond to the following questions based on your evaluation experience:

1. **How familiar are you with the current genre under evaluation?**
   - No Experience
   - Little Experience
   - Moderate Experience
   - Extensive Experience

2. **Which version of the generated music sounds most like it was composed by an expert musician?**
   - Novice Generation  
   - LoRA Generation  
   - Retrieval-Augmented Generation

3. **Which version sounds the most musical (appropriate use of instruments, genre alignment, mood/emotion conveyance)?**
   - Novice Generation  
   - LoRA Generation  
   - Retrieval-Augmented Generation

4. **Which version sounds the most professional in terms of production quality (clarity, mixing, balance)?**
   - Novice Generation  
   - LoRA Generation  
   - Retrieval-Augmented Generation

5. **Which version of the music do you prefer overall?**
   - Novice Generation  
   - LoRA Generation  
   - Retrieval-Augmented Generation

6. **Did you notice any inconsistencies between the generated music and the corresponding text prompt?**
   - Novice Generation  
   - LoRA Generation  
   - Retrieval-Augmented Generation  
   - No noticeable issues (N/A)

7. **What is the random number generated at the beginning of your experiment?**  
   *(For anonymous tracking only; not used for identification.)*

8. **What is the ID of the current prompt under evaluation?**

---
## Novice vs. LoRA v.s. RAG Prompts by Genre

| Prompt ID | Genre     | YouTube ID       | Novice Prompt                                                                                              | LoRA Rewrite | RAG Rewrite |
|-----------|-----------|------------------|-------------------------------------------------------------------------------------------------------------|--------------|-------------|
| 1         | R&B       | -6pcgdLfb_A      | R&B, male singer, string, strong bass, drums, suited for an intimate setting                               | A male singer performs an R&B song. This song is medium tempo with a string section and a strong bass line. The drums are playing a swing rhythm. The song is loud and ambient. The song is a bit noisy. This song is playing in a small room. | Create an R&B track with a male singer and female vocal, featuring a wide acoustic rhythm guitar melody, groovy bass line, drums, and string for an intimate setting. |
| 2         | Classical | -OUIEnuNd1I      | Calming classical music similar to Bach with harp                                                          | This is a soft and calming classical music. This could be the result of a harpist playing a melody in the style of Baroque music by Bach. | Create tranquil classical music reminiscent of Bach, featuring harp improvisation with delicate plucking. |
| 3         | Pop       | -rAhS48FkYw      | joyful pop song with passionate male vocal with shiny drum set sounds and wooden percussion                | A male singer performs an emotional and passionate pop song. The song starts with a wooden percussion and a shiny cymbal sound. The song then changes to an electric drum set. The first half of the song has a simple repetitive melody. The second half of the song is more complex and technologically advanced with various instruments and sounds. The overall mood of the song is joyful. | Create an upbeat pop track featuring a passionate male lead vocals, punchy guitar rhythm, punchy drumming, shiny drum set sounds, wooden percussion, synth backing, and background male vocals. |
| 4         | Soul      | 07xGXxIHOL4      | soul, romantic love song with lush saxophone, drum and piano accompaniment                                 | This is a soul song. The song is about romantic love and features a saxophone playing the melody. The song has a romantic quality to it. The song is a soul song and it features the song of a piano, drums and a bass playing the accompaniment. There is no other instrumentation in this song. | Craft a passionate and romantic love song featuring lush saxophone, piano, and drums, perfect for a relaxing live performance. Allow for improvisations and envision it as an ideal proposal song. |
| 5         | Indie     | 0H3FAoDgzhI      | indie song, female vocal, synth bass, industrial sound, medium fast                                        | This low quality audio contains an indie song that consists of a medium two bar female vocal, singing over synth bass, electric guitar chord, shimmering hi hats, crashing cymbals and ambient industrial sounds. | Create an indie song with echoing passionate female vocals, soft or mellow female vocals, a synth bass and industrial sound, complemented by a piano accompaniment. The tempo should be medium-fast, imbuing a sad and deep ambiance enhanced by reverb effects. |
| 6         | Jazz      | 8zcogfmAD_o      | classic New Orleans jazz, vintage swing feel, jazz orchestra with charming female talking intro           | This jazz clip features a live performance of a classic New Orleans jazz piece. The piece starts with a vintage jazz clarinet solo. The piece then shifts to an orchestra jazz piece. The first half of the piece has a vintage swing feel with a jazz orchestra playing the melody. There is a female voice in the background talking. The piece has a charming vibe. | Fast tempo jazz orchestra performance with vintage swing feel, featuring trombones, cheerful female talking intro, groovy bass lines, and upbeat energy. |
