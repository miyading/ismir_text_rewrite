data_path = "analysis/RAG+Survey_March+8,+2025_14.25.csv"
import pandas as pd
df = pd.read_csv(data_path, skiprows=[1,2], usecols=["Q0","Q1_1",'Q1_2',"Q1_3","Q2_1","Q2_2","Q2_3","Q3_1","Q3_2","Q3_3","Q4_1","Q4_2","Q4_3","Q5","Q6","Q7"])
# print(data.head())

print("Expertness RAG")
print(df["Q1_3"].value_counts())

# print(pd.crosstab(data["Q1_3"], data["Q7"])) # Expertness RAG ranked first by Prompt ID

print("Musicality RAG")
print(df["Q2_3"].value_counts())

print("Production Quality RAG")
print(df["Q3_3"].value_counts())

print("Preference RAG")
print(df["Q4_3"].value_counts())

#paried t-test for pairwise comparison, same p-value as linear mixed-effect model with random intercept. 
import pandas as pd
from scipy.stats import ttest_rel
# Use a custom function that maps rank r to 4 - r
def rank_to_score(rank):
    return 4 - rank   # so rank=1->3, rank=2->2, rank=3->1

def convert_ranks_to_scores(df, dimension, question_prefix):
    """
    For a given dimension (e.g., "expertness") and question prefix (e.g., "Q1"),
    create new columns in `df` that store the mapped scores
    for RAG, LoRA, and Novice.
    """
    # Example: question_prefix="Q1" => columns are Q1_3 (RAG), Q1_2 (LoRA), Q1_1 (Novice)
    rag_col = f"{question_prefix}_3"
    lora_col = f"{question_prefix}_2"
    novice_col = f"{question_prefix}_1"

    df[f"{dimension}_score_RAG"] = df[rag_col].apply(rank_to_score)
    df[f"{dimension}_score_LoRA"] = df[lora_col].apply(rank_to_score)
    df[f"{dimension}_score_Novice"] = df[novice_col].apply(rank_to_score)

def run_paired_ttest(df, dimension, group_a, group_b, alternative="greater"):
    """
    Runs a paired t-test on columns <dimension>_score_<group_a>
    vs. <dimension>_score_<group_b>, then prints the results.
    """
    col_a = f"{dimension}_score_{group_a}"
    col_b = f"{dimension}_score_{group_b}"

    stat, p_val = ttest_rel(df[col_a], df[col_b], alternative=alternative)
    print(f"{dimension.capitalize()} Paired t-test {group_a} vs. {group_b} (alternative='{alternative}'):")
    print(f"  t-statistic: {stat:.3f}")
    print(f"  p-value:     {p_val:.3f}\n")


# 2. Dictionary mapping dimension name -> question prefix
dimension_map = {
    "expertness": "Q1",
    "musicality": "Q2",
    "production": "Q3",
    "preference": "Q4"
}

# 3. Convert ranks to scores for each dimension
for dim, q_prefix in dimension_map.items():
    convert_ranks_to_scores(df, dim, q_prefix)

# 4. Run paired t-tests comparing RAG vs. Novice and RAG vs. LoRA
#    (You could also add LoRA vs. Novice if desired)

dimensions_to_test = ["expertness", "musicality", "production", "preference"]
for dim in dimensions_to_test:
    # RAG vs. Novice
    run_paired_ttest(df, dim, "RAG", "Novice", alternative="greater")

    # RAG vs. LoRA
    run_paired_ttest(df, dim, "RAG", "LoRA", alternative="greater")


import statsmodels.formula.api as smf


# Linear Mixed Effect Model Random Intercept with PromptID as random variable
# Long format 
df = df.rename(columns={'Q6': 'ParticipantID', 'Q7': 'PromptID'})
def pivot_one_dimension(df, dimension):
    """
    Given a wide DataFrame `df` with columns:
      <dimension>_score_Novice,
      <dimension>_score_LoRA,
      <dimension>_score_RAG,
    plus ParticipantID and PromptID,
    return a long DataFrame with columns:
      ParticipantID, PromptID, Version, Score
    where Version is one of {"Novice", "LoRA", "RAG"}.
    """
    # Identify the wide columns
    novice_col = f"{dimension}_score_Novice"
    lora_col   = f"{dimension}_score_LoRA"
    rag_col    = f"{dimension}_score_RAG"

    # Melt them into a single column for Score
    # a mini-DataFrame with just the needed columns
    sub_df = df[["ParticipantID", "PromptID", novice_col, lora_col, rag_col]].copy()

    # Use pd.melt
    long_df = sub_df.melt(
        id_vars=["ParticipantID", "PromptID"],
        value_vars=[novice_col, lora_col, rag_col],
        var_name="VersionCol",
        value_name="Score"
    )

    # Map the VersionCol (e.g., "expertness_score_Novice") -> "Novice"
    def get_version(name):
        # name might be "expertness_score_Novice" => last part is "Novice"
        return name.split("_")[-1]  # "Novice", "LoRA", or "RAG"

    long_df["Version"] = long_df["VersionCol"].apply(get_version)
    long_df.drop(columns=["VersionCol"], inplace=True)

    return long_df

def fit_mixed_model_promptID(df_long):
    """
    Given a long DataFrame with columns:
      ParticipantID, PromptID, Version, Score
    fit a linear mixed-effects model:
      Score ~ Version  +  (1 | PromptID).
    Returns the fitted model result.
    """
    # Treat Version as a categorical variable
    # Can specify whichever category as baseline (reference)
    model = smf.mixedlm(
        formula="Score ~ C(Version)",
        data=df_long,
        groups=df_long["ParticipantID"]
    )
    result = model.fit(method='lbfgs')
    return result

def fit_mixed_model_promptID_participantID(df_long):
    """
    Given a long DataFrame with columns:
      ParticipantID, PromptID, Version, Score
    fit a linear mixed-effects model:
      Score ~ Version * PromptID +  (1 | ParticipantID).
    Returns the fitted model result.
    """
    # Treat Version as a categorical variable
    # Can specify whichever category as baseline (reference)
    model = smf.mixedlm("Score ~ C(Version)*C(PromptID)", data=df_long, groups=df_long["ParticipantID"])
    result = model.fit(method='lbfgs')
    return result

# 2) Dimensions to analyze
dimensions = ["expertness", "musicality", "production", "preference"]

for dim in dimensions:
    # A) Pivot the data for one dimension
    df_long = pivot_one_dimension(df, dim)

    # B) Fit a linear mixed model: Score ~ Version + (1|PromptID)
    result = fit_mixed_model_promptID(df_long)
    # Fit a linear mixed model: Score ~ Version * PromptID + (1|PariticipantID)
    result_participantID = fit_mixed_model_promptID_participantID(df_long)

    # C) Print results
    print(f"=== {dim.capitalize()} Model ===")
    print(result.summary())
    print("\n")

    print(f"=== {dim.capitalize()} Participant as Random Effect Model ===")
    print(result_participantID.summary())
    print("\n")


# df_long has columns: ParticipantID, PromptID, Version, Score
# and want to ignore (not model) the fact that multiple Prompts/Participants are repeated.

# model_ols = smf.ols("Score ~ C(Version)", data=df_long).fit()
# print(model_ols.summary())



