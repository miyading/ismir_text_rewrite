data_path = "analysis/RAG+Survey_March+8,+2025_14.25.csv"
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
#    (could also add LoRA vs. Novice if desired)
# df.to_csv('survey_scores.csv', index=False)
dimensions_to_test = ["expertness", "musicality", "production", "preference"]
for dim in dimensions_to_test:
    # RAG vs. Novice
    run_paired_ttest(df, dim, "RAG", "Novice", alternative="greater")

    # RAG vs. LoRA
    run_paired_ttest(df, dim, "RAG", "LoRA", alternative="greater")

    # LoRA vs. Novice
    run_paired_ttest(df, dim, "LoRA", "Novice", alternative="greater")

import statsmodels.formula.api as smf

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
    model = smf.mixedlm(
        formula="Score ~ C(Version)",
        data=df_long,
        groups=df_long["PromptID"]
    )
    result = model.fit(method='lbfgs')
    return result

def fit_fixed_effect_model(df_long):
    """
    Given a long DataFrame with columns:
      ParticipantID, PromptID, Version, Score
    fit a ordinary least square model:
      Score ~ Version * PromptID.
    Does not account for any random effects for ParticipantID.
    """
    model = smf.ols("Score ~ C(Version) * C(PromptID)", data=df_long).fit()
    return model

def fit_mixed_model_promptID_participantID(df_long):
    """
    Given a long DataFrame with columns:
      ParticipantID, PromptID, Version, Score
    fit the 3rd linear mixed-effects model as described in the paper:
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

def plot_strip_point_collapsed(df_long, dim):
    sns.set_theme(style="whitegrid")
    
    # Initialize figure
    f, ax = plt.subplots(figsize=(10, 6))
    sns.despine(bottom=True, left=True)  # Clean up plot edges

    # Strip plot (scatter with jitter for individual points)
    sns.stripplot(
        data=df_long, x="Version", y="Score", hue="Version",
        jitter=True, alpha=0.25, zorder=1, palette="tab10", size=8,legend = False
    )

    # Overlay a point plot to show conditional means
    sns.pointplot(
        data=df_long, x="Version", y="Score", hue="Version",
        palette="dark", errorbar=None, markers="d", markersize=6, linestyle="none", legend=False
    )

    # # Improve the legend
    # sns.move_legend(
    #     ax, loc="lower right", ncol=3, frameon=True, columnspacing=1, handletextpad=0
    # )

    # Title and labels
    plt.title(dim + " Strip Plot with Mean Scores by Version")
    plt.xlabel("Version")
    plt.ylabel("Score")
    
    plt.show()

def plot_strip_point(df_long, dim):
    sns.set_theme(style="whitegrid")
    
    # Initialize figure
    f, ax = plt.subplots(figsize=(10, 6))
    sns.despine(bottom=True, left=True)  # Clean up plot edges

    # Strip plot (scatter with jitter for individual points)
    sns.stripplot(
        data=df_long, x="Version", y="Score", hue="PromptID",
        dodge=True, jitter=True, alpha=0.25, zorder=1, palette="tab10", size=8
    )

    # Overlay a point plot to show conditional means
    sns.pointplot(
        data=df_long, x="Version", y="Score", hue="PromptID",
        dodge=0.8 - 0.8 / df_long["PromptID"].nunique(),  # Adjust spacing for hue
        palette="dark", errorbar=None, markers="d", markersize=6, linestyle="none", legend=False
    )

    # Improve the legend
    sns.move_legend(
        ax, loc="lower right", ncol=3, frameon=True, columnspacing=1, handletextpad=0
    )

    # Title and labels
    plt.title(dim + " Strip Plot with Mean Scores by Version (Colored by PromptID)")
    plt.xlabel("Version")
    plt.ylabel("Score")
    
    plt.show()

def plot_strip_point_by_promptID(df_long, dim):
    sns.set_theme(style="whitegrid")
    
    # Initialize figure
    f, ax = plt.subplots(figsize=(10, 6))
    sns.despine(bottom=True, left=True)  # Clean up plot edges

    # Strip plot (scatter with jitter for individual points)
    sns.stripplot(
        data=df_long, x="PromptID", y="Score", hue="Version",
        dodge=True, jitter=True, alpha=0.25, zorder=1, palette="tab10", size=8
    )

    # Overlay a point plot to show conditional means
    sns.pointplot(
        data=df_long, x="PromptID", y="Score", hue="Version",
        dodge=0.8 - 0.8 / df_long["Version"].nunique(),  # Adjust spacing for hue
        palette="dark", errorbar=None, markers="d", markersize=6, linestyle="none", legend = False
    )

    # Improve the legend
    sns.move_legend(
        ax, loc="lower right", ncol=3, frameon=True, columnspacing=1, handletextpad=0,
    )

    # Title and labels
    plt.title(dim + " Strip Plot with Mean Scores by PromptID (Colored by Version)")
    plt.xlabel("Version")
    plt.ylabel("Score")
    
    plt.show()

for dim in dimensions:
    # A) Pivot the data for one dimension
    df_long = pivot_one_dimension(df, dim)
    # Relevel so that Novice is always the reference when modeling
    df_long["Version"] = df_long["Version"].astype("category")
    df_long["Version"] = df_long["Version"].cat.set_categories(["Novice", "LoRA", "RAG"], ordered=True)

    # B) 
    # Fit an OLS model: Score ~ Version
    result = smf.ols("Score ~ C(Version)", data=df_long).fit()
    # # Fit an OLS model: Score ~ Version * PromptID
    result_promptID = fit_fixed_effect_model(df_long)
    # # Fit a linear mixed model: Score ~ Version * PromptID + (1|PariticipantID)
    result_participantID = fit_mixed_model_promptID_participantID(df_long)

    # result_test = fit_mixed_model_promptID(df_long)

    # # C) Print results
    print(f"=== {dim.capitalize()} Score ~ Version Model ===")
    print(result.summary())
    print("\n")

    print(f"=== {dim.capitalize()} Score ~ Version * PromptID Model ===")
    print(result_promptID.summary())
    print("\n")

    print(f"=== {dim.capitalize()} Score ~ Version * PromptID + (1|ParticipantID) Model ===")
    print(result_participantID.summary())
    print("\n")

    plot_strip_point(df_long, dim)
    # plot_strip_point_by_promptID(df_long, dim)
    # plot_strip_point_collapsed(df_long, dim)

    # print(f"=== {dim.capitalize()} Score ~  Model ===")
    # print(result_test.summary())
    # print("\n")
