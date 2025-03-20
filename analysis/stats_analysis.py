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
# for dim in dimensions_to_test:
#     # RAG vs. Novice
#     run_paired_ttest(df, dim, "RAG", "Novice", alternative="greater")

#     # RAG vs. LoRA
#     run_paired_ttest(df, dim, "RAG", "LoRA", alternative="greater")


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

    print(f"=== {dim.capitalize()} Score ~ Version * PromptID + (1|PariticipantID) Model ===")
    print(result_participantID.summary())
    print("\n")

    # print(f"=== {dim.capitalize()} Score ~  Model ===")
    # print(result_test.summary())
    # print("\n")
import seaborn as sns
import matplotlib.pyplot as plt

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
    palette="dark", errorbar=None, markers="d", markersize=6, linestyle="none"
)

# Improve the legend
sns.move_legend(
    ax, loc="lower right", ncol=3, frameon=True, columnspacing=1, handletextpad=0,
)

# Title and labels
plt.title("Jittered Strip Plot with Mean Scores by Version (Colored by PromptID)")
plt.xlabel("Version")
plt.ylabel("Score")

plt.show()

# Jittered Stript Plot for Meta Audiobox metric
data = [
    {"path": "audios/RAG4", "Version": "RAG", "PromptID": "4", "CE": 7.757, "CU": 7.888, "PC": 6.003, "PQ": 8.011},
    {"path": "audios/RAG5", "Version": "RAG", "PromptID": "5", "CE": 7.071, "CU": 7.78, "PC": 5.42, "PQ": 7.782},
    {"path": "audios/RAG6", "Version": "RAG", "PromptID": "6", "CE": 7.708, "CU": 7.86, "PC": 6.526, "PQ": 8.025},
    {"path": "audios/RAG1", "Version": "RAG", "PromptID": "1", "CE": 7.576, "CU": 7.887, "PC": 6.088, "PQ": 8.072},
    {"path": "audios/RAG2", "Version": "RAG", "PromptID": "2", "CE": 7.306, "CU": 8.052, "PC": 4.178, "PQ": 8.042},
    {"path": "audios/RAG3", "Version": "RAG", "PromptID": "3", "CE": 7.59, "CU": 7.829, "PC": 6.693, "PQ": 8.01},
    {"path": "audios/LoRA1", "Version": "LoRA", "PromptID": "1", "CE": 7.443, "CU": 8.039, "PC": 5.375, "PQ": 7.724},
    {"path": "audios/LoRA2", "Version": "LoRA", "PromptID": "2", "CE": 6.902, "CU": 8.138, "PC": 2.637, "PQ": 7.809},
    {"path": "audios/LoRA3", "Version": "LoRA", "PromptID": "3", "CE": 7.305, "CU": 8.213, "PC": 5.262, "PQ": 8.314},
    {"path": "audios/LoRA4", "Version": "LoRA", "PromptID": "4", "CE": 7.655, "CU": 7.816, "PC": 5.112, "PQ": 8.127},
    {"path": "audios/LoRA5", "Version": "LoRA", "PromptID": "5", "CE": 6.857, "CU": 7.472, "PC": 6.242, "PQ": 7.505},
    {"path": "audios/LoRA6", "Version": "LoRA", "PromptID": "6", "CE": 8.168, "CU": 8.209, "PC": 6.513, "PQ": 8.028},
    {"path": "audios/Novice1", "Version": "Novice", "PromptID": "1", "CE": 7.342, "CU": 7.221, "PC": 6.759, "PQ": 7.523},
    {"path": "audios/Novice2", "Version": "Novice", "PromptID": "2", "CE": 7.082, "CU": 8.217, "PC": 2.609, "PQ": 8.004},
    {"path": "audios/Novice3", "Version": "Novice", "PromptID": "3", "CE": 7.068, "CU": 7.244, "PC": 6.757, "PQ": 7.863},
    {"path": "audios/Novice4", "Version": "Novice", "PromptID": "4", "CE": 8.082, "CU": 8.002, "PC": 5.429, "PQ": 8.252},
    {"path": "audios/Novice5", "Version": "Novice", "PromptID": "5", "CE": 2.622, "CU": 2.826, "PC": 5.894, "PQ": 3.848},
    {"path": "audios/Novice6", "Version": "Novice", "PromptID": "6", "CE": 7.123, "CU": 6.431, "PC": 6.176, "PQ": 7.186},
]


df = pd.DataFrame(data)

# Convert PromptID to string for categorical processing
# df["PromptID"] = df["PromptID"].astype(str)

# Melt the data into long format for Seaborn
df_long = df.melt(id_vars=["Version", "PromptID"], value_vars=["CE", "CU", "PC", "PQ"], var_name="Metric", value_name="Score")
print(df_long.dtypes)
df_long["Score"] = pd.to_numeric(df_long["Score"])
# Plot
sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

# Strip plot with jitter to avoid overlap
sns.stripplot(
    data=df_long, x="Version", y="Score", hue="PromptID",
    dodge=True, jitter=True, palette="tab10", alpha=0.6, size=8
)

# Overlay a point plot to show conditional means per Version
sns.pointplot(
    data=df_long, x="Version", y="Score", hue="PromptID",
    dodge=0.8 - 0.8 / df_long["PromptID"].nunique(),
    palette="dark", errorbar=None, markers="d", markersize=6, linestyle="none"
)

# Adjust legend
plt.legend(title="PromptID", bbox_to_anchor=(1, 1), loc='upper left')

plt.title("Meta Audiobox Metrics by Version and PromptID")
plt.xlabel("Version")
plt.ylabel("Score")

plt.show()