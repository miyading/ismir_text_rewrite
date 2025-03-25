data_path = "analysis/RAG+Survey_March+8,+2025_14.25.csv"
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    # print(f"=== {dim.capitalize()} Score ~  Model ===")
    # print(result_test.summary())
    # print("\n")


def pivot_all_dimensions(df, dimensions):
    frames = []
    for dimension in dimensions:
        # Define the columns for each version
        cols = {
            "Novice": f"{dimension}_score_Novice",
            "LoRA": f"{dimension}_score_LoRA",
            "RAG": f"{dimension}_score_RAG",
        }
        # Subset the dataframe for ParticipantID, PromptID and the three columns
        sub_df = df[["ParticipantID", "PromptID"] + list(cols.values())].copy()
        # Melt the wide-format data into long-format
        long_df = sub_df.melt(
            id_vars=["ParticipantID", "PromptID"],
            value_vars=list(cols.values()),
            var_name="VersionCol",
            value_name="Score"
        )
        # Extract the version name from the column name (e.g., "expertness_score_Novice" → "Novice")
        long_df["Version"] = long_df["VersionCol"].apply(lambda x: x.split("_")[-1])
        # Add the dimension as a new column
        long_df["Dimension"] = dimension
        # Drop the helper column
        frames.append(long_df.drop(columns=["VersionCol"]))
    return pd.concat(frames, ignore_index=True)

def plot_strip_point_collapsed_facetgrid(df_long, save_path):

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", n_colors=3)
    
    # Specify the desired facet order.
    facet_order = ["expertness", "musicality", "production", "preference"]
    df_long["Dimension"] = pd.Categorical(df_long["Dimension"], categories=facet_order, ordered=True)
    
    # Create a FacetGrid with one facet per Dimension.
    g = sns.FacetGrid(df_long, col="Dimension", col_order=facet_order, col_wrap=2, sharey=True, height=5)
    
    # For each facet, plot the collapsed scores (aggregated by Version).
    for ax, (dim, sub_df) in zip(g.axes.flatten(), df_long.groupby("Dimension")):
        # Plot the stripplot (using Version as x, Score as y, and Version for color).
        sns.stripplot(
            data=sub_df,
            x="Version", y="Score", hue="Version",dodge=False,
            jitter=True, alpha=0.25, zorder=1, palette=palette, size=8,
            ax=ax, legend=False
        )
        # Calculate dodge spacing based on the number of Versions.
        
        # Overlay the pointplot to show conditional means.
        sns.pointplot(
            data=sub_df,
            x="Version", y="Score", hue="Version",
            dodge=False, palette=palette, errorbar=None,
            markers="d", markersize=6, linestyle="none",
            ax=ax, legend=False
        )
        ax.set_title(f"{dim.capitalize()}")
    
    # Set y-axis limits (for scores ranging from 1 to 3) and axis labels.
    g.set(ylim=(0.9, 3.1))
    g.set_axis_labels("Version", "Score")
    g.figure.suptitle("Survey Scores across Rewrite Versions", fontsize=16)
    g.figure.subplots_adjust(top=0.9, left =0.1)
    
    # Create one combined legend for all facets.
    versions = ["Novice", "LoRA", "RAG"]
    # palette = sns.color_palette("tab10", n_colors=len(versions))
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=version,
               markerfacecolor=palette[i], markersize=10)
        for i, version in enumerate(versions)
    ]
    
    # Add the combined legend to the overall figure.
    legend = g.figure.legend(handles=legend_handles, labels=versions,
                 loc="upper right", ncol=3, frameon=True, columnspacing=1, handletextpad=0)
    legend.set_title("Version")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_strip_point_by_dimension(df_long, save_path):

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", n_colors=6)
     # desired order for the Dimension facets
    facet_order = ["expertness", "musicality", "production", "preference"]
    
    # Ensure that the Dimension column is ordered accordingly
    df_long["Dimension"] = pd.Categorical(df_long["Dimension"], categories=facet_order, ordered=True)
    
    # Create a FacetGrid with one facet per dimension
    g = sns.FacetGrid(df_long, col="Dimension", col_order=facet_order, col_wrap=2, sharey=True, height=5)
    
    # Iterate over each facet
    for ax, (dim, sub_df) in zip(g.axes.flatten(), df_long.groupby("Dimension")):
        # Draw the stripplot using the original style
        sns.stripplot(
            data=sub_df,
            x="Version", y="Score", hue="PromptID",
            dodge=True, jitter=True, alpha=0.25, zorder=1, size=8,
            ax=ax, legend= False,
            palette=palette
        )
        # Overlay the pointplot to show conditional means
        # (Using global df_long to compute the number of unique PromptIDs)
        dodge_val = 0.8 - 0.8 / df_long["PromptID"].nunique()
        sns.pointplot(
            data=sub_df,
            x="Version", y="Score", hue="PromptID",
            dodge=dodge_val, errorbar=None,
            markers="d", markersize=6, linestyle="none",
            ax=ax, legend=False, 
            palette=palette
        )
        # Adjust legend placement for this facet (if a legend exists)
        # sns.move_legend(ax, loc="lower right", ncol=3, frameon=True, columnspacing=1, handletextpad=0)
        ax.set_title(f"{dim.capitalize()}")
    
    g.set(ylim=(0.9, 3.1))
    g.set_axis_labels("Version", "Score")
    g.figure.suptitle("Survey Scores by PromptID across Rewrite Versions", fontsize=16)
    g.figure.subplots_adjust(top=0.9, left = 0.1)
    ids = ["1", "2", "3", "4", "5", "6"]
    # palette = sns.color_palette("tab10", n_colors=len(ids))
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=promptid,
               markerfacecolor=palette[i], markersize=10)
        for i, promptid in enumerate(ids)
    ]
    
    legend = g.figure.legend(handles=legend_handles, labels=ids,
                 loc="upper right", ncol=3, frameon=True, columnspacing=1, handletextpad=0)
    legend.set_title("PromptID")
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_strip_point_by_promptID_facetgrid(df_long, save_path):

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", n_colors=3)
    # desired order for the Dimension facets
    facet_order = ["expertness", "musicality", "production", "preference"]
    
    # Ensure that the Dimension column is ordered accordingly
    df_long["Dimension"] = pd.Categorical(df_long["Dimension"], categories=facet_order, ordered=True)
    
    # Create a FacetGrid with one facet per Dimension
    g = sns.FacetGrid(df_long, col="Dimension", col_order=facet_order, col_wrap=2, sharey=True, height=5)
    
    # Iterate over each facet (each Dimension)
    for ax, (dim, sub_df) in zip(g.axes.flatten(), df_long.groupby("Dimension")):
        # Draw the stripplot with the original styling:
        sns.stripplot(
            data=sub_df,
            x="PromptID", 
            y="Score", 
            hue="Version",
            dodge=True, 
            jitter=True, 
            alpha=0.25, 
            zorder=1, 
            # palette="tab10", 
            size=8,
            ax=ax, legend=False, palette=palette
        )
        
        # Compute dodge value based on the number of unique Versions
        dodge_val = 0.8 - 0.8 / sub_df["Version"].nunique()
        
        # Overlay the pointplot to show conditional means
        sns.pointplot(
            data=sub_df,
            x="PromptID", 
            y="Score", 
            hue="Version",
            dodge=dodge_val,
            # palette="dark", 
            errorbar=None, 
            markers="d", 
            markersize=6, 
            linestyle="none", 
            ax=ax,
            legend=False, 
            palette=palette
        )
        ax.set_title(f"{dim.capitalize()}")
    
    # Set y-axis limits to suit scores in the 1 to 3 range
    g.set(ylim=(0.9, 3.1))
    g.set_axis_labels("PromptID", "Score")
    g.figure.suptitle("Survey Scores by Rewrite Version across PromptIDs", fontsize=16)
    g.figure.subplots_adjust(top=0.9, left=0.1)

    versions = ["Novice", "LoRA", "RAG"]
    # palette = sns.color_palette("tab10", n_colors=len(versions))
    
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=version,
               markerfacecolor=palette[i], markersize=10)
        for i, version in enumerate(versions)
    ]
    legend = g.figure.legend(handles=legend_handles, labels=versions,
                 loc="upper right", ncol=3, frameon=True, columnspacing=1, handletextpad=0)
    legend.set_title("Version")
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
dimensions = ["expertness", "musicality", "production", "preference"]
df_long_all = pivot_all_dimensions(df, dimensions)

# ensure Version is categorical with desired ordering:
df_long_all["Version"] = df_long_all["Version"].astype("category")
df_long_all["Version"] = df_long_all["Version"].cat.set_categories(["Novice", "LoRA", "RAG"], ordered=True)

# Plot the faceted strip-and-point plot:
plot_strip_point_collapsed_facetgrid(df_long_all, "analysis/figures/survey_collapsed_plot.png")
plot_strip_point_by_dimension(df_long_all,"analysis/figures/survey_by_version_plot.png")
plot_strip_point_by_promptID_facetgrid(df_long_all,"analysis/figures/survey_by_PromptID_plot.png")