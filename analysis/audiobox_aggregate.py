import json
import os
import numpy as np
from collections import defaultdict
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf

def aggregate_scores(file_path):
    data = defaultdict(lambda: {"CE": [], "CU": [], "PC": [], "PQ": [], "paths": []})
    unaltered_entries = []

    # Read the jsonl file
    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            path = entry["path"]
            file_name = os.path.basename(path)

            # Check if the filename follows the "_x" pattern
            if "_" in file_name:
                prompt_id_x = file_name.split("_")[-1].split(".")[0]  # Extract _x
                key = f"audios/RAG{prompt_id_x}"  # Group under RAG_x

                data[key]["CE"].append(entry["CE"])
                data[key]["CU"].append(entry["CU"])
                data[key]["PC"].append(entry["PC"])
                data[key]["PQ"].append(entry["PQ"])
                data[key]["paths"].append(path)  # Store original paths
            else:
                # Keep unaltered entries
                unaltered_entries.append({key: round(value, 3) 
                                          if key in {"CE", "CU", "PC", "PQ"} else value.replace(".wav", "") if key == "path" else value for key, value in entry.items()})

    # Compute mean and variance for grouped data
    results = []
    for key, values in data.items():
        result = {
            "path": key,
            "CE_mean": round(np.mean(values["CE"]),3),
            "CE_var": round(np.var(values["CE"]),3),
            "CU_mean": round(np.mean(values["CU"]),3),
            "CU_var": round(np.var(values["CU"]),3),
            "PC_mean": round(np.mean(values["PC"]),3),
            "PC_var": round(np.var(values["PC"]),3),
            "PQ_mean": round(np.mean(values["PQ"]),3),
            "PQ_var": round(np.var(values["PQ"]),3),
        }
        results.append(result)

    # Combine grouped results with unaltered entries
    final_results = results + unaltered_entries

    return final_results

file_path = "analysis/combined.jsonl" 
aggregated_results = aggregate_scores(file_path)

# Print the aggregated results
# for result in aggregated_results:
#     print(json.dumps(result, indent=4))


# Jittered Stript Plot for Meta Audiobox metric
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

rows = []
with open(file_path, "r") as f:
    for line in f:
        entry = json.loads(line)
        path = entry["path"]               # e.g. "audios/973807_2.wav" for RAG 12 audios per prompt or "audios/LoRA1.wav"
        filename = os.path.basename(path)
        
        if "_" in filename:
            # => RAG format: something like "973807_2.wav"
            version = "RAG"
            prompt_id = filename.split("_")[-1].replace(".wav", "")
        elif filename.startswith("LoRA"):
            # => LoRA format: something like "LoRA1.wav"
            version = "LoRA"
            prompt_id = filename.replace("LoRA", "").replace(".wav", "")
        elif filename.startswith("Novice"):
            # => Novice format: something like "Novice1.wav"
            version = "Novice"
            prompt_id = filename.replace("Novice", "").replace(".wav", "")
        
        # Build a dict for one row
        row_data = {
            "Version": version,
            "PromptID": prompt_id,
            "CE": entry["CE"],
            "CU": entry["CU"],
            "PC": entry["PC"],
            "PQ": entry["PQ"],
        }
        rows.append(row_data)

# Convert to a DataFrame
df = pd.DataFrame(rows)
# print(df)

# Melt the data into long format for Seaborn
df_long = df.melt(id_vars=["Version", "PromptID"], value_vars=["CU", "PC", "PQ", "CE"], var_name="Metric", value_name="Score")
# print(df_long.dtypes)
df_long["Version"] = df_long["Version"].astype("category")
df_long["PromptID"] = df_long["PromptID"].astype("category")
df_long["Version"] = df_long["Version"].cat.set_categories(["Novice", "LoRA", "RAG"], ordered=True)
df_long["PromptID"] = df_long["PromptID"].cat.set_categories(["1", "2", "3", "4", "5", "6"], ordered=True)
df_long["Score"] = pd.to_numeric(df_long["Score"])

def plot_metrics_by_promptid_facetgrid(df_long, save_path):
    sns.set_theme(style="whitegrid")
    # Use a colorblind friendly palette for the boxplots
    palette = sns.color_palette("colorblind", n_colors=3)
    # Initialize the catplot with boxplot as base:
    g = sns.catplot(
        data=df_long,
        x="PromptID", y="Score",
        col="Metric", col_wrap=2,
        hue="Version",
        kind="box",
        height=5, aspect=1,
        showcaps=True,
        boxprops={"alpha": 0.6},
        whiskerprops={"linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 2},
        dodge=True,
        sharey=True,  # use a consistent y-axis across subplots
        legend=False, 
        palette=palette
    )

    # Overlay stripplots on each subplot:
    for ax, metric in zip(g.axes.flatten(), df_long["Metric"].unique()):
        sub_df = df_long[df_long["Metric"] == metric]
        sns.stripplot(
            data=sub_df,
            x="PromptID",
            y="Score",
            hue="Version",
            dodge=True,
            jitter=True,
            alpha=0.6,
            size=6,
            ax=ax,
            legend=False, 
            palette=palette
        )
        ax.set_title(f"{metric}")

    # Adjust y-axis and overall labels:
    g.set(ylim=(2, 10))
    g.set_axis_labels("PromptID", "Score")
    g.figure.suptitle("Audiobox Scores by Rewrite Version across PromptIDs", fontsize=16)
    g.figure.subplots_adjust(top=0.9)
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
    # Display the plot
    plt.show()

# Pass the long-format DataFrame:
plot_metrics_by_promptid_facetgrid(df_long, "analysis/figures/audiobox_by_PromptID_plot")

# marginalize promptID, Version across four score dimensions in facetgrid
def plot_metrics_facetgrid(df_long, save_path):
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", n_colors=3)
    # Initialize the catplot with boxplot as base
    g = sns.catplot(
        data=df_long,
        x="Version", y="Score",
        col="Metric", col_wrap=2,
        hue = "Version",
        kind="box",
        height=5, aspect=1,
        showcaps=True,
        boxprops={"alpha": 0.6},
        whiskerprops={"linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 2},
        dodge=True,
        sharey=True,  # consistent Y-axis 
        legend=False,
        palette=palette
    )

    # Overlay stripplots on each subplot
    for ax, metric in zip(g.axes.flatten(), df_long["Metric"].unique()):
        sub_df = df_long[df_long["Metric"] == metric]
        sns.stripplot(
            data=sub_df,
            x="Version",
            y="Score",
            hue="Version",
            dodge=True,
            jitter=True,
            alpha=0.6,
            size=6,
            ax=ax,
            legend = False,
            palette=palette
        )
        ax.set_title(f"{metric}")

    # Set Y-axis range
    g.set(ylim=(2, 10))
    g.set_axis_labels("Version", "Score")
    g.figure.suptitle("Audiobox Scores across Rewrite Versions", fontsize=16)
    g.figure.subplots_adjust(top=0.9)
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

plot_metrics_facetgrid(df_long, "analysis/figures/audiobox_collapsed_plot")
# "In this plot, the diamond markers represent the means, while the jittered circular points represent the individual responses from each participant.""
def plot_metrics_by_version_then_promptid_facetgrid(df_long, save_path):

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("colorblind", n_colors=6)
    
    # Create a catplot with x as Version (aggregated) and facet by Metric.
    g = sns.catplot(
        data=df_long,
        x="Version", y="Score",
        col="Metric", col_wrap=2,
        hue="PromptID",  # Within each version, promptIDs are distinguished by color.
        kind="box",  # Base plot as boxplot for summary statistics.
        height=5, aspect=1,
        showcaps=True,
        boxprops={"alpha": 0.6},
        whiskerprops={"linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 2},
        dodge=True,
        sharey=True,
        legend=False,
        palette=palette
    )
    
    # Overlay a stripplot to show individual data points.
    for ax, metric in zip(g.axes.flatten(), df_long["Metric"].unique()):
        sub_df = df_long[df_long["Metric"] == metric]
        sns.stripplot(
            data=sub_df,
            x="Version", y="Score",
            hue="PromptID", 
            dodge=True,
            jitter=True,
            alpha=0.6,
            size=6,
            ax=ax,
            legend=False,
            palette=palette
        )
        ax.set_title(f"{metric}")
    
    # Adjust y-axis and overall labels.
    g.set(ylim=(2, 10))
    g.set_axis_labels("Version", "Score")
    g.figure.suptitle("Audiobox Scores by PromptID across Rewrite Versions", fontsize=16)
    g.figure.subplots_adjust(top=0.9)
    
    # Create one combined legend for the PromptID groups using dot markers.
    prompt_ids = sorted(df_long["PromptID"].unique())  # Assumes six unique prompt IDs.
    # palette = sns.color_palette("tab10", n_colors=len(prompt_ids))
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', label=pid,
               markerfacecolor=palette[i], markersize=10)
        for i, pid in enumerate(prompt_ids)
    ]
    
    legend = g.figure.legend(
        handles=legend_handles,
        labels=prompt_ids,
        loc="upper right",
        ncol=3,
        frameon=True,
        columnspacing=1,
        handletextpad=0
    )
    legend.set_title("PromptID")
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
plot_metrics_by_version_then_promptid_facetgrid(df_long, "analysis/figures/audiobox_by_version_plot")


# Linear Model 
unique_metrics = ["CE", "CU", "PC", "PQ"]
results = {}
for metric in unique_metrics:
    # Subset the data for the current metric
    subset = df_long[df_long["Metric"] == metric]
    
    # Fit the linear mixed-effects model
    # Here, 'Score ~ Version' defines the fixed effects.
    # 'groups=subset["PromptID"]' specifies that PromptID is the random effect grouping variable.
    model = smf.mixedlm("Score ~ Version", subset, groups=subset["PromptID"])
    result = model.fit()
    
    results[metric] = result
    print(f"Results for metric: {metric}")
    print(result.summary())
    print("\n")

# Example randomness of the Diffusion process Prompt 5 Indie 
data = pd.read_json('analysis/diffusion_variation.jsonl', lines=True)
df = pd.DataFrame(data)

# Extract version from the path assuming the version is the first token in the second part of the path
df['version'] = df['path'].apply(lambda x: x.split('/')[1].split(' ')[0])

# Set the custom order for versions
version_order = ['Novice', 'LoRA', 'RAG']
df['version'] = pd.Categorical(df['version'], categories=version_order, ordered=True)

# Now when sort by version or plot, they will follow the order "Novice", "LoRA", "RAG"
versions = df['version'].cat.categories.tolist()

# Define score types
score_types = ['CU', 'PC', 'PQ', 'CE']

# Create boxplots for each score grouped by the custom version order
fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
for i, score in enumerate(score_types):
    data_to_plot = [df[df['version'] == v][score] for v in version_order]
    axes[i].boxplot(data_to_plot, labels=version_order)
    axes[i].set_title(score)
    axes[i].set_xlabel('Version')
    if i == 0:
        axes[i].set_ylabel('Score')

plt.suptitle('Randomness of Diffusion Process PromptID:5 Scores across Rewrite Versions', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.style.use('tableau-colorblind10')
plt.show()