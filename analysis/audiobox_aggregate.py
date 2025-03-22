import json
import os
import numpy as np
from collections import defaultdict

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
for result in aggregated_results:
    print(json.dumps(result, indent=4))


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
print(df)

# Melt the data into long format for Seaborn
df_long = df.melt(id_vars=["Version", "PromptID"], value_vars=["CU", "PC", "PQ", "CE"], var_name="Metric", value_name="Score")
print(df_long.dtypes)
df_long["Version"] = df_long["Version"].astype("category")
df_long["PromptID"] = df_long["PromptID"].astype("category")
df_long["Version"] = df_long["Version"].cat.set_categories(["Novice", "LoRA", "RAG"], ordered=True)
df_long["PromptID"] = df_long["PromptID"].cat.set_categories(["1", "2", "3", "4", "5", "6"], ordered=True)
df_long["Score"] = pd.to_numeric(df_long["Score"])

def plot_metrics_by_promptid_facetgrid(df_long):
    sns.set_theme(style="whitegrid")

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
        sharey=True  # use a consistent y-axis across subplots
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
            legend=False
        )
        ax.set_title(f"{metric} by PromptID")

    # Adjust y-axis and overall labels:
    g.set(ylim=(2, 10))
    g.set_axis_labels("PromptID", "Score")
    g.figure.suptitle("Score Distributions by PromptID (per Metric)", fontsize=16)
    g.figure.subplots_adjust(top=0.9)

    # Display the plot
    plt.show()

# Call the function with your long-format DataFrame:
plot_metrics_by_promptid_facetgrid(df_long)

# marginalize promptID, Version across four score dimensions in facetgrid
def plot_metrics_facetgrid(df_long):
    sns.set_theme(style="whitegrid")

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
        sharey=True  # consistent Y-axis 
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
            legend=False
        )
        ax.set_title(f"{metric} by Version")

    # Set Y-axis range
    g.set(ylim=(2, 10))
    g.set_axis_labels("Version", "Score")
    g.figure.suptitle("Score Distributions by Version (per Metric)", fontsize=16)
    g.figure.subplots_adjust(top=0.9)

    plt.show()

plot_metrics_facetgrid(df_long)