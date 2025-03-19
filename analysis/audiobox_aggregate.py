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