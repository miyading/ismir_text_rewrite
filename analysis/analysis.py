data_path = "analysis/RAG+Survey_March+6,+2025_21.01.csv"
import pandas as pd
data = pd.read_csv(data_path)
# print(data.head())
print(data["Q2_3"][:5])