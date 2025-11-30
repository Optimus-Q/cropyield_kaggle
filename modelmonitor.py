import re
import streamlit as st
import pandas as pd

pattern = (
    r"^(.*?) - INFO - Optuna Training Summary \| Start: (.*?) \| End: (.*?) "
    r"\| Duration: (.*?) seconds \| OPTUNA RMSE: (.*?) \| TEST RMSE: (.*?) "
    r"\| Model: (.*)$"
)

records = []
with open("./train/training logs/train.log", "r") as f:
    for line in f:
        match = re.match(pattern, line.strip())
        if match:
            records.append(match.groups())

df = pd.DataFrame(records, columns=[
    "log_time",
    "optuna_start",
    "optuna_end",
    "duration_sec",
    "optuna_rmse",
    "test_rmse",
    "model_name"
])

# Convert numerical fields
df["duration_sec"] = df["duration_sec"].astype(float)
df["optuna_rmse"] = df["optuna_rmse"].astype(float)
df["test_rmse"] = df["test_rmse"].astype(float)

st.title("Crop ID Training Model Monitoring")
st.dataframe(df)