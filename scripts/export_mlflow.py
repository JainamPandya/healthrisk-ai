import mlflow
import os
import pandas as pd

os.makedirs("reports", exist_ok=True)
df = mlflow.search_runs(experiment_names=["HealthRisk_Threshold_Tuning"])
if isinstance(df, pd.DataFrame):
    df.to_csv("reports/mlflow_runs.csv", index=False)
    print("Exported MLflow runs to reports/mlflow_runs.csv")
