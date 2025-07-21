import os
import pandas as pd


csv_directory = 'D:\AI\MachineLearningOriginal'

csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

dataframes = []

for file in csv_files:
    path = os.path.join(csv_directory, file)
    try:
        df = pd.read_csv(path, low_memory=False)
        dataframes.append(df)
        print(f"{file} loaded, {df.shape[0]} entries.")
    except Exception as e:
        print(f"Error at {file}: {e}")

df_sum = pd.concat(dataframes, ignore_index=True)

df_sum.to_csv('CICIDS2017_original.csv', index=False)
print(f"Fertig! Gesamtgröße: {df_sum.shape}")
