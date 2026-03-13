import pandas as pd
import os

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'City': ['New York', 'Los Angeles', 'Chicago']}
new_row_loc = { 'Name': 'David', 'Age': 40, 'City': 'Houston'}

df = pd.DataFrame(data)
df.loc[len(df)] = new_row_loc
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, 'sample_data.csv')
df.to_csv(file_path, index=False)
print(f"DataFrame saved to {file_path}")
