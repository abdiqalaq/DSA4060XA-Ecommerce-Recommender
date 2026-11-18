
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_url
import requests

# Dataset information
repo = "McAuley-Lab/Amazon-Reviews-2023"
filename = "benchmark/5core/rating_only/Electronics.csv"

# Create output directory
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

print("Connecting to Amazon Electronics dataset...", flush=True)

url = hf_hub_url(
    repo_id=repo,
    filename=filename,
    repo_type="dataset"
)

# Stream a sample without saving the full file
response = requests.get(
    url,
    stream=True,
    timeout=120
)
response.raise_for_status()

records = []
max_rows = 100000

try:
    lines = response.iter_lines(decode_unicode=True)

    # Read CSV header
    header = next(lines)
    print("CSV header:", header)

    import csv
    from io import StringIO

    reader = csv.reader(lines)

    for row in reader:
        records.append(row)

        if len(records) >= max_rows:
            break
finally:
    response.close()

df = pd.DataFrame(records)

file_path = output_dir / "electronics_reviews_sample.csv"
df.to_csv(file_path, index=False)

print("\nDataset saved successfully!")
print("Total sampled rows:", len(df))
print("Columns:", df.shape[1])
print(df.head())
