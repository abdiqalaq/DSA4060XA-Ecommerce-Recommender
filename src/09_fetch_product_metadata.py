

from pathlib import Path
import csv
import json
import time
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEANED = ROOT / "data" / "cleaned"
RESULTS = ROOT / "results"

CLEANED.mkdir(parents=True, exist_ok=True)

OUTPUT = CLEANED / "product_metadata.csv"
CHECKPOINT = CLEANED / "metadata_checkpoint.json"

URL = (
    "https://huggingface.co/datasets/"
    "McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/"
    "meta_Electronics.jsonl"
)

COLUMNS = [
    "parent_asin",
    "title",
    "price",
    "average_rating",
    "image_url",
    "main_category",
]

SAVE_EVERY = 50
MAX_RETRIES = 10

# ------------------------------------------
# 1. Collect product IDs
# ------------------------------------------

print("Collecting product IDs...", flush=True)

files = [
    CLEANED / "electronics_cleaned.csv",
    RESULTS / "popularity_rankings.csv",
    RESULTS / "collaborative_recommendations.csv",
    RESULTS / "svd_recommendations.csv",
]

products = set()

for file in files:
    if file.exists():
        df = pd.read_csv(file, usecols=["parent_asin"])
        products.update(
            df["parent_asin"].dropna().astype(str)
        )

if not products:
    raise RuntimeError("No product IDs found.")

print(f"Looking for {len(products)} products...", flush=True)

# ------------------------------------------
# 2. Load previously saved metadata
# ------------------------------------------

matched = {}

if OUTPUT.exists():
    old = pd.read_csv(OUTPUT).fillna("")

    for row in old.to_dict("records"):
        asin = str(row["parent_asin"])
        if asin in products:
            matched[asin] = row

print(f"Previously saved: {len(matched)}", flush=True)

remaining = products - set(matched)

offset = 0
processed = 0

if CHECKPOINT.exists():
    checkpoint = json.loads(CHECKPOINT.read_text())
    offset = int(checkpoint.get("offset", 0))
    processed = int(checkpoint.get("processed", 0))

# ------------------------------------------
# 3. Save progress
# ------------------------------------------

def save_progress():
    temp = OUTPUT.with_suffix(".tmp")

    with temp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(matched.values())

    temp.replace(OUTPUT)

    checkpoint_data = {
        "offset": offset,
        "processed": processed,
    }

    temp_checkpoint = CHECKPOINT.with_suffix(".tmp")
    temp_checkpoint.write_text(
        json.dumps(checkpoint_data),
        encoding="utf-8"
    )
    temp_checkpoint.replace(CHECKPOINT)

    print(
        f"Saved {len(matched)} products | "
        f"Processed {processed:,} records",
        flush=True
    )

# ------------------------------------------
# 4. Extract product information
# ------------------------------------------

def extract_product(item):
    images = item.get("images") or []
    image_url = ""

    for img in images:
        if isinstance(img, dict):
            image_url = (
                img.get("large")
                or img.get("hi_res")
                or img.get("thumb")
                or ""
            )
            if image_url:
                break

    return {
        "parent_asin": str(item.get("parent_asin")),
        "title": item.get("title") or "",
        "price": item.get("price"),
        "average_rating": item.get("average_rating"),
        "image_url": image_url,
        "main_category": item.get("main_category") or "",
    }

# ------------------------------------------
# 5. Stream with retry and resume
# ------------------------------------------

retries = 0

print("Connecting to Amazon metadata...", flush=True)

while remaining and retries < MAX_RETRIES:
    headers = {"Accept-Encoding": "identity"}

    if offset:
        headers["Range"] = f"bytes={offset}-"

    try:
        with requests.get(
            URL,
            headers=headers,
            stream=True,
            timeout=(30, 120)
        ) as response:

            response.raise_for_status()

            if offset and response.status_code != 206:
                print(
                    "Server does not support resume. "
                    "Restarting scan from beginning.",
                    flush=True
                )
                offset = 0
                processed = 0

            print(
                f"Connected (HTTP {response.status_code})",
                flush=True
            )

            # Read raw bytes to track complete JSON lines
            buffer = b""

            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue

                buffer += chunk

                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)

                    offset += len(line) + 1
                    processed += 1

                    if not line.strip():
                        continue

                    try:
                        item = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue

                    asin = str(item.get("parent_asin") or "")

                    if asin not in remaining:
                        continue

                    matched[asin] = extract_product(item)
                    remaining.remove(asin)

                    if len(matched) % SAVE_EVERY == 0:
                        save_progress()

                    if not remaining:
                        break

                if not remaining:
                    break

                if processed % 100000 == 0:
                    print(
                        f"Scanned {processed:,} | "
                        f"Matched {len(matched)}",
                        flush=True
                    )

            # If connection ends normally, save remaining matches
            save_progress()

            if not remaining:
                break

            if not buffer:
                print("Reached end of metadata file.", flush=True)
                break

            # Incomplete final line: reconnect from its start
            raise requests.exceptions.ConnectionError(
                "Stream ended with an incomplete JSON line"
            )

    except (
        requests.exceptions.RequestException,
        ConnectionError
    ) as error:

        retries += 1
        save_progress()

        print(
            f"\nConnection interrupted: {error}",
            flush=True
        )

        if retries >= MAX_RETRIES:
            print("Maximum retries reached.", flush=True)
            break

        wait = min(retries * 5, 30)

        print(
            f"Retrying in {wait} seconds...",
            flush=True
        )

        time.sleep(wait)

# ------------------------------------------
# 6. Final summary
# ------------------------------------------

save_progress()

print("\n===== METADATA RESULTS =====")
print(f"Products requested: {len(products)}")
print(f"Products matched: {len(matched)}")
print(f"Products unmatched: {len(remaining)}")
print(f"Records scanned: {processed:,}")
print(f"Saved to: {OUTPUT}")

print("\nMetadata processing finished!")
