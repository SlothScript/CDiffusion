import re
import gzip
import shutil
import os
from pathlib import Path
from multiprocessing import Pool

# Use DATA_DIR environment variable or default to /data
data_dir = Path(os.environ.get("DATA_DIR", "/data"))

# If /data is not writable, fall back to local data directory
if not os.access(data_dir, os.W_OK):
    data_dir = Path("data")

INPUT_FILE = data_dir / "corpus.txt.gz"
OUTPUT_FILE = data_dir / "corpus_clean.txt"
BATCH_SIZE = 1000
NUM_WORKERS = 10

def clean_text(text: str) -> str:
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = text.replace("\\", "")
    text = re.sub(r'\s{2,}', ' ', text)
    text = ''.join(ch for ch in text if ch.isprintable())
    return text.strip()

def process_batch(lines):
    return [clean_text(line) + "\n" for line in lines if line.strip()]


def main():
    print("Reading file...")
    with gzip.open(INPUT_FILE, "rt", encoding="utf-8", errors="ignore") as fin:
        all_lines = fin.readlines()

    total_lines = len(all_lines)
    print(f"Total lines: {total_lines}")

    batches = [
        all_lines[i:i + BATCH_SIZE]
        for i in range(0, total_lines, BATCH_SIZE)
    ]

    print("Processing...")
    with Pool(NUM_WORKERS) as pool, open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for i, results in enumerate(pool.imap(process_batch, batches)):
            fout.writelines(results)
            percent = ((i + 1) * BATCH_SIZE / total_lines) * 100
            print(f"\rProgress: {min(percent, 100):.1f}%", end="", flush=True)

    print("\nDone.")

    # Compress cleaned output
    print("\nCompressing...")
    src = OUTPUT_FILE
    dst = Path(str(OUTPUT_FILE) + ".gz")

    with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
        shutil.copyfileobj(f_in, f_out, length=1024 * 1024)

    original_size = os.path.getsize(src)
    compressed_size = os.path.getsize(dst)

    percent_decrease = (1 - (compressed_size / original_size)) * 100
    print(
        f"Compression reduced size by {percent_decrease:.2f}% "
        f"({compressed_size / (1024**3):.2f} GB)"
    )

    os.remove(src)


if __name__ == "__main__":
    main()