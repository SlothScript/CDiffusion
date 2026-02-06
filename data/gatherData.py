import gzip
import shutil
import os
from datasets import load_dataset

TARGET_SIZE_GB = 1.0
TARGET_BYTES = int(TARGET_SIZE_GB * 1024**3)

print("Loading Wikipedia dataset...")
dataset = load_dataset(
    "wikipedia",
    "20220301.en",
    split="train",
    streaming=True
)

current_size = 0
pages_written = 0

with open("corpus.txt", "w", buffering=1024 * 1024) as f:
    for article in dataset:
        text = article["text"]  # type: ignore

        if len(text) < 500:
            continue

        f.write(text)
        f.write("\n")

        current_size += len(text)
        pages_written += 1

        if current_size >= TARGET_BYTES:
            break

        if pages_written % 1000 == 0:
            size_mb = current_size / (1024 * 1024)
            percent = (current_size / TARGET_BYTES) * 100
            print(
                f"Written {pages_written} articles\t\t"
                f"{size_mb:.1f} MB\t"
                f"{percent:.2f}%"
            )

final_gb = current_size / (1024**3)
print(f"Total pages: {pages_written}\t\tSize: {final_gb:.3f} GB")


# Compress
print("\nCompressing...")

src = "corpus.txt"
dst = "corpus.txt.gz"

with open(src, "rb") as f_in, gzip.open(dst, "wb", compresslevel=1) as f_out:
    shutil.copyfileobj(f_in, f_out, length=1024 * 1024)

original_size = os.path.getsize(src)
compressed_size = os.path.getsize(dst)

percent_decrease = (1 - (compressed_size / original_size)) * 100

print(f"Compression reduced size by {percent_decrease:.2f}% ({compressed_size / (1024 * 1024 * 1024):.2f} GB)")

os.remove(src)