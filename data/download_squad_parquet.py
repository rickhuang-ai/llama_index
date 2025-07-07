import datasets
import os

# Download SQuAD v2.0 using Hugging Face Datasets
squad = datasets.load_dataset("squad_v2")

# Output directory
out_dir = "./data/squad_v2_parquet"
os.makedirs(out_dir, exist_ok=True)

# Save train split as Parquet
train_path = os.path.join(out_dir, "train.parquet")
squad["train"].to_parquet(train_path)
print(f"Saved train split to {train_path}")

# Save validation split as Parquet
val_path = os.path.join(out_dir, "validation.parquet")
squad["validation"].to_parquet(val_path)
print(f"Saved validation split to {val_path}")
