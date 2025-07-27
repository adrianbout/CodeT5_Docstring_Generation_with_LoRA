import os
import zipfile
import json
import subprocess
import tarfile

import pandas as pd
import numpy as np
import glob
import re
from sklearn.model_selection import train_test_split # Used for sampling subsets
import os

# Login using ur account on kaggle and download this dataset / It should downlaod python.zip file /
# drag the python.zip file into a folder called data
# https://www.kaggle.com/datasets/omduggineni/codesearchnet?select=python

# Define the directory where you want to extract the contents
extract_dir = 'data'

# Define the path to the zip file
zip_file_path = 'data/python.zip'


try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"'{zip_file_path}' successfully unzipped to '{extract_dir}'")
except FileNotFoundError:
    print(f"Error: The file '{zip_file_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    

# --- Configuration ---
# Directory where the raw unzipped JSONL files are located
raw_data_base_path = "data/python/final/jsonl/"
# Directory where the processed (cleaned and split) JSONL files will be saved
processed_data_save_path = "data/processedData/"

# Ensure the processedData directory exists
os.makedirs(processed_data_save_path, exist_ok=True)

# --- NEW: Desired number of samples for the FINAL training set (after cleaning) ---
# Set this to your preferred training set size. E.g., 1000, 5000, 10000, etc.
DESIRED_TRAIN_SAMPLES = 2000 # You requested about 1000 rows

# Desired overall split percentages (e.g., 80% train, 10% valid, 10% test)
TRAIN_PERCENTAGE = 0.80
VALID_PERCENTAGE = 0.10
TEST_PERCENTAGE = 0.10

# --- Function to display dataset statistics ---
def show_stats(train_df, valid_df, test_df, title="📊 Dataset Statistics"):
    """Show simple statistics for the datasets."""
    print(f"\n{title}")
    print("=" * 40)

    datasets = [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]

    for name, df in datasets:
        if len(df) > 0:
            avg_doc_len = df['docstring'].str.len().mean()
            avg_code_len = df['code'].str.len().mean()

            print(f"{name}:")
            print(f"  📝 Samples: {len(df):,}")
            print(f"  📏 Avg docstring: {avg_doc_len:.0f} chars")
            print(f"  💻 Avg code: {avg_code_len:.0f} chars")
            print()
        else:
            print(f"{name}: No data loaded or sampled.")
            print()

# --- Data Loading Function (from raw unzipped files, but keeps them separate) ---
def load_and_combine_raw_per_split(base_path):
    """
    Loads all data from the raw unzipped JSONL files for each split (train, valid, test)
    and combines them into a single DataFrame for that split.
    """
    print(f"📥 Loading raw data from '{base_path}' for each split...")

    train_raw_df = pd.DataFrame()
    valid_raw_df = pd.DataFrame()
    test_raw_df = pd.DataFrame()

    # Load train data
    train_files = glob.glob(os.path.join(base_path, "train", "*.jsonl"))
    if train_files:
        train_raw_df = pd.concat([pd.read_json(f, lines=True) for f in train_files], ignore_index=True)
        print(f"  Loaded {len(train_raw_df):,} raw samples for Train.")
    else:
        print(f"🚨 Warning: No .jsonl files found for Train in '{os.path.join(base_path, 'train')}'.")

    # Load valid data
    valid_files = glob.glob(os.path.join(base_path, "valid", "*.jsonl"))
    if valid_files:
        valid_raw_df = pd.concat([pd.read_json(f, lines=True) for f in valid_files], ignore_index=True)
        print(f"  Loaded {len(valid_raw_df):,} raw samples for Valid.")
    else:
        print(f"🚨 Warning: No .jsonl files found for Valid in '{os.path.join(base_path, 'valid')}'.")

    # Load test data
    test_files = glob.glob(os.path.join(base_path, "test", "*.jsonl"))
    if test_files:
        test_raw_df = pd.concat([pd.read_json(f, lines=True) for f in test_files], ignore_index=True)
        print(f"  Loaded {len(test_raw_df):,} raw samples for Test.")
    else:
        print(f"🚨 Warning: No .jsonl files found for Test in '{os.path.join(base_path, 'test')}'.")

    print("✅ Raw data loading per split complete.")
    return train_raw_df, valid_raw_df, test_raw_df

# --- Data Cleaning Function ---
def clean_data(df):
    # cleaning for code and docstring pairs,
    # including removal of internal docstrings from the 'code' field
    if df.empty:
        return pd.DataFrame()

    initial_len = len(df)
    print(f"🧹 Cleaning {initial_len:,} samples...")

    # Keep only code and docstring
    df = df[['docstring', 'code']].copy()

    # Remove missing values
    df = df.dropna()

    # Clean strings (remove leading/trailing whitespace)
    df['docstring'] = df['docstring'].str.strip()
    df['code'] = df['code'].str.strip()

    # ---  Truncate docstring to first line ---
    df['docstring'] = df['docstring'].apply(lambda x: x.split('\n')[0].strip())

    # Remove empty strings after stripping
    df = df[df['docstring'] != '']
    df = df[df['code'] != '']

    # Remove internal docstrings from the 'code' column ---
    # Regex to match triple-quoted strings (docstrings)
    # This pattern tries to be robust for both """ and ''' and multiline content
    docstring_pattern = re.compile(r'(?:"""|\'\'\')(?:.|\n)*?(?:"""|\'\'\')')
    df['code'] = df['code'].apply(lambda x: docstring_pattern.sub('', x, count=1).strip()) # Only remove the first occurrence

    # Re-check for empty code after docstring removal
    df = df[df['code'] != '']


    # Length filters
    # Docstring length filter should consider the new, shorter length
    df = df[df['docstring'].str.len().between(10, 150)] # Adjusted max length for first line
    df = df[df['code'].str.len().between(50, 1000)] # Code length might change after docstring removal

    # Remove test functions (case-insensitive)
    df = df[~df['code'].str.contains(r'def test_', flags=re.IGNORECASE, na=False)]

    # Remove duplicates based on code content
    df = df.drop_duplicates(subset=['code'])

    print(f"✅ Cleaned from {initial_len:,} to {len(df):,} samples")
    return df.reset_index(drop=True)

# --- Main execution flow ---

# 1. Load raw data for each split separately
train_raw, valid_raw, test_raw = load_and_combine_raw_per_split(raw_data_base_path)

# 2. Clean each dataset independently
print("\n🧹 Cleaning each dataset independently...")
train_clean = clean_data(train_raw)
valid_clean = clean_data(valid_raw)
test_clean = clean_data(test_raw)
print("✅ Independent cleaning complete.")
show_stats(train_clean, valid_clean, test_clean, title="Cleaned Original Splits Statistics")


# 3. Adjust dataset sizes based on DESIRED_TRAIN_SAMPLES and desired ratios
print(f"\nAdjusting dataset sizes to achieve {DESIRED_TRAIN_SAMPLES:,} training samples and optimal ratios...")

# Sample the training set to the desired size
if len(train_clean) > DESIRED_TRAIN_SAMPLES:
    train_df_new = train_clean.sample(n=DESIRED_TRAIN_SAMPLES, random_state=42, replace=False)
    print(f"  Sampled Train set down to {len(train_df_new):,} samples.")
else:
    train_df_new = train_clean.copy()
    print(f"  Train set ({len(train_df_new):,}) is already smaller than or equal to desired {DESIRED_TRAIN_SAMPLES:,}. Using full cleaned Train set.")

# Calculate target sizes for validation and test based on the new train set size
# If train_df_new is TRAIN_PERCENTAGE of the conceptual total,
# then target_valid_size = (len(train_df_new) / TRAIN_PERCENTAGE) * VALID_PERCENTAGE
target_valid_size = int(len(train_df_new) * (VALID_PERCENTAGE / TRAIN_PERCENTAGE))
target_test_size = int(len(train_df_new) * (TEST_PERCENTAGE / TRAIN_PERCENTAGE))

# Sample from the cleaned validation and test sets
valid_sample_size = min(target_valid_size, len(valid_clean))
test_sample_size = min(target_test_size, len(test_clean))

valid_df_new = valid_clean.sample(n=valid_sample_size, random_state=42, replace=False)
test_df_new = test_clean.sample(n=test_sample_size, random_state=42, replace=False)

print(f"  Calculated target validation size: {target_valid_size:,} (actual sampled: {valid_sample_size:,})")
print(f"  Calculated target test size: {target_test_size:,} (actual sampled: {test_sample_size:,})")
print("✅ Datasets adjusted.")

# --- Show statistics for the new splits ---
show_stats(train_df_new, valid_df_new, test_df_new, title="Optimized Dataset Statistics (New Split)")

# --- Save the new datasets to JSONL files ---
print("\nSaving new datasets to JSONL files...")
train_df_new.to_json(os.path.join(processed_data_save_path, 'train_processed.jsonl'), orient='records', lines=True)
valid_df_new.to_json(os.path.join(processed_data_save_path, 'valid_processed.jsonl'), orient='records', lines=True)
test_df_new.to_json(os.path.join(processed_data_save_path, 'test_processed.jsonl'), orient='records', lines=True)
print(f"✅ New datasets saved to '{processed_data_save_path}'.")