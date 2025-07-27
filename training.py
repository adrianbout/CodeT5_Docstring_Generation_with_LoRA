# Fine-tuning CodeT5 Model (Salesforce/codet5-base) with LoRA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
import evaluate
import numpy as np
import torch
import pandas as pd
import os

# --- LoRA/PEFT specific imports ---
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Loading the JSONL files
print("Attempting to load data from JSONL files...")
base_data_path = "data/processedData/" 

try:
    train_clean = pd.read_json(os.path.join(base_data_path, 'train_processed.jsonl'), lines=True)
    valid_clean = pd.read_json(os.path.join(base_data_path, 'valid_processed.jsonl'), lines=True)
    test_clean = pd.read_json(os.path.join(base_data_path, 'test_processed.jsonl'), lines=True)
    print("✅ Data loaded successfully from JSONL files.")
    print(f"Train data shape: {train_clean.shape}")
    print(f"Validation data shape: {valid_clean.shape}")
    print(f"Test data shape: {test_clean.shape}")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure 'train_processed.jsonl', 'valid_processed.jsonl', and 'test_processed.jsonl' are in the script's directory.")
    exit()

# --- 1. Load Pre-trained CodeT5 Model and Tokenizer ---
model_name = "Salesforce/codet5-base"
print(f"Loading CodeT5 model: {model_name}...")

# Check for CUDA availability and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name) # Load model
print(f"✅ Loaded CodeT5 model: {model_name}")

# --- Apply LoRA ---
print("Applying LoRA to the model...")

lora_config = LoraConfig(
    r=32, # LoRA attention dimension (rank)
    lora_alpha=64, # Scaling factor
    target_modules=["q", "v"], # Common target modules for T5-like models (CodeT5 also uses 'q' and 'v')
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM, # Specify task type
)

model = get_peft_model(model, lora_config)
model.to(device) # Move the PEFT-wrapped model to GPU if available
model.print_trainable_parameters() # Show the number of trainable parameters
print("✅ LoRA applied. Only a small fraction of parameters will be trained.")


# --- 2. Prepare Data for Model Training ---
print("Converting pandas DataFrames to Hugging Face Dataset objects...")
train_dataset = Dataset.from_pandas(train_clean)
valid_dataset = Dataset.from_pandas(valid_clean)
test_dataset = Dataset.from_pandas(test_clean)
print("✅ Datasets converted.")

# Define a function tokenize the data
def tokenize_function(examples):
    # Setup the tokenizer for the inputs - code
    inputs = [code for code in examples["code"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    # Setup the tokenizer for targets - output
    labels = tokenizer(text_target=examples["docstring"], max_length=256, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("⚙️ Tokenizing datasets...")
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["code", "docstring"])
tokenized_valid_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=["code", "docstring"])
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=["code", "docstring"])

print("✅ Datasets tokenized.")
print(f"Tokenized Train Dataset size: {len(tokenized_train_dataset):,}")
print(f"Tokenized Valid Dataset size: {len(tokenized_valid_dataset):,}")
print(f"Tokenized Test Dataset size: {len(tokenized_test_dataset):,}")

# --- 3. Define a Metric for Evaluation ---
metric = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Check if pad_token_id is available, otherwise pick a safe default (e.g., 0)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        print("Warning: Tokenizer does not have a pad_token_id. Using 0 as fallback.")
        pad_token_id = 0 # Fallback if no specific pad_token_id

    # Replace -100 with pad_token_id in preds
    preds = np.where(preds == -100, pad_token_id, preds)
    # --- End Crucial Modification ---

    # Your existing fix for dtype (which is now confirmed as int64, so this line
    # might become redundant, but keeping it doesn't hurt if future inputs vary)
    if preds.dtype != np.int64 and preds.dtype != np.int32:
        print(f"Warning: preds dtype is {preds.dtype}, casting to int64.")
        preds = preds.astype(np.int64)
    
    # Debug prints (can remove after verifying fix)
    # print(f"DEBUG: preds type before decode: {type(preds)}")
    # print(f"DEBUG: preds dtype before decode: {preds.dtype}")
    # print(f"DEBUG: preds shape before decode: {preds.shape}")
    # print(f"DEBUG: Sample preds values (first 5): {preds.flatten()[:5]}")
    # print(f"DEBUG: Max value in preds after cleaning: {preds.max()}")
    # print(f"DEBUG: Min value in preds after cleaning: {preds.min()}") # Should now be >=0

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Labels part is correct, as -100 is expected here and replaced with pad_token_id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(pred.strip().split("\n")) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split("\n")) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# --- 4. Define Training Arguments ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./codet5_base_fine_tuned_docstrings_lora", # MODIFIED: Changed output directory name for consistency
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1.0e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none",
    logging_dir='./logs_codet5_base_lora', # MODIFIED: Changed logging directory name for consistency
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    push_to_hub=False,
)

# --- 5. Initialize and Train the Seq2SeqTrainer ---
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("🚀 Starting fine-tuning with CodeT5 base with LoRA...")
trainer.train()
print("✅ Fine-tuning complete!")

print("\n--- CodeT5 base LoRA Fine-tuning complete. ---")

# --- 6. Evaluate on the Test Set (Optional, but good practice) ---
print("\n📝 Evaluating on the test set...")
results = trainer.evaluate(tokenized_test_dataset)
print(results)

# --- 7. Save the Fine-tuned Model and Tokenizer ---
# When using PEFT, save_model will save the adapters and the PEFT config.
# The base model weights are not modified.
final_model_path = "./fine_tuned_codet5_base_docstring_model_lora"
trainer.save_model(final_model_path)
tokenizer.save_pretrained(final_model_path) # Tokenizer is the same
print(f"🎉 Fine-tuned LoRA adapters and tokenizer saved to {final_model_path}")

# Example of how to load and use the fine-tuned model for inference
print("\nTesting inference with the fine-tuned model (LoRA adapters will be loaded automatically):")
from transformers import pipeline
# For pipeline with PEFT, you need to load the base model first, then the PEFT model
from peft import PeftModel, PeftConfig

# Load the PEFT config
peft_config_loaded = PeftConfig.from_pretrained(final_model_path)
# Load the base model
# This will now  load "Salesforce/codet5-base" as the base model
base_model_inference = AutoModelForSeq2SeqLM.from_pretrained(peft_config_loaded.base_model_name_or_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
# Load the PEFT model (which attaches the LoRA adapters to the base model)
model_for_inference = PeftModel.from_pretrained(base_model_inference, final_model_path).to(device)

# Ensure device is set for the pipeline
# The pipeline will use the `model_for_inference` which has LoRA adapters loaded
generator = pipeline("text2text-generation", model=model_for_inference, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

sample_code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""

input_text = "summarize: " + sample_code # Add the "summarize: " prefix for inference

generated_docstring = generator(input_text, max_length=256, num_beams=5, early_stopping=True)
print("\n--- Original Code ---")
print(sample_code)
print("\n--- Generated Docstring ---")
print(generated_docstring[0]['generated_text'])

sample_code_2 = """
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
"""
input_text_2 = "summarize: " + sample_code_2 # Add the "summarize: " prefix for inference
generated_docstring_2 = generator(input_text_2, max_length=128, num_beams=5, early_stopping=True)
print("\n--- Original Code 2 ---")
print(sample_code_2)
print("\n--- Generated Docstring 2 ---")
print(generated_docstring_2[0]['generated_text'])