import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel, PeftConfig
import os

# Define the path where your fine-tuned model (LoRA adapters) and tokenizer are saved
FINETUNED_MODEL_PATH = "./codet5_base_fine_tuned_docstrings_lora"
#FINETUNED_MODEL_PATH = "./FINETUNED_MODEL"

# --- 1. Load PEFT Config and Base Model ---
print(f"Loading PEFT config from: {FINETUNED_MODEL_PATH}")
try:
    peft_config_loaded = PeftConfig.from_pretrained(FINETUNED_MODEL_PATH)
    print("✅ PEFT config loaded successfully.")
except Exception as e:
    print(f"❌ Error loading PEFT config: {e}")
    print("Please ensure the directory contains 'adapter_config.json'.")
    exit()

# The base model name is stored in the PEFT config
base_model_name = peft_config_loaded.base_model_name_or_path
print(f"Base model identified from config: {base_model_name}")

# Set device (GPU if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the base model
print(f"Loading base model: {base_model_name}...")
try:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    print("✅ Base model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading base model: {e}")
    print(f"Please ensure '{base_model_name}' is a valid Hugging Face model ID or path.")
    exit()

# Load the tokenizer
print(f"Loading tokenizer from: {FINETUNED_MODEL_PATH}")
try:
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    print("Please ensure the directory contains tokenizer files (e.g., 'tokenizer.json', 'tokenizer_config.json').")
    exit()

# --- 2. Load the PEFT Model (attach LoRA adapters to the base model) ---
print(f"Loading LoRA adapters from: {FINETUNED_MODEL_PATH}")
try:
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)
    model.to(device)
    model.eval() # Set model to evaluation mode (important for inference)
    print("✅ LoRA adapters loaded and model prepared for inference.")
except Exception as e:
    print(f"❌ Error loading PEFT model: {e}")
    print("Ensure 'adapter_model.bin' and 'adapter_config.json' are in the specified path.")
    exit()

# --- 3. Create a Hugging Face Pipeline for Easy Inference ---
print("\nInitializing text2text-generation pipeline...")
try:
    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    print("✅ Pipeline initialized.")
except Exception as e:
    print(f"❌ Error initializing pipeline: {e}")
    exit()

# --- 4. Define Sample Code for Inference ---
print("\n--- Performing Inference ---")

sample_codes = [
    # 1. Basic factorial (corrected)
    """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
""",
    # 2. Calculate Average
    """
def calculate_average(numbers):
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
""",
    # 3. Quicksort
    """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
""",
    # 4. Check if a number is prime
    """
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True
""",
    # 5. Reverse a string
    """
def reverse_string(s):
    return s[::-1]
""",
    # 6. Find the maximum element in a list
    """
def find_max(data_list):
    if not data_list:
        raise ValueError("List cannot be empty")
    max_val = data_list[0]
    for item in data_list:
        if item > max_val:
            max_val = item
    return max_val
""",
    # 7. Count word occurrences in a sentence
    """
from collections import Counter

def count_words(sentence):
    words = sentence.lower().split()
    return Counter(words)
""",
    # 8. Fibonacci sequence
    """
def fibonacci(n):
    a, b = 0, 1
    for i in range(n):
        yield a
        a, b = b, a + b
""",
    # 9. Simple file read
    """
def read_file_content(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    return content
""",
    # 10. Simple class with a method
    """
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def calculate_area(self):
        import math
        return math.pi * (self.radius ** 2)
""",
    # 11. Sum of even numbers
    """
def sum_even_numbers(numbers):
    total = 0
    for num in numbers:
        if num % 2 == 0:
            total += num
    return total
""",
    # 12. Check for palindrome
    """
def is_palindrome(text):
    cleaned_text = "".join(char.lower() for char in text if char.isalnum())
    return cleaned_text == cleaned_text[::-1]
""",
    # 13. Merge two dictionaries
    """
def merge_dictionaries(dict1, dict2):
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict
""",
    # 14. Convert Celsius to Fahrenheit
    """
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32
""",
    # 15. Linear search
    """
def linear_search(arr, target):
    for i, element in enumerate(arr):
        if element == target:
            return i
    return -1
"""
]

# --- 5. Generate Docstrings and Print Results ---

def generate_docstring(code_snippet, pipeline_generator):
    input_text = "summarize: " + code_snippet
    
    # Using max_new_tokens for output length control
    # num_beams=5 is a good balance.
    # no_repeat_ngram_size=3 helps prevent repetition.
    generated_output = pipeline_generator(
        input_text,
        max_new_tokens=100, # Adjusted for potentially longer docstrings
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return generated_output[0]['generated_text']

for i, code in enumerate(sample_codes):
    print(f"\n--- Original Code {i+1} ---")
    print(code)
    generated_docstring = generate_docstring(code, generator)
    print(f"\n--- Generated Docstring {i+1} ---")
    print(generated_docstring)

print("\n--- Inference Script Finished ---")