import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from peft import PeftModel, PeftConfig

@st.cache_resource
def load_model_pipeline(model_path="./FINETUNED_MODEL"):
    # Load PEFT config
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path

    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Create pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    return pipe

# Load once
pipe = load_model_pipeline()

# UI
st.title("CodeT5 LoRA Docstring Generator")
st.write("Paste a Python function below to generate its docstring.")

code_input = st.text_area("Your Python function", height=250)

if st.button("Generate Docstring"):
    if not code_input.strip():
        st.warning("Please enter a function.")
    else:
        prompt = "summarize: " + code_input
        with st.spinner("Generating..."):
            result = pipe(
                prompt,
                max_new_tokens=100,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            docstring = result[0]['generated_text']
            st.success("Generated Docstring:")
            st.code(docstring, language="python")
