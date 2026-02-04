from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# --- Lightweight chat model (works offline, free, small) ---
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,  # use float32 for CPU compatibility
    low_cpu_mem_usage=True
)
model.to("cpu")  # force CPU mode for laptops

# Create text-generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True
)

# Wrap in LangChain HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=pipe)

# Example conversation (Anthropic-like chat)
prompt = """You are a helpful and polite AI assistant.
Question: What is the full form of AI?
Answer:"""

print("Generated text:\n")
print(llm.invoke(prompt))
 