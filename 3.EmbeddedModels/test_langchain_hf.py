from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
model.to("cpu")

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=pipe)

# --- Chat-style system prompt ---
prompt = """You are a helpful assistant.
Question: Write a 5 line poem on cricket
Answer:"""

print("Generated text:\n", llm.invoke(prompt))
