from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize the model for text generation
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map="auto"
)

# Text to summarize
text = """Artificial Intelligence (AI) has been a subject of fascination and intensive research for decades. AI technologies have evolved from basic algorithms to advanced machine learning models, profoundly impacting industries, healthcare, and everyday life. The future of AI promises even more revolutionary changes, with potential advancements in autonomous vehicles, personalized medicine, and intelligent automation."""

# Create a prompt for summarization
prompt = f"Summarize the following text in 2-3 sentences:\n\n{text}\n\nSummary:"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Extract only the generated summary (remove the input prompt)
summary = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print("Summary:", summary.strip())