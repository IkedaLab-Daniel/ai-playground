from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize the tokenizer and model
model_name = "meta-llama/Llama-2-7b-chat-hf"  # or other Llama 2 variants
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# Generate text based on a prompt
prompt = "What is the future of artificial intelligence?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Text:", generated_text)