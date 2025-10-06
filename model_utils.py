from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

# Load sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Load GPT-2 model for text generation
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def analyze_sentiment(text):
    """Detect sentiment from input text"""
    result = sentiment_analyzer(text)[0]
    return result["label"]

def generate_text(prompt, sentiment_label, max_len=150):
    """Generate text aligned with detected sentiment"""
    if sentiment_label == "NEGATIVE":
        modified_prompt = f"Write a sad and negative paragraph about: {prompt}"
    elif sentiment_label == "POSITIVE":
        modified_prompt = f"Write a happy and positive paragraph about: {prompt}"
    else:
        modified_prompt = f"Write a neutral and balanced paragraph about: {prompt}"
    
    inputs = tokenizer.encode(modified_prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=max_len,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.9,
        do_sample=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

