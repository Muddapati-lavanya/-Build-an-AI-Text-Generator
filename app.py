import streamlit as st
from model_utils import analyze_sentiment, generate_text

# Streamlit setup
st.set_page_config(page_title="AI Sentiment Text Generator", page_icon="🤖")
st.title("🤖 AI Sentiment-Aligned Text Generator")
st.write("Enter a prompt and get an AI-generated paragraph matching its sentiment!")

# User input
user_prompt = st.text_area("✏️ Enter your prompt:", height=100)
max_len = st.slider("Select max text length:", min_value=50, max_value=300, value=150, step=10)

if st.button("Generate Text"):
    if user_prompt.strip() == "":
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("Analyzing sentiment..."):
            sentiment = analyze_sentiment(user_prompt)
            st.success(f"Detected Sentiment: **{sentiment}**")

        with st.spinner("Generating sentiment-aligned text..."):
            generated = generate_text(user_prompt, sentiment, max_len=max_len)
            st.subheader("📝 Generated Text:")
            st.write(generated)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Hugging Face Transformers.")
