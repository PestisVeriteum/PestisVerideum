# -------------------------
# streamlit_app.py
# -------------------------

import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load model and vectorizer
# -------------------------
@st.cache_data
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -------------------------
# Prediction function
# -------------------------
def predict_news(text):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]  # 0 = Fake, 1 = True
    probability = model.predict_proba(text_vector)[0][prediction]  # confidence
    return prediction, probability

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Pestis Veriteum - Fake News Detector", page_icon="📰")

st.title("Pestis Veriteum 📰")
st.subheader("Detect Fake News and Check Truth Scores")

st.write("""
Enter any news article, social media post, or statement below and get a prediction 
of whether it’s likely **True** or **Fake**, along with a confidence score.
""")

# Text input
user_input = st.text_area("Paste news text here:")

if st.button("Check News"):
    if user_input.strip() != "":
        label, confidence = predict_news(user_input)
        result_text = "✅ True" if label == 1 else "❌ Fake"
        st.markdown(f"### Prediction: {result_text}")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
    else:
        st.warning("Please enter some text to check.")

# -------------------------
# Optional: Footer
# -------------------------
st.markdown("---")
st.markdown("Pestis Veriteum AI | Built for research and educational purposes")

