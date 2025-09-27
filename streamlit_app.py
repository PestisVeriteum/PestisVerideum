import streamlit as st
import joblib

# Load your saved model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit app UI
st.set_page_config(page_title="Pestis Veriteum - Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Pestis Veriteum")
st.subheader("Your AI-powered Fake News Detector")

st.markdown(
    "Type or paste a news headline/statement below, and the model will predict "
    "whether it's **likely true** or **fake**."
)

# Text input
user_input = st.text_area("Enter a news headline or statement:", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        # Transform input and predict
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        # Display result
        if prediction == 1:
            st.success(f"✅ This looks **TRUE** with {proba[1]*100:.2f}% confidence.")
        else:
            st.error(f"❌ This looks **FAKE** with {proba[0]*100:.2f}% confidence.")

        # Show probability breakdown
        st.progress(float(proba[1]))
        st.write(f"Truth Score: {proba[1]*100:.2f}%")
