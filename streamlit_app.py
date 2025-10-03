# =============================================
# 🦠 PestisVeriteum - Fake News Detector (Streamlit)
# Stylish UI + Login + Live Predictions
# =============================================

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime

# -------------------------
# 1️⃣ Login Setup (simple version)
# -------------------------
def check_login(username, password):
    # In a real app, use a secure database!
    # Example credentials
    users = {"admin": "pass123", "user": "pestis2025"}
    return username in users and users[username] == password

# -------------------------
# 2️⃣ Streamlit Page config
# -------------------------
st.set_page_config(
    page_title="PestisVeriteum",
    page_icon="🦠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------
# 3️⃣ Sidebar login
# -------------------------
st.sidebar.image("https://i.imgur.com/0S4gHln.png", width=100)  # Logo
st.sidebar.title("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if login_button:
    if check_login(username, password):
        st.session_state.authenticated = True
        st.sidebar.success("Logged in successfully!")
    else:
        st.session_state.authenticated = False
        st.sidebar.error("Incorrect username or password!")

# -------------------------
# 4️⃣ If authenticated, show main app
# -------------------------
if st.session_state.authenticated:
    st.markdown("<h1 style='color: darkred;'>🦠 PestisVeriteum</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:gray;'>Type a claim and get its truth label instantly!</p>", unsafe_allow_html=True)
    
    # Optional: Live time
    st.markdown(f"<p style='color:blue;'>Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)

    # -------------------------
    # 5️⃣ Load Model & Tokenizer
    # -------------------------
    @st.cache_resource(show_spinner=False)
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained("./fakenews_model")
        model = BertForSequenceClassification.from_pretrained("./fakenews_model").to(device)
        return tokenizer, model, device

    tokenizer, model, device = load_model()

    # Label mapping
    label_mapping = {0: 'half-true', 1:'mostly-true', 2:'false', 3:'true', 4:'barely-true', 5:'pants-fire'}

    # -------------------------
    # 6️⃣ Input & Prediction
    # -------------------------
    user_input = st.text_area("Enter a claim here:", height=120)

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please type a claim before clicking Predict.")
        else:
            # Prediction
            model.eval()
            inputs = tokenizer(user_input, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            pred_idx = torch.argmax(outputs.logits, dim=1).item()
            prediction = label_mapping[pred_idx]

            # Show result with nice styling
            st.markdown(f"<h2 style='color:green;'>Prediction: {prediction.upper()}</h2>", unsafe_allow_html=True)

            # Optional: show raw logits
            if st.checkbox("Show raw logits"):
                st.write(outputs.logits.cpu().numpy())

    # -------------------------
    # 7️⃣ Footer
    # -------------------------
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='color:gray;'>Powered by PestisVeriteum AI - 2025</p>", unsafe_allow_html=True)

else:
    st.info("Please log in using the sidebar to access PestisVeriteum.")
