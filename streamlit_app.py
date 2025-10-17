# =============================================
# 🦠 PestisVeriteum - Fake News Detector (Full UI + Auth + DB)
# =============================================

import streamlit as st
import torch
import os
import sqlite3
import hashlib
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime

# ------------------------------------------------------
# DATABASE SETUP
# ------------------------------------------------------
def init_db():
    conn = sqlite3.connect("pestisveriteum.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    username TEXT,
                    claim TEXT,
                    label TEXT,
                    date TEXT)''')
    conn.commit()
    return conn

conn = init_db()

# ------------------------------------------------------
# PASSWORD HASHING
# ------------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password, hashed):
    return hash_password(password) == hashed

# ------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = "./fakenews_model"
    if not os.path.exists(model_path):
        st.warning("⚠️ Local model not found, downloading fallback model...")
        model_path = "bert-base-uncased"

    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
    except:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

# ------------------------------------------------------
# CUSTOM CSS (Beautiful + Clean)
# ------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 10% 10%, #0f1724 0%, #071028 35%, #001320 100%);
    color: #f2f2f2;
}
h1, h2, h3 {
    color: #ff5a5a;
    text-align: center;
    font-family: 'Segoe UI';
}
.stButton>button {
    background: linear-gradient(90deg, #ff5a5a, #ff7b00);
    color: white;
    border-radius: 10px;
    padding: 0.6em 1.2em;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #ff7b00, #ff5a5a);
}
.sidebar .sidebar-content {
    background-color: #0d1321;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="PestisVeriteum", page_icon="🦠", layout="centered")

# ------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------
st.sidebar.image("https://i.imgur.com/0S4gHln.png", width=120)
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧪 Detector", "ℹ️ About", "📬 Contact"])

# ------------------------------------------------------
# LOGIN / SIGNUP SYSTEM
# ------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.sidebar.subheader("Login / Sign Up")

    login_tab, signup_tab = st.sidebar.tabs(["🔐 Login", "🆕 Sign Up"])

    # --- Login tab
    with login_tab:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            c = conn.cursor()
            c.execute("SELECT password_hash FROM users WHERE username=?", (username,))
            row = c.fetchone()
            if row and verify_password(password, row[0]):
                st.session_state.user = username
                st.success(f"Welcome back, {username}!")
            else:
                st.error("Invalid credentials.")

    # --- Signup tab
    with signup_tab:
        new_user = st.text_input("New Username", key="new_user")
        new_pass = st.text_input("New Password", type="password", key="new_pass")
        if st.button("Sign Up"):
            if len(new_user) < 3 or len(new_pass) < 6:
                st.warning("Username ≥ 3 chars, Password ≥ 6 chars.")
            else:
                try:
                    conn.execute("INSERT INTO users VALUES (?, ?)", (new_user, hash_password(new_pass)))
                    conn.commit()
                    st.success("Account created! Please log in.")
                except:
                    st.error("Username already exists.")

else:
    st.sidebar.success(f"👤 Logged in as: {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.rerun()

# ------------------------------------------------------
# MAIN CONTENT BASED ON NAVIGATION
# ------------------------------------------------------
if st.session_state.user:

    # 🏠 HOME
    if page == "🏠 Home":
        st.markdown("<h1>Welcome to PestisVeriteum</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p style='text-align:center;color:#cfd9e4;'>
        The next-generation fake news detection AI.<br>
        Enter any claim, and PestisVeriteum will instantly analyze its truthfulness
        using a fine-tuned BERT model.<br><br>
        🧠 <b>Trusted AI, built for truth.</b>
        </p>
        """, unsafe_allow_html=True)

    # 🧪 DETECTOR
    elif page == "🧪 Detector":
        st.markdown("<h1>Fake News Detector</h1>", unsafe_allow_html=True)
        tokenizer, model, device = load_model()

        label_mapping = {
            0: 'half-true', 1: 'mostly-true', 2: 'false',
            3: 'true', 4: 'barely-true', 5: 'pants-fire'
        }

        claim = st.text_area("Enter your claim:", height=120)
        if st.button("Analyze"):
            if claim.strip() == "":
                st.warning("Please type something first.")
            else:
                model.eval()
                inputs = tokenizer(claim, return_tensors="pt", padding="max_length",
                                   truncation=True, max_length=128).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                pred_idx = torch.argmax(outputs.logits, dim=1).item()
                label = label_mapping[pred_idx]

                st.markdown(f"<h2 style='color:lime;'>Prediction: {label.upper()}</h2>", unsafe_allow_html=True)

                conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?)",
                             (st.session_state.user, claim, label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()

    # ℹ️ ABOUT
    elif page == "ℹ️ About":
        st.markdown("<h1>About PestisVeriteum</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:justify;'>
        PestisVeriteum is a truth verification project built with modern deep learning tools.
        It utilizes <b>BERT (Bidirectional Encoder Representations from Transformers)</b>
        to classify claims into multiple truthfulness levels.<br><br>
        <b>Mission:</b> Fight misinformation with AI.<br>
        <b>Created by:</b> <i>PestisVeriteum Research Lab</i><br>
        <b>Year:</b> 2025
        </div>
        """, unsafe_allow_html=True)

    # 📬 CONTACT
    elif page == "📬 Contact":
        st.markdown("<h1>Contact Us</h1>", unsafe_allow_html=True)
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        msg = st.text_area("Your Message")
        if st.button("Send Message"):
            if not msg.strip():
                st.warning("Please write a message.")
            else:
                conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?)",
                             (name or "Anonymous", email, msg, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
                st.success("✅ Message sent successfully!")

    # Footer
    st.markdown("<hr><center>© 2025 PestisVeriteum. All rights reserved.</center>", unsafe_allow_html=True)
else:
    st.info("🔒 Please log in or sign up from the sidebar to continue.")


