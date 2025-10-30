# =============================================
# 🦠 PestisVeriteum - Fake News Detector (Modern UI + Auth + DB)
# =============================================

import streamlit as st
import torch
import os
import sqlite3
import hashlib
from datetime import datetime
from verifier.factcheck import verify_claim  # <-- Real fact verifier imported here

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
# STYLING (Glassmorphism + Clean Design)
# ------------------------------------------------------
st.set_page_config(page_title="PestisVeriteum", page_icon="🦠", layout="centered")

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at 10% 10%, #020617 0%, #0b1225 50%, #000814 100%);
    color: #f5f5f5;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 { color: #ff6b6b; text-align: center; letter-spacing: 1px; }
div[data-testid="stSidebar"] {
    background: rgba(5, 10, 25, 0.85);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
.stButton>button {
    background: linear-gradient(90deg, #ff5a5a, #ff7b00);
    color: white;
    border-radius: 12px;
    padding: 0.6em 1.4em;
    border: none;
    font-weight: 600;
    transition: all 0.3s ease-in-out;
}
.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #ff7b00, #ff5a5a);
}
.stTextInput>div>div>input, textarea {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: white !important;
}
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 15px;
    padding: 1.5em;
    margin-top: 1em;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
footer { text-align: center; color: #aaa; padding-top: 10px; font-size: 0.9em; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------
st.sidebar.image("https://i.imgur.com/0S4gHln.png", width=120)
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧪 Detector", "ℹ️ About", "📬 Contact"])

# ------------------------------------------------------
# AUTH SYSTEM
# ------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None

if st.session_state.user is None:
    st.markdown("<h1>🧠 PestisVeriteum</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#bfcde0;'>Next-Generation Fake News Detection AI</p>", unsafe_allow_html=True)

    tabs = st.tabs(["🔐 Login", "🆕 Sign Up"])

    with tabs[0]:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            c = conn.cursor()
            c.execute("SELECT password_hash FROM users WHERE username=?", (username,))
            row = c.fetchone()
            if row and verify_password(password, row[0]):
                st.session_state.user = username
                st.success(f"Welcome back, {username}!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")

    with tabs[1]:
        new_user = st.text_input("New Username", key="new_user")
        new_pass = st.text_input("New Password", type="password", key="new_pass")
        if st.button("Sign Up"):
            if len(new_user) < 3 or len(new_pass) < 6:
                st.warning("Username ≥ 3 chars, Password ≥ 6 chars.")
            else:
                try:
                    conn.execute("INSERT INTO users VALUES (?, ?)", (new_user, hash_password(new_pass)))
                    conn.commit()
                    st.success("✅ Account created! You can now log in.")
                except:
                    st.error("❌ Username already exists.")

else:
    st.sidebar.success(f"👤 Logged in as: {st.session_state.user}")
    if st.sidebar.button("Logout"):
        st.session_state.user = None
        st.experimental_rerun()

    # ------------------------------------------------------
    # MAIN CONTENT
    # ------------------------------------------------------
    if page == "🏠 Home":
        st.markdown("<h1>Welcome to PestisVeriteum</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card' style='text-align:center;'>
        <p>The next-generation fake news detection AI.<br>
        Enter any claim, and PestisVeriteum will instantly analyze its truthfulness using a fine-tuned NLI model.<br><br>
        🧠 <b>Trusted AI, built for truth.</b>
        </p>
        </div>
        """, unsafe_allow_html=True)

    elif page == "🧪 Detector":
        st.markdown("<h1>Fake News Detector</h1>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        claim = st.text_area("Enter your claim:", height=120)

        if st.button("🔍 Analyze"):
            if claim.strip() == "":
                st.warning("Please type something first.")
            else:
                with st.spinner("Analyzing claim using AI..."):
                    result = verify_claim(claim)

                label = result["label"]
                confidence = result["confidence"]

                color = "lime" if label == "True" else ("red" if label == "False" else "orange")

                st.markdown(f"<h3 style='color:{color};text-align:center;'>✅ Result: {label}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center;'>Confidence: {confidence:.2f}</p>", unsafe_allow_html=True)

                conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?)",
                             (st.session_state.user, claim, label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "ℹ️ About":
        st.markdown("<h1>About PestisVeriteum</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
        <p>PestisVeriteum is a deep-learning-powered truth verification system that uses 
        <b>Natural Language Inference (NLI)</b> and <b>Transformer-based reasoning</b> to verify factual accuracy.</p>
        <ul>
            <li><b>Mission:</b> Combat misinformation with AI.</li>
            <li><b>Technology:</b> RoBERTa-large trained on FEVER & MNLI datasets.</li>
            <li><b>Built by:</b> PestisVeriteum Research Lab, 2025.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    elif page == "📬 Contact":
        st.markdown("<h1>Contact Us</h1>", unsafe_allow_html=True)
        st.markdown("<div class='card'>", unsafe_allow_html=True)
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
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<footer>© 2025 PestisVeriteum. All rights reserved.</footer>", unsafe_allow_html=True)
