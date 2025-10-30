# =============================================
# 🧠 PestisVeriteum - Truth Verification AI (Smart Web + MNLI)
# =============================================

import streamlit as st
import torch
import os
import sqlite3
import hashlib
from datetime import datetime
from transformers import pipeline
from googlesearch import search  # lightweight web info fetch
import requests
from bs4 import BeautifulSoup

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
# MODEL SETUP
# ------------------------------------------------------
@st.cache_resource
def load_model():
    model_id = "facebook/bart-large-mnli"
    return pipeline("zero-shot-classification", model=model_id)

verifier = load_model()

# ------------------------------------------------------
# WEB INFO FETCH
# ------------------------------------------------------
def fetch_web_summary(query, max_results=3):
    """Fetch short factual snippets from the web."""
    snippets = []
    try:
        for url in search(query, num_results=max_results):
            try:
                page = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(page.text, "html.parser")
                text = ' '.join(p.text for p in soup.find_all('p'))
                snippets.append(text[:500])
            except Exception:
                continue
    except Exception:
        pass
    return " ".join(snippets) if snippets else ""

# ------------------------------------------------------
# CLAIM VERIFICATION
# ------------------------------------------------------
def verify_claim(claim):
    evidence = fetch_web_summary(claim)
    context = evidence if evidence else "No strong evidence found online."

    res = verifier(
        f"Claim: {claim}. Based on evidence: {context}",
        candidate_labels=["true", "false", "unclear"]
    )

    label = res["labels"][0].capitalize()
    confidence = res["scores"][0]
    return {"label": label, "confidence": confidence, "evidence": context[:800]}

# ------------------------------------------------------
# STYLING
# ------------------------------------------------------
st.set_page_config(page_title="PestisVeriteum", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #03071e, #0b1225, #1a1a2e);
    color: #e8eaed;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #ff6b6b;
    text-align: center;
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
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff7b00, #ff5a5a);
}
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 15px;
    padding: 1.5em;
    margin-top: 1em;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------
st.sidebar.image("https://i.imgur.com/0S4gHln.png", width=120)
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🧪 Detector", "📊 History", "ℹ️ About", "📬 Contact"])

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
    # PAGES
    # ------------------------------------------------------
    if page == "🏠 Home":
        st.markdown("<h1>Welcome to PestisVeriteum</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card' style='text-align:center;'>
        <p><b>PestisVeriteum</b> is the AI built to detect misinformation using modern Natural Language Reasoning and live web analysis.<br><br>
        🧠 <b>Built for Truth. Powered by Intelligence.</b></p>
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
                with st.spinner("Analyzing claim using AI and the web..."):
                    result = verify_claim(claim)

                label = result["label"]
                confidence = result["confidence"]
                evidence = result["evidence"]

                color = "lime" if label == "True" else ("red" if label == "False" else "orange")
                st.markdown(f"<h3 style='color:{color};text-align:center;'>🧠 Result: {label}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align:center;'>Confidence: {confidence:.2f}</p>", unsafe_allow_html=True)
                with st.expander("🔎 Web Evidence"):
                    st.write(evidence)

                conn.execute("INSERT INTO predictions VALUES (?, ?, ?, ?)",
                             (st.session_state.user, claim, label, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()
        st.markdown("</div>", unsafe_allow_html=True)

    elif page == "📊 History":
        st.markdown("<h1>Your Past Verifications</h1>", unsafe_allow_html=True)
        c = conn.cursor()
        c.execute("SELECT claim, label, date FROM predictions WHERE username=?", (st.session_state.user,))
        data = c.fetchall()
        if not data:
            st.info("No history yet.")
        else:
            for claim, label, date in data[::-1]:
                st.markdown(f"🕒 **{date}** — `{label}`\n> {claim}")

    elif page == "ℹ️ About":
        st.markdown("<h1>About PestisVeriteum</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div class='card'>
        <p>PestisVeriteum is a cutting-edge truth verification engine powered by <b>BART-Large-MNLI</b> for natural language inference and <b>real-time web evidence</b> retrieval.</p>
        <p>Developed by Gaya Tahir © 2025 — <b>Fighting misinformation with AI and science.</b></p>
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

    st.markdown("<footer style='text-align:center;color:#777;'>© 2025 PestisVeriteum. All rights reserved.</footer>", unsafe_allow_html=True)
