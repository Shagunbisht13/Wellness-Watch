"""
Module 13: Dashboard Module (Streamlit + Flask + Matplotlib)
Displays:
 - Live emotion
 - Alert logs
 - Proof history
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time
from datetime import datetime

# Add project root to sys.path to import database_helper
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from database.database_helper import fetch_recent, fetch_stats

st.set_page_config(
    page_title="Harassment Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the dashboard
st.markdown("""
<style>
    .reportview-container {
        background: #0f1115;
    }
    .metric-card {
        background-color: #1e212b;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid #2d313f;
    }
    .metric-label {
        font-size: 14px;
        color: #a0aec0;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
    }
    .harassment { color: #ff4b4b; }
    .stress { color: #ffa421; }
    .normal { color: #2eeb7a; }
    
    .alert-box {
        padding: 20px;
        border-radius: 10px;
        background-color: rgba(255, 75, 75, 0.1);
        border: 2px solid #ff4b4b;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 20px;
        animation: blink 2s linear infinite;
    }
    @keyframes blink {
        50% { opacity: 0.5; }
    }
    .section-header {
        font-size: 22px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #2d313f;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Harassment Detection Dashboard")
st.markdown("Visualizing real-time multimodal data and proof evidence.")

# Auto-refresh mechanism
placeholder = st.empty()

def render_dashboard():
    with placeholder.container():
        # Fetch data from MongoDB
        records = fetch_recent(100)
        stats = fetch_stats()
        
        latest = records[0] if records else None
        
        # ── 1. LIVE EMOTION MODULE ─────────────────────────────────────────────
        st.markdown('<div class="section-header">📡 Live Emotion Status</div>', unsafe_allow_html=True)
        
        if latest:
            # Alert Module Trigger
            if latest["final_result"] == "harassment":
                st.markdown(f"""
                <div class="alert-box">
                    <h2>🚨 HARASSMENT DETECTED 🚨</h2>
                    <p>Immediate action required! Event recorded at {latest['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            
            def get_color_class(val):
                v = str(val).lower()
                if "harassment" in v or "angry" in v or "abusive" in v: return "harassment"
                if "stress" in v or "sad" in v: return "stress"
                return "normal"

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Last Updated</div>
                    <div class="metric-value" style="color: #cbd5e1;">{latest['timestamp'].split(' ')[1]}</div>
                </div>""", unsafe_allow_html=True)
                
            with col2:
                c = get_color_class(latest['face_emotion'])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Face Emotion</div>
                    <div class="metric-value {c}">{latest['face_emotion'].upper()}</div>
                </div>""", unsafe_allow_html=True)

            with col3:
                c = get_color_class(latest['speech_emotion'])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Voice Emotion</div>
                    <div class="metric-value {c}">{latest['speech_emotion'].upper()}</div>
                </div>""", unsafe_allow_html=True)

            with col4:
                c = get_color_class(latest['final_result'])
                st.markdown(f"""
                <div class="metric-card" style="border-color: #4b5563;">
                    <div class="metric-label">Final Result</div>
                    <div class="metric-value {c}">{latest['final_result'].upper()}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Waiting for data from capture module...")

        # ── 2. ALERT LOGS MODULE ─────────────────────────────────────────────
        st.markdown('<div class="section-header">⚠️ Alert Logs Module</div>', unsafe_allow_html=True)
        
        if records:
            # Filter only harassments or stress for alert logs
            alerts = [r for r in records if r["final_result"] in ["harassment", "stress"]]
            if alerts:
                df_alerts = pd.DataFrame(alerts)[["timestamp", "face_emotion", "speech_emotion", "text_prediction", "final_result"]]
                # Apply color styling
                def highlight_alerts(s):
                    if s.final_result == 'harassment': return ['background-color: rgba(255, 75, 75, 0.2)'] * len(s)
                    elif s.final_result == 'stress': return ['background-color: rgba(255, 164, 33, 0.2)'] * len(s)
                    return [''] * len(s)
                
                st.dataframe(df_alerts.style.apply(highlight_alerts, axis=1), use_container_width=True, hide_index=True)
            else:
                st.success("No alerts generated yet. All systems normal.")
        
        # ── 3. PROOF HISTORY MODULE ──────────────────────────────────────────
        st.markdown('<div class="section-header">📁 Proof History Module</div>', unsafe_allow_html=True)
        
        if records:
            # Only show records that actually have proof saved
            proofs = [r for r in records if r.get("proof_image") or r.get("proof_audio")]
            if proofs:
                for p in proofs[:10]: # show latest 10 proofs
                    with st.expander(f"Proof Evidence — {p['timestamp']} [{p['final_result'].upper()}]"):
                        pcol1, pcol2 = st.columns([1, 2])
                        with pcol1:
                            if p.get("proof_image") and os.path.exists(p["proof_image"]):
                                st.image(p["proof_image"], caption=f"Face: {p['face_emotion']}", use_container_width=True)
                            else:
                                st.caption("No image proof available.")
                        with pcol2:
                            if p.get("proof_audio") and os.path.exists(p["proof_audio"]):
                                st.audio(p["proof_audio"], format="audio/wav")
                            else:
                                st.caption("No audio proof available.")
                            st.markdown(f"**Text Spoken:** {p['text_prediction']}")
            else:
                st.info("No proof evidence saved yet (Proof is only saved during harassment events).")

        # ── 4. STATISTICS & MATPLOTLIB CHARTS ────────────────────────────────
        st.markdown('<div class="section-header">📊 Matplotlib Analytics</div>', unsafe_allow_html=True)
        
        if stats:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0e1117")
            ax.set_facecolor("#0e1117")
            
            COLORS = {"harassment": "#ff4b4b", "stress": "#ffa421", "normal behaviour": "#2eeb7a"}
            labels = list(stats.keys())
            sizes = list(stats.values())
            colors = [COLORS.get(l, "#6366f1") for l in labels]
            
            ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors, textprops={'color':"w"})
            ax.set_title("Overall Incident Distribution", color="white", pad=20)
            
            st.pyplot(fig, use_container_width=False)

while True:
    render_dashboard()
    time.sleep(3)
