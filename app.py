"""
New Product Profit Prediction — Streamlit Web Application
Assignment 2: Model to Web Application

Loads the pre-trained preprocessing + Linear Regression pipeline (model.pkl)
and predicts First_Year_Profit for a new product launch based on
manager-entered inputs.
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="New Product Profit Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

CATEGORY_OPTIONS = [
    "Consumer Electronics", "Smart Home", "Health & Fitness", "Wearable Tech",
    "Personal Audio", "Kitchen Appliances", "Gaming Gear", "Office Equipment",
    "Outdoor Tech", "Home Automation",
]
REGION_OPTIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', 'Poppins', sans-serif;
}

/* ================================================
   GLOBAL BACKGROUND — Deep Plum / Midnight / Navy
   ================================================ */
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(190, 24, 93, 0.22) 0%, transparent 45%),
        radial-gradient(circle at 85% 0%, rgba(124, 58, 237, 0.22) 0%, transparent 45%),
        radial-gradient(circle at 80% 90%, rgba(236, 72, 153, 0.18) 0%, transparent 50%),
        radial-gradient(circle at 10% 95%, rgba(251, 146, 60, 0.14) 0%, transparent 45%),
        linear-gradient(135deg, #1a0b1f 0%, #1e1033 35%, #16103b 70%, #0f172a 100%);
    min-height: 100vh;
    color: #f1f5f9;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
    background-image:
        radial-gradient(circle at 20% 30%, rgba(251, 113, 133, 0.04) 0%, transparent 2px),
        radial-gradient(circle at 70% 60%, rgba(251, 191, 36, 0.04) 0%, transparent 2px);
    background-size: 40px 40px;
}

[data-testid="stAppViewContainer"] > section[data-testid="stMain"] {
    padding: 1.2rem 2.2rem 3rem 1rem;
    z-index: 1;
    position: relative;
}

/* ================================================
   SIDEBAR PANEL — Dark Eggplant
   ================================================ */
[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #1a0a24 0%, #20113a 50%, #18092e 100%);
    border-right: 1px solid rgba(244, 114, 182, 0.15);
    box-shadow: 6px 0 40px rgba(0, 0, 0, 0.45);
    position: relative;
    overflow: hidden;
}

[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #fb7185, #f59e0b, #ec4899, #f97316);
    background-size: 300% 100%;
    animation: gradientShift 6s ease infinite;
}

[data-testid="stSidebarUserContent"] {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

/* Sidebar floating decorative orbs */
[data-testid="stSidebar"]::after {
    content: '';
    position: absolute;
    width: 220px; height: 220px;
    top: 80px; right: -80px;
    background: radial-gradient(circle, rgba(236, 72, 153, 0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
}

/* ================================================
   SIDEBAR HEADER
   ================================================ */
.sidebar-header {
    position: relative;
    z-index: 1;
    text-align: center;
    padding: 0.4rem 0.3rem 1.4rem 0.3rem;
    margin-bottom: 1rem;
    border-bottom: 1px dashed rgba(251, 113, 133, 0.25);
}

.sidebar-emoji {
    font-size: 3rem;
    display: inline-block;
    margin-bottom: 0.6rem;
    filter: drop-shadow(0 0 18px rgba(251, 146, 60, 0.5));
    animation: float 5s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0) rotate(-4deg); }
    50% { transform: translateY(-8px) rotate(4deg); }
}

.sidebar-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 1.45rem;
    line-height: 1.15;
    margin: 0 0 0.35rem 0;
    background: linear-gradient(135deg, #fb7185 0%, #f97316 40%, #f472b6 75%, #fbbf24 100%);
    background-size: 280% 280%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 7s ease infinite;
    letter-spacing: -0.01em;
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

.sidebar-tagline {
    font-size: 0.78rem;
    color: #fda4af;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin: 0;
}

/* ================================================
   SECTION HEADER (inside sidebar) — matching reference
   ================================================ */
.sb-section {
    position: relative;
    z-index: 1;
    margin: 1.3rem 0;
}

.sb-section-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0 0 0.75rem 0.25rem;
}

.sb-section-icon {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.05rem;
    background: linear-gradient(135deg, rgba(251, 113, 133, 0.25), rgba(245, 158, 11, 0.22));
    border: 1px solid rgba(251, 113, 133, 0.4);
    box-shadow: 0 0 14px rgba(251, 113, 133, 0.2), inset 0 1px 0 rgba(255,255,255,0.08);
}

.sb-section-label {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 0.92rem;
    background: linear-gradient(135deg, #fecdd3, #fdba74);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.2px;
}

.sb-section-sub {
    font-size: 0.7rem;
    color: rgba(253, 186, 116, 0.65);
    font-weight: 500;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    margin-left: 0.1rem;
}

/* ================================================
   INPUT CARDS (Cream / Peach) — reference-style white cards inside dark sidebar
   ================================================ */
.input-card {
    background: linear-gradient(145deg, #fff7ed 0%, #fff1f2 100%);
    border-radius: 18px;
    padding: 1rem 1.05rem 1.05rem 1.05rem;
    margin-bottom: 0.85rem;
    border: 1px solid rgba(254, 205, 211, 0.55);
    box-shadow:
        0 4px 14px rgba(0, 0, 0, 0.22),
        0 0 0 1px rgba(255, 255, 255, 0.5) inset,
        0 -1px 0 rgba(251, 113, 133, 0.15) inset;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.input-card::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(251, 113, 133, 0.05), transparent);
    transition: left 0.7s ease;
    pointer-events: none;
}

.input-card:hover {
    transform: translateY(-2px);
    box-shadow:
        0 8px 22px rgba(0, 0, 0, 0.28),
        0 0 0 1px rgba(255, 255, 255, 0.6) inset,
        0 0 22px rgba(251, 113, 133, 0.18);
    border-color: rgba(251, 113, 133, 0.6);
}

.input-card:hover::after { left: 100%; }

/* Override input text color for light cards */
.sb-label,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSlider label {
    color: #7c2d12 !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    margin-bottom: 0.3rem !important;
}

[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input {
    background: #ffffff !important;
    border: 1.5px solid rgba(251, 113, 133, 0.25) !important;
    border-radius: 12px !important;
    color: #431407 !important;
    font-weight: 500 !important;
    transition: all 0.25s ease !important;
    box-shadow: inset 0 1px 2px rgba(124, 45, 18, 0.06) !important;
    min-height: 40px;
}

[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div:hover,
[data-testid="stSidebar"] .stNumberInput input:hover,
[data-testid="stSidebar"] .stTextInput input:hover {
    border-color: rgba(251, 113, 133, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(251, 113, 133, 0.1), inset 0 1px 2px rgba(124, 45, 18, 0.06) !important;
}

[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]:focus-within > div,
[data-testid="stSidebar"] .stNumberInput:focus-within input,
[data-testid="stSidebar"] .stTextInput:focus-within input {
    border-color: rgba(249, 115, 22, 0.7) !important;
    box-shadow:
        0 0 0 4px rgba(249, 115, 22, 0.14),
        0 0 14px rgba(251, 113, 133, 0.18),
        inset 0 1px 2px rgba(124, 45, 18, 0.06) !important;
}

/* Slider in sidebar */
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stTickBar"] > div {
    background: rgba(124, 45, 18, 0.2) !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] > div > div > div {
    background: linear-gradient(90deg, #fb7185, #f97316) !important;
}
[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] > div > div {
    background: linear-gradient(135deg, #fb7185, #f59e0b) !important;
    box-shadow: 0 0 10px rgba(251, 113, 133, 0.55) !important;
}

/* Reduce Streamlit vertical spacing in sidebar */
[data-testid="stSidebar"] .stNumberInput,
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stSlider {
    margin-bottom: 0.35rem !important;
}
[data-testid="stSidebar"] .stNumberInput > div,
[data-testid="stSidebar"] .stSelectbox > div,
[data-testid="stSidebar"] .stSlider > div {
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
}

/* ================================================
   PREDICT BUTTON — Coral → Amber → Rose gradient
   ================================================ */
.sb-predict-btn {
    margin-top: 1.2rem;
    position: relative;
    z-index: 1;
}

[data-testid="stSidebar"] .stFormSubmitButton > button,
[data-testid="stSidebar"] button[kind="primary"] {
    position: relative !important;
    width: 100% !important;
    padding: 0.95rem 1.2rem !important;
    font-family: 'Poppins', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    color: #fffbeb !important;
    letter-spacing: 0.5px !important;
    background: linear-gradient(135deg, #f97316 0%, #fb7185 35%, #ec4899 70%, #f472b6 100%) !important;
    background-size: 220% 220% !important;
    border: none !important;
    border-radius: 16px !important;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow:
        0 6px 20px rgba(251, 113, 133, 0.35),
        0 0 30px rgba(236, 72, 153, 0.18),
        inset 0 1px 0 rgba(255,255,255,0.2) !important;
    overflow: hidden !important;
    animation: gradientShift 4s ease infinite !important;
}

[data-testid="stSidebar"] .stFormSubmitButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.65s ease;
}

[data-testid="stSidebar"] .stFormSubmitButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow:
        0 14px 36px rgba(251, 113, 133, 0.5),
        0 0 60px rgba(236, 72, 153, 0.28),
        inset 0 1px 0 rgba(255,255,255,0.25) !important;
}
[data-testid="stSidebar"] .stFormSubmitButton > button:hover::before { left: 100%; }

[data-testid="stSidebar"] .stFormSubmitButton > button:active {
    transform: translateY(0) scale(0.97) !important;
}

/* ================================================
   MAIN AREA HERO BANNER
   ================================================ */
.hero-banner {
    position: relative;
    padding: 2.4rem 2.8rem 2.8rem 2.8rem;
    border-radius: 28px;
    overflow: hidden;
    margin-bottom: 1.8rem;
    background:
        linear-gradient(135deg, rgba(40, 14, 59, 0.85) 0%, rgba(55, 20, 90, 0.7) 45%, rgba(76, 29, 149, 0.6) 100%);
    backdrop-filter: blur(26px) saturate(180%);
    -webkit-backdrop-filter: blur(26px) saturate(180%);
    border: 1px solid rgba(251, 113, 133, 0.28);
    box-shadow:
        0 0 70px rgba(236, 72, 153, 0.18),
        0 0 140px rgba(124, 58, 237, 0.15),
        inset 0 1px 0 rgba(255,255,255,0.07);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -60%; left: -20%;
    width: 60%; height: 160%;
    background: radial-gradient(ellipse, rgba(251, 113, 133, 0.18) 0%, transparent 65%);
    transform: rotate(-15deg);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -50%; right: -10%;
    width: 55%; height: 160%;
    background: radial-gradient(ellipse, rgba(249, 115, 22, 0.15) 0%, transparent 65%);
    transform: rotate(20deg);
    pointer-events: none;
}

.hero-top-row {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 1.3rem;
    position: relative;
    z-index: 1;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.95rem;
    background: linear-gradient(135deg, rgba(251, 113, 133, 0.25), rgba(245, 158, 11, 0.22));
    border: 1px solid rgba(251, 113, 133, 0.45);
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    color: #fecaca;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    backdrop-filter: blur(12px);
}

.hero-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #fb923c;
    box-shadow: 0 0 10px #fb923c;
    animation: pulseDot 1.8s ease-in-out infinite;
}
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.3); }
}

.hero-big {
    position: relative;
    z-index: 1;
    font-family: 'Poppins', sans-serif;
    font-weight: 900;
    font-size: 3.1rem;
    line-height: 1.08;
    margin: 0 0 1rem 0;
    letter-spacing: -0.025em;
}
.hero-big .part1 {
    background: linear-gradient(135deg, #fecdd3 0%, #fff7ed 40%, #fecaca 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
}
.hero-big .part2 {
    background: linear-gradient(135deg, #fb7185 0%, #f59e0b 40%, #ec4899 75%, #fbbf24 100%);
    background-size: 280% 280%;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientShift 6s ease infinite;
    display: inline;
}

.hero-sub {
    position: relative;
    z-index: 1;
    font-size: 1.05rem;
    line-height: 1.7;
    color: rgba(254, 215, 170, 0.85);
    max-width: 680px;
    font-weight: 400;
    margin: 0;
}
.hero-sub strong {
    background: linear-gradient(135deg, #fdba74, #f9a8d4);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* ================================================
   INFO CARDS (Reference-style teal row → our coral row)
   ================================================ */
.info-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 1.8rem;
}

.info-card {
    background: linear-gradient(145deg, rgba(41, 13, 64, 0.7), rgba(30, 12, 59, 0.65));
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    border: 1px solid rgba(251, 113, 133, 0.22);
    border-radius: 20px;
    padding: 1.1rem 1.15rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    position: relative;
    overflow: hidden;
}

.info-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(251, 113, 133, 0.5), transparent);
    opacity: 0.6;
}

.info-card:hover {
    transform: translateY(-3px);
    border-color: rgba(251, 113, 133, 0.5);
    box-shadow: 0 12px 30px rgba(0,0,0,0.35), 0 0 30px rgba(251, 113, 133, 0.18);
}

.info-icon-wrap {
    flex-shrink: 0;
    width: 48px; height: 48px;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, rgba(251, 113, 133, 0.28), rgba(245, 158, 11, 0.22));
    border: 1px solid rgba(251, 113, 133, 0.45);
    font-size: 1.4rem;
    box-shadow: 0 0 16px rgba(251, 113, 133, 0.25), inset 0 1px 0 rgba(255,255,255,0.1);
}

.info-body {
    flex: 1;
    min-width: 0;
}

.info-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: rgba(254, 205, 211, 0.7);
    margin-bottom: 0.28rem;
}
.info-value {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 1.15rem;
    line-height: 1.2;
    background: linear-gradient(135deg, #fff7ed 0%, #fecaca 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ================================================
   RESULT DISPLAY — Reference-style big card + outline icons
   ================================================ */
.result-hero {
    position: relative;
    padding: 3rem 2.5rem 2.6rem 2.5rem;
    border-radius: 28px;
    overflow: hidden;
    text-align: center;
    background:
        linear-gradient(145deg, rgba(45, 16, 70, 0.88) 0%, rgba(58, 19, 97, 0.8) 50%, rgba(30, 12, 59, 0.9) 100%);
    backdrop-filter: blur(30px) saturate(200%);
    -webkit-backdrop-filter: blur(30px) saturate(200%);
    border: 1px solid rgba(251, 113, 133, 0.32);
    box-shadow:
        0 0 90px rgba(236, 72, 153, 0.22),
        0 22px 60px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.07);
    animation: resultIn 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    margin-bottom: 1.5rem;
}

@keyframes resultIn {
    0% { opacity: 0; transform: scale(0.92) translateY(24px); }
    100% { opacity: 1; transform: scale(1) translateY(0); }
}

.result-hero::before {
    content: '';
    position: absolute;
    top: -3px; left: -3px; right: -3px; bottom: -3px;
    background: linear-gradient(135deg, #fb7185, #f59e0b, #ec4899, #f97316, #fb7185);
    background-size: 400% 400%;
    border-radius: 30px;
    z-index: -1;
    opacity: 0.55;
    filter: blur(10px);
    animation: gradientShift 5s ease infinite;
}

.result-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.42rem 1.1rem;
    border-radius: 999px;
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 0.76rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(14px);
}
.result-tag.profit {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.25), rgba(245, 158, 11, 0.22));
    border: 1px solid rgba(251, 191, 36, 0.5);
    color: #fde68a;
}
.result-tag.loss {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(220, 38, 38, 0.2));
    border: 1px solid rgba(248, 113, 113, 0.5);
    color: #fecaca;
}

.result-figure {
    font-family: 'Poppins', sans-serif;
    font-weight: 900;
    font-size: 4.4rem;
    line-height: 1;
    margin: 0 0 0.8rem 0;
    letter-spacing: -0.03em;
}
.result-figure.profit {
    background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 25%, #fb923c 55%, #fcd34d 80%, #fde68a 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 30px rgba(251, 191, 36, 0.4));
    animation: valuePop 1.1s cubic-bezier(0.34, 1.56, 0.64, 1) 0.15s both;
}
.result-figure.loss {
    background: linear-gradient(135deg, #f87171 0%, #ef4444 30%, #dc2626 60%, #fca5a5 100%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 30px rgba(239, 68, 68, 0.35));
    animation: valuePop 1.1s cubic-bezier(0.34, 1.56, 0.64, 1) 0.15s both;
}
@keyframes valuePop {
    0% { opacity: 0; transform: scale(0.55); filter: blur(22px); }
    60% { opacity: 1; transform: scale(1.09); filter: blur(0); }
    100% { transform: scale(1); }
}

.result-caption {
    font-size: 1.1rem;
    font-weight: 500;
    color: rgba(254, 215, 170, 0.9);
    margin: 0 0 2rem 0;
}

/* 3 outlined icon stats like the reference */
.stats-triplet {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.1rem;
}

.stat-pill {
    padding: 1.1rem 1rem;
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(30, 12, 59, 0.7), rgba(20, 10, 48, 0.6));
    border: 1px solid rgba(251, 113, 133, 0.22);
    backdrop-filter: blur(14px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.stat-pill::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(251, 113, 133, 0.06), transparent);
    transition: left 0.7s ease;
}
.stat-pill:hover {
    transform: translateY(-3px);
    border-color: rgba(251, 113, 133, 0.5);
    box-shadow: 0 10px 28px rgba(0,0,0,0.35), 0 0 26px rgba(251, 113, 133, 0.18);
}
.stat-pill:hover::after { left: 100%; }

.stat-top {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.65rem;
}

.stat-circle {
    flex-shrink: 0;
    width: 44px; height: 44px;
    border-radius: 50%;
    border: 2px solid rgba(251, 113, 133, 0.55);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    background: radial-gradient(circle, rgba(251, 113, 133, 0.18), rgba(251, 113, 133, 0.04));
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06), 0 0 14px rgba(251, 113, 133, 0.2);
}

.stat-k {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
    color: rgba(253, 186, 116, 0.85);
    margin-bottom: 0.2rem;
}
.stat-v {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    line-height: 1.1;
    background: linear-gradient(135deg, #fff7ed, #fecdd3);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ================================================
   EXPANDERS — reference styled, coral palette
   ================================================ */
.streamlit-expanderHeader {
    background: linear-gradient(145deg, rgba(40, 14, 59, 0.75), rgba(25, 10, 48, 0.7)) !important;
    backdrop-filter: blur(16px);
    border: 1px solid rgba(251, 113, 133, 0.28) !important;
    border-radius: 18px !important;
    padding: 1rem 1.25rem !important;
    transition: all 0.3s ease !important;
    color: #fef3c7 !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    margin-bottom: 0.8rem !important;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(251, 113, 133, 0.55) !important;
    box-shadow: 0 0 26px rgba(251, 113, 133, 0.2) !important;
    transform: translateY(-1px);
}
[data-testid="stExpander"] {
    border: none !important;
}
[data-testid="stExpanderDetails"] {
    background: rgba(20, 10, 48, 0.5) !important;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(251, 113, 133, 0.18) !important;
    border-top: none !important;
    border-radius: 0 0 18px 18px !important;
    padding: 1.25rem !important;
    margin: -0.8rem 0 1rem 0 !important;
    color: #fef3c7 !important;
    line-height: 1.75;
}
[data-testid="stExpanderDetails"] strong {
    color: #fdba74 !important;
    font-weight: 800 !important;
}
[data-testid="stExpanderDetails"] li {
    color: #fed7aa !important;
}

/* ================================================
   ALERTS — styled like input cards
   ================================================ */
[data-testid="stAlertContainer"] {
    margin-bottom: 1rem !important;
}
[data-testid="stAlert"] {
    background: linear-gradient(145deg, #fff1f2, #fff7ed) !important;
    backdrop-filter: blur(14px) !important;
    border: 1.5px solid rgba(239, 68, 68, 0.4) !important;
    border-radius: 16px !important;
    box-shadow: 0 0 24px rgba(239, 68, 68, 0.15) !important;
    color: #7f1d1d !important;
    font-weight: 500 !important;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    color: #7f1d1d !important;
}
[data-testid="stAlert"][kind="warning"] {
    background: linear-gradient(145deg, #fffbeb, #fff7ed) !important;
    border-color: rgba(245, 158, 11, 0.45) !important;
    color: #78350f !important;
}
[data-testid="stAlert"][kind="warning"] [data-testid="stMarkdownContainer"] {
    color: #78350f !important;
}

/* ================================================
   DATAFRAME
   ================================================ */
[data-testid="stDataFrame"] {
    background: rgba(30, 12, 59, 0.6) !important;
    border-radius: 16px !important;
    border: 1px solid rgba(251, 113, 133, 0.25) !important;
    overflow: hidden !important;
    box-shadow: 0 6px 22px rgba(0,0,0,0.35) !important;
}

/* ================================================
   TOP BAR / HEADER
   ================================================ */
[data-testid="stHeader"] {
    background: rgba(15, 8, 32, 0.55) !important;
    backdrop-filter: blur(18px);
    border-bottom: 1px solid rgba(251, 113, 133, 0.12);
}
[data-testid="stToolbar"] { right: 1rem; top: 0.75rem; }

/* ================================================
   FOOTER / SIGN-OFF CARD
   ================================================ */
.foot-card {
    margin-top: 2rem;
    padding: 1.3rem 1.8rem;
    text-align: center;
    border-radius: 20px;
    background: linear-gradient(145deg, rgba(25, 10, 48, 0.7), rgba(15, 8, 32, 0.65));
    backdrop-filter: blur(18px);
    border: 1px solid rgba(251, 113, 133, 0.18);
    color: rgba(253, 186, 116, 0.8);
    font-size: 0.85rem;
    line-height: 1.6;
    box-shadow: 0 0 30px rgba(236, 72, 153, 0.08);
}
.foot-card strong {
    background: linear-gradient(135deg, #fb7185, #f59e0b);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* ================================================
   SCROLLBAR
   ================================================ */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: rgba(26, 11, 31, 0.6); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #fb7185, #f97316);
    border-radius: 5px;
}
::-webkit-scrollbar-thumb:hover { background: linear-gradient(180deg, #fda4af, #fdba74); }

[data-testid="stSidebar"] ::-webkit-scrollbar { width: 8px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-track { background: rgba(26, 10, 36, 0.4); }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #ec4899, #fb923c);
    border-radius: 4px;
}

/* ================================================
   RESPONSIVE
   ================================================ */
@media (max-width: 1024px) {
    .info-strip { grid-template-columns: repeat(2, 1fr); }
    .hero-big { font-size: 2.3rem; }
}
@media (max-width: 768px) {
    .info-strip { grid-template-columns: 1fr; }
    .stats-triplet { grid-template-columns: 1fr; }
    .hero-big { font-size: 1.8rem; }
    .result-figure { font-size: 2.6rem !important; }
    .hero-banner { padding: 1.8rem 1.4rem; }
    .result-hero { padding: 2rem 1.2rem; }
}
</style>
""", unsafe_allow_html=True)

# ================================================
# SIDEBAR — All inputs (matching reference split layout)
# ================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-emoji">💸</div>
        <h2 class="sidebar-title">Profit Oracle</h2>
        <p class="sidebar-tagline">Launch Intelligence Suite</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        # --- Section 1: Product & Market ---
        st.markdown("""
        <div class="sb-section">
            <div class="sb-section-sub">Step 01</div>
            <div class="sb-section-title">
                <span class="sb-section-icon">🎯</span>
                <span class="sb-section-label">Product &amp; Market</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        product_category = st.selectbox("Product Category", CATEGORY_OPTIONS, index=0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        launch_region = st.selectbox("Launch Region", REGION_OPTIONS, index=0)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Section 2: Cost Inputs ---
        st.markdown("""
        <div class="sb-section">
            <div class="sb-section-sub">Step 02</div>
            <div class="sb-section-title">
                <span class="sb-section-icon">💲</span>
                <span class="sb-section-label">Cost &amp; Pricing</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        development_cost = st.number_input(
            "Development Cost ($)", min_value=0.0, max_value=5_000_000.0,
            value=150_000.0, step=5_000.0,
            help="Total R&D / engineering cost to develop the product.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        launch_marketing_spend = st.number_input(
            "Launch Marketing Spend ($)", min_value=0.0, max_value=5_000_000.0,
            value=100_000.0, step=5_000.0,
            help="Total marketing budget for the first-year launch.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        unit_production_cost = st.number_input(
            "Unit Production Cost ($)", min_value=0.01, max_value=100_000.0,
            value=60.0, step=1.0,
            help="Cost to manufacture a single unit.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        unit_selling_price = st.number_input(
            "Unit Selling Price ($)", min_value=0.01, max_value=100_000.0,
            value=150.0, step=1.0,
            help="Retail price per unit.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Section 3: Demand & Reach ---
        st.markdown("""
        <div class="sb-section">
            <div class="sb-section-sub">Step 03</div>
            <div class="sb-section-title">
                <span class="sb-section-icon">📈</span>
                <span class="sb-section-label">Demand &amp; Reach</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        first_year_units_sold = st.number_input(
            "First-Year Units Sold", min_value=0, max_value=10_000_000,
            value=15_000, step=500,
            help="Forecasted number of units sold in the first year.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        customer_rating = st.slider(
            "Expected Customer Rating", min_value=1.0, max_value=5.0,
            value=4.3, step=0.1,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        distribution_coverage = st.slider(
            "Distribution Coverage", min_value=0.0, max_value=1.0,
            value=0.80, step=0.01,
            help="Fraction of target retail/online channels the product will be available in.",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-predict-btn">', unsafe_allow_html=True)
        submitted = st.form_submit_button("✨  PREDICT PROFIT", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ================================================
# MAIN AREA — Hero banner + Info strip + Result
# ================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-top-row">
        <div class="hero-badge">
            <span class="hero-dot"></span>
            AI-DRIVEN · ML MODEL LIVE
        </div>
    </div>
    <h1 class="hero-big">
        <span class="part1">Smart </span><span class="part2">Profit Forecaster</span>
    </h1>
    <p class="hero-sub">
        Configure your launch plan in the sidebar and the system will estimate the
        <strong>First-Year Profit</strong> using a trained Linear Regression pipeline
        built on thousands of historical product launches.
    </p>
</div>
""", unsafe_allow_html=True)

# Quick-info strip (4 reference-style cards)
st.markdown(f"""
<div class="info-strip">
    <div class="info-card">
        <div class="info-icon-wrap">🧪</div>
        <div class="info-body">
            <div class="info-label">Model</div>
            <div class="info-value">Linear Regression</div>
        </div>
    </div>
    <div class="info-card">
        <div class="info-icon-wrap">🏆</div>
        <div class="info-body">
            <div class="info-label">Test R²</div>
            <div class="info-value">94.3%</div>
        </div>
    </div>
    <div class="info-card">
        <div class="info-icon-wrap">📦</div>
        <div class="info-body">
            <div class="info-label">Product</div>
            <div class="info-value" style="font-size:0.92rem;line-height:1.25;">{product_category}</div>
        </div>
    </div>
    <div class="info-card">
        <div class="info-icon-wrap">🌍</div>
        <div class="info-body">
            <div class="info-label">Region</div>
            <div class="info-value" style="font-size:0.95rem;">{launch_region}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================
# VALIDATION + PREDICTION
# ================================================
if submitted:
    errors = []

    if unit_selling_price <= unit_production_cost:
        errors.append(
            "Unit Selling Price must be greater than Unit Production Cost "
            "(a product cannot be profitably sold below its production cost)."
        )
    if unit_production_cost <= 0:
        errors.append("Unit Production Cost must be a positive number.")
    if first_year_units_sold < 0:
        errors.append("First-Year Units Sold cannot be negative.")
    if development_cost < 0 or launch_marketing_spend < 0:
        errors.append("Development Cost and Marketing Spend cannot be negative.")
    if not (0 <= distribution_coverage <= 1):
        errors.append("Distribution Coverage must be between 0 and 1.")
    if not (1 <= customer_rating <= 5):
        errors.append("Customer Rating must be between 1 and 5.")

    if errors:
        for e in errors:
            st.error(e)
    else:
        unit_margin = unit_selling_price - unit_production_cost
        margin_x_units = unit_margin * first_year_units_sold
        upfront = development_cost + launch_marketing_spend

        input_df = pd.DataFrame([{
            "Development_Cost": development_cost,
            "Launch_Marketing_Spend": launch_marketing_spend,
            "Unit_Margin": unit_margin,
            "Margin_x_Units": margin_x_units,
            "Distribution_Coverage": distribution_coverage,
            "Customer_Rating": customer_rating,
            "Product_Category": product_category,
            "Launch_Region": launch_region,
        }])

        prediction = model.predict(input_df)[0]
        is_profit = prediction >= 0
        display_value = f"${prediction:,.0f}" if is_profit else f"-${abs(prediction):,.0f}"
        tag_cls = "profit" if is_profit else "loss"
        fig_cls = "profit" if is_profit else "loss"
        tag_text = "✓ PROFIT PROJECTION" if is_profit else "⚠  LOSS PROJECTION"

        st.markdown(f"""
        <div class="result-hero">
            <div class="result-tag {tag_cls}">
                <span class="hero-dot"></span>
                {tag_text}
            </div>
            <div class="result-figure {fig_cls}">{display_value}</div>
            <div class="result-caption">Estimated First-Year Profit</div>

            <div class="stats-triplet">
                <div class="stat-pill">
                    <div class="stat-top">
                        <div class="stat-circle">📊</div>
                    </div>
                    <div class="stat-k">Unit Margin</div>
                    <div class="stat-v">${unit_margin:,.2f}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-top">
                        <div class="stat-circle">📦</div>
                    </div>
                    <div class="stat-k">Sales Contribution</div>
                    <div class="stat-v">${margin_x_units:,.0f}</div>
                </div>
                <div class="stat-pill">
                    <div class="stat-top">
                        <div class="stat-circle">🚀</div>
                    </div>
                    <div class="stat-k">Upfront Investment</div>
                    <div class="stat-v">${upfront:,.0f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not is_profit:
            st.warning(
                "The model predicts a first-year **loss** for this configuration. "
                "Consider adjusting pricing, marketing spend, or volume targets."
            )

        with st.expander("💡  What does this mean for a Product Manager?"):
            st.markdown(
                f"""
                Based on the inputs provided, this product is expected to generate
                approximately **${prediction:,.0f}** in profit during its first year on the market.

                - **Unit margin** (selling price − production cost): **${unit_margin:,.2f} per unit**
                - **Total contribution from sales volume** (margin × units sold): **${margin_x_units:,.0f}**
                - Upfront investment (development + marketing): **${upfront:,.0f}**

                A manager can use this estimate to compare **different pricing, marketing, or
                distribution scenarios** before committing to a launch plan — for example,
                testing whether a higher marketing spend or wider distribution coverage
                would be justified by the resulting profit uplift.

                *Note: this is a statistical estimate based on historical launches with
                similar characteristics, not a guarantee of actual performance.*
                """
            )

        with st.expander("🔍  See the exact inputs sent to the model"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

st.markdown("""
<div class="foot-card">
    <strong>Model:</strong> Linear Regression pipeline (StandardScaler + OneHotEncoder) trained on
    historical New Product launches · Assignment 2 — AI Spreadsheets &amp; Python Programming
</div>
""", unsafe_allow_html=True)
