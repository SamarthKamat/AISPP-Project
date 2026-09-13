"""
New Product Profit Prediction — Streamlit Web Application
Assignment 2: Model to Web Application

Loads the pre-trained preprocessing + Linear Regression pipeline (model.pkl)
and predicts First_Year_Profit for a new product launch based on
manager-entered inputs.
"""

import pickle
import textwrap
import numpy as np
import pandas as pd
import streamlit as st


def flat_html(s):
    return "\n".join(line.lstrip() for line in s.split("\n")).strip()


st.set_page_config(
    page_title="Profit Oracle · Smart Profit Forecaster",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
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

st.markdown(flat_html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    * {
      font-family: 'Inter', 'Poppins', 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    .stApp {
      background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(139, 92, 246, 0.35) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 0%, rgba(236, 72, 153, 0.25) 0%, transparent 60%),
        radial-gradient(ellipse 70% 60% at 50% 120%, rgba(59, 130, 246, 0.22) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 90% 90%, rgba(34, 211, 238, 0.18) 0%, transparent 60%),
        linear-gradient(135deg, #0a0618 0%, #120a28 25%, #0f0820 50%, #180a2e 75%, #0b051a 100%);
      min-height: 100vh;
      color: #e5e7eb;
      position: relative;
      overflow-x: hidden;
    }

    .stApp::before {
      content: '';
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      background-image:
        linear-gradient(rgba(167, 139, 250, 0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(167, 139, 250, 0.04) 1px, transparent 1px);
      background-size: 52px 52px;
      animation: gridDrift 20s linear infinite;
    }
    @keyframes gridDrift {
      0% { background-position: 0 0; }
      100% { background-position: 52px 52px; }
    }

    .floating-orb {
      position: fixed;
      border-radius: 50%;
      pointer-events: none;
      z-index: 1;
      filter: blur(60px);
      opacity: 0.5;
    }
    .orb-1 {
      width: 400px; height: 400px;
      background: radial-gradient(circle, rgba(139, 92, 246, 0.6) 0%, transparent 70%);
      top: -10%; left: -5%;
      animation: floatOrb1 18s ease-in-out infinite;
    }
    .orb-2 {
      width: 350px; height: 350px;
      background: radial-gradient(circle, rgba(236, 72, 153, 0.5) 0%, transparent 70%);
      top: 30%; right: -8%;
      animation: floatOrb2 22s ease-in-out infinite;
    }
    .orb-3 {
      width: 320px; height: 320px;
      background: radial-gradient(circle, rgba(34, 211, 238, 0.4) 0%, transparent 70%);
      bottom: -10%; left: 30%;
      animation: floatOrb3 25s ease-in-out infinite;
    }
    @keyframes floatOrb1 {
      0%, 100% { transform: translate(0, 0) scale(1); }
      33% { transform: translate(40px, 60px) scale(1.1); }
      66% { transform: translate(-30px, 40px) scale(0.95); }
    }
    @keyframes floatOrb2 {
      0%, 100% { transform: translate(0, 0) scale(1); }
      33% { transform: translate(-50px, 30px) scale(0.9); }
      66% { transform: translate(30px, -40px) scale(1.15); }
    }
    @keyframes floatOrb3 {
      0%, 100% { transform: translate(0, 0) scale(1); }
      50% { transform: translate(60px, -50px) scale(1.1); }
    }

    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stAppViewContainer"] > section[data-testid="stMain"] {
      padding: 1.8rem 2.8rem 4rem 2.8rem !important;
      z-index: 5;
      position: relative;
      overflow-x: hidden;
    }
    [data-testid="stAppViewContainer"] > section[data-testid="stMain"] > [data-testid="block-container"] {
      padding-top: 0 !important;
      max-width: 1520px;
    }

    [data-testid="stHeader"] {
      background: linear-gradient(180deg, rgba(10, 6, 24, 0.96) 0%, rgba(10, 6, 24, 0.4) 100%) !important;
      backdrop-filter: saturate(200%) blur(28px);
      -webkit-backdrop-filter: saturate(200%) blur(28px);
      border-bottom: 1px solid rgba(167, 139, 250, 0.12);
      height: 66px;
      z-index: 50;
    }
    [data-testid="stToolbar"] { right: 1.25rem; top: 1rem; }

    .app-hero-wrap {
      position: relative;
      z-index: 6;
      animation: fadeSlideDown 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.05s both;
    }
    @keyframes fadeSlideDown {
      0% { opacity: 0; transform: translateY(-24px); }
      100% { opacity: 1; transform: translateY(0); }
    }

    .hero-banner {
      position: relative;
      padding: 3.4rem 3.4rem 3.6rem 3.4rem;
      border-radius: 28px;
      overflow: hidden;
      margin-bottom: 2rem;
      background:
        linear-gradient(145deg, rgba(30, 18, 68, 0.8) 0%, rgba(44, 22, 88, 0.7) 35%, rgba(26, 14, 58, 0.85) 70%, rgba(20, 10, 46, 0.92) 100%);
      backdrop-filter: blur(40px) saturate(200%);
      -webkit-backdrop-filter: blur(40px) saturate(200%);
      border: 1px solid rgba(167, 139, 250, 0.22);
      box-shadow:
        0 40px 100px rgba(0, 0, 0, 0.55),
        0 0 0 1px rgba(255, 255, 255, 0.03) inset,
        0 0 120px rgba(139, 92, 246, 0.18);
    }
    .hero-banner::before {
      content: '';
      position: absolute;
      top: -50%;
      left: -15%;
      width: 65%;
      height: 200%;
      background: radial-gradient(ellipse, rgba(139, 92, 246, 0.28) 0%, transparent 65%);
      transform: rotate(-12deg);
      pointer-events: none;
      animation: glowDrift1 12s ease-in-out infinite;
    }
    .hero-banner::after {
      content: '';
      position: absolute;
      bottom: -55%;
      right: -8%;
      width: 65%;
      height: 200%;
      background: radial-gradient(ellipse, rgba(236, 72, 153, 0.2) 0%, rgba(34, 211, 238, 0.12) 40%, transparent 70%);
      transform: rotate(18deg);
      pointer-events: none;
      animation: glowDrift2 14s ease-in-out infinite;
    }
    @keyframes glowDrift1 {
      0%, 100% { transform: rotate(-12deg) translateX(0); }
      50% { transform: rotate(-8deg) translateX(40px); }
    }
    @keyframes glowDrift2 {
      0%, 100% { transform: rotate(18deg) translateX(0); }
      50% { transform: rotate(14deg) translateX(-30px); }
    }
    .hero-border-top {
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2.5px;
      background: linear-gradient(90deg,
        transparent 0%,
        #6366f1 15%,
        #8b5cf6 30%,
        #a855f7 45%,
        #ec4899 55%,
        #f43f5e 70%,
        #22d3ee 85%,
        transparent 100%
      );
      box-shadow: 0 0 40px rgba(139, 92, 246, 0.55);
      animation: shimmer 4s linear infinite;
      background-size: 200% 100%;
    }
    @keyframes shimmer {
      0% { background-position: 200% 0; }
      100% { background-position: -200% 0; }
    }

    .hero-top-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 1.6rem;
      position: relative;
      z-index: 2;
    }
    .brand-left-main {
      display: flex;
      align-items: center;
      gap: 1.4rem;
    }
    .brand-logo-wrap {
      flex-shrink: 0;
      width: 72px; height: 72px;
      border-radius: 22px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg,
        rgba(99, 102, 241, 0.5) 0%,
        rgba(139, 92, 246, 0.45) 35%,
        rgba(236, 72, 153, 0.4) 70%,
        rgba(34, 211, 238, 0.35) 100%
      );
      border: 1px solid rgba(167, 139, 250, 0.55);
      box-shadow:
        0 16px 40px rgba(99, 102, 241, 0.45),
        0 0 0 1px rgba(255, 255, 255, 0.08) inset,
        0 0 60px rgba(139, 92, 246, 0.25);
      backdrop-filter: blur(14px);
      position: relative;
      overflow: hidden;
      animation: logoBounce 5s ease-in-out infinite;
    }
    @keyframes logoBounce {
      0%, 100% { transform: translateY(0) rotate(0deg); }
      50% { transform: translateY(-4px) rotate(1deg); }
    }
    .brand-logo-wrap::before {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 25%, rgba(255,255,255,0.25) 50%, transparent 75%);
      transform: translateX(-120%);
      animation: logoShine 6s ease-in-out infinite;
    }
    @keyframes logoShine {
      0%, 45%, 100% { transform: translateX(-120%); }
      65% { transform: translateX(120%); }
    }
    .brand-logo { font-size: 2.4rem; line-height: 1; margin: 0; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3)); }
    .brand-text-main { display: flex; flex-direction: column; gap: 0.3rem; }
    .brand-title-main {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 800;
      font-size: 2.2rem;
      line-height: 1.05;
      margin: 0;
      letter-spacing: -0.035em;
      background: linear-gradient(135deg,
        #ffffff 0%,
        #e0e7ff 20%,
        #c4b5fd 40%,
        #f0abfc 60%,
        #a5f3fc 80%,
        #ffffff 100%
      );
      background-size: 300% 300%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: brandGradient 8s ease infinite;
    }
    @keyframes brandGradient {
      0%, 100% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
    }
    .brand-tagline-main {
      font-size: 0.88rem;
      color: rgba(196, 181, 253, 0.95);
      font-weight: 600;
      letter-spacing: 0.04em;
      margin: 0;
    }

    .hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.55rem;
      padding: 0.5rem 1.15rem;
      background: rgba(139, 92, 246, 0.15);
      border: 1px solid rgba(167, 139, 250, 0.4);
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 700;
      color: #c4b5fd;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      backdrop-filter: blur(12px);
      box-shadow: 0 0 30px rgba(139, 92, 246, 0.15);
    }
    .hero-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: linear-gradient(135deg, #8b5cf6, #f472b6);
      box-shadow: 0 0 14px #8b5cf6, 0 0 28px rgba(139, 92, 246, 0.4);
      animation: pulseDotVibrant 1.6s ease-in-out infinite;
    }
    @keyframes pulseDotVibrant {
      0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 14px #8b5cf6, 0 0 28px rgba(139, 92, 246, 0.4); }
      50% { opacity: 0.5; transform: scale(1.55); box-shadow: 0 0 22px #ec4899, 0 0 44px rgba(236, 72, 153, 0.5); }
    }

    .hero-meta {
      display: inline-flex;
      align-items: center;
      gap: 0.9rem;
      font-size: 0.76rem;
      color: rgba(196, 181, 253, 0.9);
      font-weight: 600;
    }
    .hero-meta span { display: inline-flex; align-items: center; gap: 0.35rem; }

    .hero-big {
      position: relative;
      z-index: 2;
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 800;
      font-size: 3.1rem;
      line-height: 1.06;
      margin: 1.6rem 0 1.1rem 0;
      letter-spacing: -0.04em;
      max-width: 860px;
    }
    .hero-big .part1 {
      background: linear-gradient(135deg, #ffffff 0%, #e9d5ff 40%, #ddd6fe 100%);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .hero-big .part2 {
      background: linear-gradient(135deg,
        #818cf8 0%,
        #a78bfa 18%,
        #c084fc 35%,
        #e879f9 52%,
        #f0abfc 68%,
        #22d3ee 85%,
        #67e8f9 100%
      );
      background-size: 320% 320%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: gradientShiftVibrant 6s ease infinite;
      display: inline;
      filter: drop-shadow(0 0 24px rgba(168, 85, 247, 0.25));
    }
    @keyframes gradientShiftVibrant {
      0%, 100% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
    }
    .hero-sub {
      position: relative;
      z-index: 2;
      font-size: 1.05rem;
      line-height: 1.8;
      color: rgba(226, 232, 240, 0.92);
      max-width: 760px;
      font-weight: 400;
      margin: 0;
    }
    .hero-sub strong {
      color: #e0e7ff;
      font-weight: 700;
      background: linear-gradient(135deg, #c4b5fd, #f0abfc);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .info-strip {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1.2rem;
      margin-bottom: 2rem;
      z-index: 6;
      position: relative;
    }
    .info-card {
      background: linear-gradient(145deg, rgba(26, 18, 56, 0.92) 0%, rgba(20, 13, 44, 0.92) 100%);
      backdrop-filter: blur(22px);
      -webkit-backdrop-filter: blur(22px);
      border: 1px solid rgba(167, 139, 250, 0.18);
      border-radius: 22px;
      padding: 1.5rem 1.55rem;
      transition: all 0.4s cubic-bezier(0.22, 1, 0.36, 1);
      display: flex;
      align-items: flex-start;
      gap: 1.1rem;
      position: relative;
      overflow: hidden;
      box-shadow:
        0 16px 40px rgba(0, 0, 0, 0.35),
        0 0 0 1px rgba(255, 255, 255, 0.04) inset,
        0 0 50px rgba(99, 102, 241, 0.06);
      animation: staggerCardIn 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .info-card:nth-child(1) { animation-delay: 0.15s; }
    .info-card:nth-child(2) { animation-delay: 0.25s; }
    @keyframes staggerCardIn {
      0% { opacity: 0; transform: translateY(20px) scale(0.97); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    .info-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(129, 140, 248, 0.8) 25%,
        rgba(192, 132, 252, 0.7) 50%,
        rgba(236, 72, 153, 0.7) 75%,
        transparent 100%
      );
      opacity: 0.9;
    }
    .info-card::after {
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at var(--mx, 50%) var(--my, 0%),
        rgba(139, 92, 246, 0.15) 0%,
        transparent 55%
      );
      opacity: 0;
      transition: opacity 0.4s ease;
      pointer-events: none;
    }
    .info-card:hover {
      transform: translateY(-6px) scale(1.01);
      border-color: rgba(167, 139, 250, 0.45);
      box-shadow:
        0 28px 60px rgba(0, 0, 0, 0.48),
        0 0 60px rgba(139, 92, 246, 0.22),
        0 0 0 1px rgba(167, 139, 250, 0.15) inset;
    }
    .info-card:hover::after { opacity: 1; }
    .info-icon-wrap {
      flex-shrink: 0;
      width: 54px; height: 54px;
      border-radius: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg,
        rgba(99, 102, 241, 0.35) 0%,
        rgba(168, 85, 247, 0.3) 50%,
        rgba(236, 72, 153, 0.25) 100%
      );
      border: 1px solid rgba(167, 139, 250, 0.5);
      font-size: 1.5rem;
      box-shadow:
        0 8px 22px rgba(99, 102, 241, 0.35),
        0 0 0 1px rgba(255, 255, 255, 0.06) inset;
      position: relative;
      transition: transform 0.4s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .info-card:hover .info-icon-wrap {
      transform: rotate(-6deg) scale(1.08);
    }
    .info-body { flex: 1; min-width: 0; }
    .info-label {
      font-size: 0.68rem;
      font-weight: 800;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: rgba(167, 139, 250, 0.95);
      margin-bottom: 0.4rem;
    }
    .info-value {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 700;
      font-size: 1.22rem;
      line-height: 1.2;
      color: #f8fafc;
    }
    .accuracy-highlight {
      background: linear-gradient(135deg, #6ee7b7 0%, #22d3ee 50%, #a78bfa 100%);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      font-size: 1.35rem;
    }

    .input-panel-wrap {
      animation: fadeSlideUp 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.35s both;
      z-index: 6;
      position: relative;
    }
    @keyframes fadeSlideUp {
      0% { opacity: 0; transform: translateY(28px); }
      100% { opacity: 1; transform: translateY(0); }
    }

    .input-panel {
      position: relative;
      padding: 2.6rem 2.6rem 2.8rem 2.6rem;
      border-radius: 26px;
      overflow: hidden;
      margin-bottom: 2.2rem;
      background: linear-gradient(180deg,
        rgba(28, 18, 60, 0.94) 0%,
        rgba(22, 14, 48, 0.96) 100%
      );
      backdrop-filter: blur(30px) saturate(180%);
      -webkit-backdrop-filter: blur(30px) saturate(180%);
      border: 1px solid rgba(167, 139, 250, 0.2);
      box-shadow:
        0 32px 80px rgba(0, 0, 0, 0.5),
        0 0 0 1px rgba(255, 255, 255, 0.04) inset,
        0 0 80px rgba(99, 102, 241, 0.1);
    }
    .input-panel::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2.5px;
      background: linear-gradient(90deg,
        transparent 0%,
        #6366f1 15%,
        #8b5cf6 35%,
        #ec4899 55%,
        #f43f5e 70%,
        #22d3ee 85%,
        transparent 100%
      );
      background-size: 200% 100%;
      animation: shimmer 5s linear infinite;
      opacity: 0.9;
    }
    .input-panel-header {
      position: relative;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 1rem;
      margin-bottom: 2rem;
      padding-bottom: 1.5rem;
      border-bottom: 1px solid rgba(167, 139, 250, 0.1);
    }
    .input-panel-title-wrap { display: flex; align-items: center; gap: 1.1rem; }
    .input-panel-icon {
      width: 54px; height: 54px;
      border-radius: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.5rem;
      background: linear-gradient(135deg,
        rgba(99, 102, 241, 0.35) 0%,
        rgba(168, 85, 247, 0.3) 50%,
        rgba(236, 72, 153, 0.28) 100%
      );
      border: 1px solid rgba(167, 139, 250, 0.45);
      box-shadow:
        0 8px 24px rgba(99, 102, 241, 0.3),
        0 0 0 1px rgba(255, 255, 255, 0.06) inset;
    }
    .input-panel-title {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 700;
      font-size: 1.4rem;
      background: linear-gradient(135deg, #f8fafc 0%, #e0e7ff 50%, #c4b5fd 100%);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      margin: 0;
      letter-spacing: -0.02em;
    }
    .input-panel-sub {
      font-size: 0.78rem;
      color: rgba(196, 181, 253, 0.85);
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-top: 0.2rem;
    }
    .input-panel-meta {
      display: inline-flex;
      align-items: center;
      gap: 0.55rem;
      padding: 0.5rem 1.05rem;
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.18), rgba(236, 72, 153, 0.12));
      border: 1px solid rgba(167, 139, 250, 0.35);
      border-radius: 999px;
      font-size: 0.72rem;
      font-weight: 800;
      color: #e0e7ff;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      box-shadow: 0 0 30px rgba(99, 102, 241, 0.12);
    }

    .fs-section { position: relative; z-index: 2; margin-bottom: 1.6rem; }
    .fs-section-head { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 1.1rem; }
    .fs-step {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 700;
      font-size: 0.68rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: #0f0820;
      padding: 0.35rem 0.8rem;
      background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
      border-radius: 10px;
      box-shadow: 0 4px 14px rgba(139, 92, 246, 0.35);
      position: relative;
      overflow: hidden;
    }
    .fs-step::after {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 30%, rgba(255,255,255,0.35) 50%, transparent 70%);
      transform: translateX(-100%);
      animation: stepShine 3.5s ease-in-out infinite;
    }
    @keyframes stepShine {
      0%, 50%, 100% { transform: translateX(-100%); }
      70% { transform: translateX(100%); }
    }
    .fs-section-title {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 600;
      font-size: 1.02rem;
      color: #e0e7ff;
      letter-spacing: -0.01em;
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }
    .fs-section-icon {
      width: 32px; height: 32px;
      border-radius: 10px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 0.95rem;
      background: linear-gradient(135deg,
        rgba(99, 102, 241, 0.28) 0%,
        rgba(168, 85, 247, 0.22) 100%
      );
      border: 1px solid rgba(167, 139, 250, 0.4);
      color: #c4b5fd;
    }

    .input-card {
      background: linear-gradient(160deg, rgba(34, 22, 66, 0.95) 0%, rgba(26, 16, 50, 0.95) 100%);
      border-radius: 18px;
      padding: 1.1rem 1.1rem 1.15rem 1.1rem;
      border: 1px solid rgba(167, 139, 250, 0.16);
      box-shadow:
        0 6px 20px rgba(0, 0, 0, 0.32),
        0 0 0 1px rgba(255, 255, 255, 0.03) inset;
      transition: all 0.38s cubic-bezier(0.22, 1, 0.36, 1);
      position: relative;
      overflow: hidden;
    }
    .input-card::before {
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at var(--mx, 50%) var(--my, 50%),
        rgba(139, 92, 246, 0.12) 0%,
        transparent 60%
      );
      opacity: 0;
      transition: opacity 0.35s ease;
      pointer-events: none;
    }
    .input-card:hover {
      border-color: rgba(167, 139, 250, 0.42);
      box-shadow:
        0 12px 32px rgba(0, 0, 0, 0.42),
        0 0 0 1px rgba(167, 139, 250, 0.15),
        0 0 40px rgba(99, 102, 241, 0.15),
        0 0 0 1px rgba(255, 255, 255, 0.04) inset;
      transform: translateY(-3px);
    }
    .input-card:hover::before { opacity: 1; }

    label, .stSelectbox label p, .stNumberInput label, .stSlider label p {
      color: #e0e7ff !important;
      font-weight: 600 !important;
      font-size: 0.83rem !important;
      margin-bottom: 0.4rem !important;
      letter-spacing: 0.01em;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label {
      margin-bottom: 0.5rem !important;
      padding-bottom: 0 !important;
    }

    .stSelectbox [data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
      background: linear-gradient(180deg, rgba(18, 12, 40, 0.92) 0%, rgba(15, 10, 34, 0.92) 100%) !important;
      border: 1px solid rgba(167, 139, 250, 0.22) !important;
      border-radius: 12px !important;
      color: #f1f5f9 !important;
      font-weight: 500 !important;
      font-size: 0.9rem !important;
      min-height: 46px;
      transition: all 0.28s cubic-bezier(0.22, 1, 0.36, 1) !important;
      padding-left: 0.95rem !important;
      padding-right: 0.95rem !important;
      box-shadow:
        inset 0 2px 4px rgba(0, 0, 0, 0.3),
        0 0 0 0 rgba(99, 102, 241, 0) !important;
    }
    .stSelectbox [data-baseweb="select"] > div:hover,
    .stNumberInput input:hover,
    .stTextInput input:hover {
      border-color: rgba(167, 139, 250, 0.5) !important;
      box-shadow:
        inset 0 2px 4px rgba(0, 0, 0, 0.3),
        0 0 0 4px rgba(99, 102, 241, 0.12) !important;
    }
    .stSelectbox [data-baseweb="select"]:focus-within > div,
    .stNumberInput:focus-within input,
    .stTextInput:focus-within input {
      border-color: rgba(167, 139, 250, 0.78) !important;
      background: linear-gradient(180deg, rgba(22, 14, 48, 0.96) 0%, rgba(18, 12, 40, 0.96) 100%) !important;
      box-shadow:
        0 0 0 5px rgba(99, 102, 241, 0.18),
        0 0 28px rgba(167, 139, 250, 0.22),
        inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    [data-baseweb="select"] svg { color: #a5b4fc !important; }
    [data-baseweb="popover"] {
      background: linear-gradient(180deg, #1a0f3c 0%, #140a2e 100%) !important;
      border: 1px solid rgba(167, 139, 250, 0.28) !important;
      border-radius: 14px !important;
      box-shadow:
        0 28px 60px rgba(0,0,0,0.58),
        0 0 50px rgba(99, 102, 241, 0.18) !important;
      backdrop-filter: blur(20px);
    }
    [data-baseweb="popover"] ul li {
      color: #e2e8f0 !important;
      font-size: 0.85rem !important;
      padding: 0.65rem 0.95rem !important;
      transition: all 0.2s ease;
    }
    [data-baseweb="popover"] ul li:hover {
      background: linear-gradient(90deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.18) 100%) !important;
      color: #ffffff !important;
    }

    .stNumberInput button[aria-label="Increase"],
    .stNumberInput button[aria-label="Decrease"] {
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.18) 100%) !important;
      border: 1px solid rgba(167, 139, 250, 0.38) !important;
      color: #c4b5fd !important;
      border-radius: 8px !important;
      transition: all 0.2s ease !important;
    }
    .stNumberInput button[aria-label="Increase"]:hover,
    .stNumberInput button[aria-label="Decrease"]:hover {
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.38) 0%, rgba(168, 85, 247, 0.32) 100%) !important;
      color: #ffffff !important;
      transform: scale(1.08);
    }

    [data-baseweb="slider"] [data-testid="stTickBar"] > div {
      background: rgba(167, 139, 250, 0.25) !important;
      height: 2.5px !important;
    }
    [data-baseweb="slider"] > div > div > div {
      background: linear-gradient(90deg,
        #6366f1 0%,
        #8b5cf6 35%,
        #a855f7 65%,
        #ec4899 100%
      ) !important;
      height: 5px !important;
      border-radius: 999px;
      box-shadow: 0 0 16px rgba(139, 92, 246, 0.45);
    }
    [data-baseweb="slider"] [role="slider"] > div > div {
      background: linear-gradient(135deg, #a5b4fc, #f0abfc) !important;
      box-shadow:
        0 0 0 5px rgba(129, 140, 248, 0.2),
        0 0 20px rgba(168, 85, 247, 0.7),
        0 0 40px rgba(236, 72, 153, 0.35) !important;
      width: 18px !important;
      height: 18px !important;
      border: 2.5px solid #fff !important;
      transition: transform 0.2s cubic-bezier(0.22, 1, 0.36, 1);
    }
    [data-baseweb="slider"] [role="slider"]:hover > div > div {
      transform: scale(1.2);
    }
    [data-testid="stSliderThumbValue"] {
      background: linear-gradient(135deg, #6366f1, #a855f7) !important;
      color: #fff !important;
      font-weight: 700 !important;
      font-size: 0.74rem !important;
      border: none !important;
      border-radius: 8px !important;
      padding: 0.15rem 0.5rem !important;
      top: -2rem !important;
      box-shadow: 0 6px 16px rgba(99, 102, 241, 0.45);
    }

    .stNumberInput, .stSelectbox, .stSlider { margin-bottom: 0.45rem !important; }
    .stNumberInput > div, .stSelectbox > div, .stSlider > div {
      padding-bottom: 0 !important;
      margin-bottom: 0 !important;
    }

    .form-submit-wrap {
      position: relative;
      z-index: 2;
      margin-top: 2rem;
      padding-top: 1.7rem;
      border-top: 1px solid rgba(167, 139, 250, 0.1);
    }
    .stFormSubmitButton > button, button[kind="primary"] {
      position: relative !important;
      width: 100% !important;
      padding: 1.1rem 1.6rem !important;
      font-family: 'Space Grotesk', 'Poppins', sans-serif !important;
      font-size: 0.96rem !important;
      font-weight: 800 !important;
      color: #ffffff !important;
      letter-spacing: 0.08em !important;
      background: linear-gradient(135deg,
        #4f46e5 0%,
        #7c3aed 20%,
        #9333ea 40%,
        #c026d3 60%,
        #db2777 80%,
        #f43f5e 100%
      ) !important;
      background-size: 280% 280% !important;
      border: none !important;
      border-radius: 16px !important;
      transition: all 0.4s cubic-bezier(0.22, 1, 0.36, 1) !important;
      box-shadow:
        0 16px 44px rgba(79, 70, 229, 0.5),
        0 0 60px rgba(168, 85, 247, 0.25),
        0 0 0 1px rgba(167, 139, 250, 0.45) inset,
        0 -2px 0 rgba(255, 255, 255, 0.14) inset !important;
      overflow: hidden !important;
      text-transform: uppercase;
      min-height: 58px;
    }
    .stFormSubmitButton > button::before {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 20%, rgba(255, 255, 255, 0.28) 50%, transparent 80%);
      transform: translateX(-100%);
      transition: transform 0.8s ease;
    }
    .stFormSubmitButton > button::after {
      content: '';
      position: absolute;
      inset: -2px;
      border-radius: 18px;
      padding: 2px;
      background: linear-gradient(135deg, #818cf8, #f472b6, #22d3ee, #818cf8);
      background-size: 300% 300%;
      animation: borderSpin 4s linear infinite;
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.6;
      pointer-events: none;
    }
    @keyframes borderSpin {
      0%, 100% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
    }
    .stFormSubmitButton > button:hover {
      transform: translateY(-4px) scale(1.005) !important;
      background-size: 320% 320% !important;
      background-position: 100% 100% !important;
      box-shadow:
        0 26px 60px rgba(99, 102, 241, 0.65),
        0 0 90px rgba(168, 85, 247, 0.4),
        0 0 0 1px rgba(192, 132, 252, 0.6) inset,
        0 -2px 0 rgba(255, 255, 255, 0.18) inset !important;
    }
    .stFormSubmitButton > button:hover::before { transform: translateX(100%); }
    .stFormSubmitButton > button:active { transform: translateY(-1px) scale(0.99) !important; }

    .result-hero {
      position: relative;
      padding: 3.6rem 3rem 3rem 3rem;
      border-radius: 28px;
      overflow: hidden;
      text-align: center;
      background:
        linear-gradient(160deg, rgba(30, 18, 66, 0.95) 0%, rgba(44, 22, 92, 0.9) 35%, rgba(26, 14, 58, 0.96) 70%, rgba(18, 10, 42, 0.98) 100%);
      backdrop-filter: blur(40px) saturate(200%);
      -webkit-backdrop-filter: blur(40px) saturate(200%);
      border: 1px solid rgba(167, 139, 250, 0.3);
      box-shadow:
        0 40px 100px rgba(0, 0, 0, 0.6),
        0 0 120px rgba(99, 102, 241, 0.18),
        0 0 0 1px rgba(255, 255, 255, 0.04) inset;
      animation: resultBounceIn 0.95s cubic-bezier(0.34, 1.56, 0.64, 1);
      margin-bottom: 1.8rem;
    }
    @keyframes resultBounceIn {
      0% { opacity: 0; transform: scale(0.9) translateY(30px); }
      60% { opacity: 1; transform: scale(1.025) translateY(-4px); }
      100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    .result-hero::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
      background: linear-gradient(90deg,
        transparent 0%,
        #6366f1 15%,
        #8b5cf6 30%,
        #a855f7 50%,
        #ec4899 70%,
        #f43f5e 85%,
        transparent 100%
      );
      background-size: 200% 100%;
      animation: shimmer 4s linear infinite;
    }
    .result-hero::after {
      content: '';
      position: absolute;
      top: -30%;
      left: 50%;
      transform: translateX(-50%);
      width: 80%;
      height: 80%;
      background: radial-gradient(ellipse, var(--result-glow, rgba(16, 185, 129, 0.22)) 0%, transparent 65%);
      pointer-events: none;
      animation: resultGlow 3s ease-in-out infinite;
    }
    @keyframes resultGlow {
      0%, 100% { opacity: 0.6; transform: translateX(-50%) scale(1); }
      50% { opacity: 1; transform: translateX(-50%) scale(1.08); }
    }
    .result-tag {
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
      padding: 0.55rem 1.35rem;
      border-radius: 999px;
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 800;
      font-size: 0.76rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      margin-bottom: 1.8rem;
      backdrop-filter: blur(12px);
      position: relative;
      overflow: hidden;
      animation: tagPulse 2.2s ease-in-out infinite;
    }
    @keyframes tagPulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.03); }
    }
    .result-tag.profit {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(34, 211, 238, 0.14) 100%);
      border: 1px solid rgba(52, 211, 153, 0.5);
      color: #6ee7b7;
      box-shadow: 0 0 40px rgba(16, 185, 129, 0.2);
    }
    .result-tag.loss {
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(251, 113, 133, 0.15) 100%);
      border: 1px solid rgba(248, 113, 113, 0.52);
      color: #fecaca;
      box-shadow: 0 0 40px rgba(239, 68, 68, 0.2);
    }
    .result-figure {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 900;
      font-size: 5.4rem;
      line-height: 1;
      margin: 0 0 1.2rem 0;
      letter-spacing: -0.04em;
      position: relative;
      z-index: 2;
    }
    .result-figure.profit {
      background: linear-gradient(135deg,
        #a7f3d0 0%,
        #34d399 20%,
        #10b981 40%,
        #22d3ee 60%,
        #67e8f9 80%,
        #a5f3fc 100%
      );
      background-size: 280% 280%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: valuePop 1.2s cubic-bezier(0.34, 1.56, 0.64, 1) 0.2s both, gradientShiftVibrant 5s ease infinite;
      filter: drop-shadow(0 0 40px rgba(16, 185, 129, 0.45));
    }
    .result-figure.loss {
      background: linear-gradient(135deg,
        #fecaca 0%,
        #fb7185 20%,
        #f43f5e 40%,
        #ef4444 60%,
        #f87171 80%,
        #fca5a5 100%
      );
      background-size: 280% 280%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: valuePop 1.2s cubic-bezier(0.34, 1.56, 0.64, 1) 0.2s both, gradientShiftVibrant 5s ease infinite;
      filter: drop-shadow(0 0 40px rgba(239, 68, 68, 0.45));
    }
    @keyframes valuePop {
      0% { opacity: 0; transform: scale(0.5); filter: blur(24px); }
      60% { opacity: 1; transform: scale(1.12); filter: blur(0); }
      100% { transform: scale(1); }
    }
    .result-caption {
      font-size: 1.12rem;
      font-weight: 600;
      color: rgba(226, 232, 240, 0.95);
      margin: 0 0 2.4rem 0;
      position: relative;
      z-index: 2;
    }

    .stats-triplet {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1.2rem;
      position: relative;
      z-index: 2;
    }
    .stat-pill {
      padding: 1.35rem 1.25rem;
      border-radius: 20px;
      background: linear-gradient(160deg, rgba(24, 15, 54, 0.85) 0%, rgba(18, 11, 40, 0.85) 100%);
      border: 1px solid rgba(167, 139, 250, 0.22);
      backdrop-filter: blur(16px);
      transition: all 0.4s cubic-bezier(0.22, 1, 0.36, 1);
      position: relative;
      overflow: hidden;
      text-align: left;
      box-shadow:
        0 0 0 1px rgba(255, 255, 255, 0.04) inset,
        0 8px 24px rgba(0,0,0,0.3);
      animation: statIn 0.7s cubic-bezier(0.34, 1.56, 0.64, 1) both;
    }
    .stat-pill:nth-child(1) { animation-delay: 0.45s; }
    .stat-pill:nth-child(2) { animation-delay: 0.55s; }
    .stat-pill:nth-child(3) { animation-delay: 0.65s; }
    @keyframes statIn {
      0% { opacity: 0; transform: translateY(18px) scale(0.95); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    .stat-pill::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1.5px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(129, 140, 248, 0.7) 40%,
        rgba(236, 72, 153, 0.6) 60%,
        transparent 100%
      );
    }
    .stat-pill:hover {
      transform: translateY(-5px) scale(1.02);
      border-color: rgba(167, 139, 250, 0.45);
      box-shadow:
        0 20px 44px rgba(0,0,0,0.48),
        0 0 50px rgba(99, 102, 241, 0.22),
        0 0 0 1px rgba(255,255,255,0.06) inset;
    }
    .stat-top { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.8rem; }
    .stat-circle {
      flex-shrink: 0;
      width: 48px; height: 48px;
      border-radius: 14px;
      border: 1px solid rgba(167, 139, 250, 0.5);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
      background: linear-gradient(135deg,
        rgba(99, 102, 241, 0.3),
        rgba(168, 85, 247, 0.22)
      );
      box-shadow: 0 0 18px rgba(99, 102, 241, 0.28);
      transition: transform 0.4s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .stat-pill:hover .stat-circle {
      transform: rotate(-8deg) scale(1.1);
    }
    .stat-k {
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 800;
      color: rgba(167, 139, 250, 0.9);
      margin-bottom: 0.3rem;
    }
    .stat-v {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 700;
      font-size: 1.35rem;
      line-height: 1.15;
      background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 50%, #c4b5fd 100%);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .streamlit-expanderHeader {
      background: linear-gradient(160deg, rgba(28, 18, 60, 0.94) 0%, rgba(22, 14, 48, 0.94) 100%) !important;
      backdrop-filter: blur(16px);
      border: 1px solid rgba(167, 139, 250, 0.2) !important;
      border-radius: 18px !important;
      padding: 1.1rem 1.4rem !important;
      transition: all 0.3s cubic-bezier(0.22, 1, 0.36, 1) !important;
      color: #e0e7ff !important;
      font-weight: 700 !important;
      font-size: 0.96rem !important;
      margin-bottom: 1rem !important;
      box-shadow:
        0 8px 24px rgba(0,0,0,0.35),
        0 0 0 1px rgba(255,255,255,0.04) inset !important;
      font-family: 'Space Grotesk', sans-serif !important;
    }
    .streamlit-expanderHeader:hover {
      border-color: rgba(167, 139, 250, 0.48) !important;
      box-shadow:
        0 14px 36px rgba(0,0,0,0.45),
        0 0 36px rgba(99, 102, 241, 0.18),
        0 0 0 1px rgba(255,255,255,0.06) inset !important;
      transform: translateY(-2px);
    }
    [data-testid="stExpander"] { border: none !important; }
    [data-testid="stExpanderDetails"] {
      background: rgba(18, 11, 40, 0.75) !important;
      backdrop-filter: blur(14px);
      border: 1px solid rgba(167, 139, 250, 0.15) !important;
      border-top: none !important;
      border-radius: 0 0 18px 18px !important;
      padding: 1.5rem !important;
      margin: -1rem 0 1.1rem 0 !important;
      color: #cbd5e1 !important;
      line-height: 1.85;
      font-size: 0.95rem;
    }
    [data-testid="stExpanderDetails"] strong {
      background: linear-gradient(135deg, #c4b5fd, #f0abfc);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800 !important;
    }
    [data-testid="stExpanderDetails"] li { color: #cbd5e1 !important; margin-bottom: 0.35rem; }

    [data-testid="stAlertContainer"] { margin-bottom: 1.1rem !important; }
    [data-testid="stAlert"] {
      background: linear-gradient(160deg, rgba(30, 14, 30, 0.95) 0%, rgba(24, 10, 22, 0.95) 100%) !important;
      backdrop-filter: blur(16px) !important;
      border: 1px solid rgba(248, 113, 113, 0.45) !important;
      border-left: 5px solid #f43f5e !important;
      border-radius: 16px !important;
      box-shadow:
        0 14px 36px rgba(0, 0, 0, 0.4),
        0 0 40px rgba(244, 63, 94, 0.12) !important;
      color: #fecaca !important;
      font-weight: 500 !important;
      padding: 1.1rem 1.25rem !important;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
      color: #fecaca !important;
      font-size: 0.92rem;
      line-height: 1.7;
    }
    [data-testid="stAlert"][kind="warning"] {
      background: linear-gradient(160deg, rgba(38, 26, 10, 0.95) 0%, rgba(30, 20, 8, 0.95) 100%) !important;
      border: 1px solid rgba(251, 191, 36, 0.5) !important;
      border-left: 5px solid #f59e0b !important;
      color: #fde68a !important;
      box-shadow:
        0 14px 36px rgba(0, 0, 0, 0.4),
        0 0 40px rgba(245, 158, 11, 0.12) !important;
    }
    [data-testid="stAlert"][kind="warning"] [data-testid="stMarkdownContainer"] { color: #fde68a !important; }

    [data-testid="stDataFrame"] {
      background: linear-gradient(160deg, rgba(24, 15, 54, 0.82) 0%, rgba(18, 11, 40, 0.82) 100%) !important;
      border-radius: 18px !important;
      border: 1px solid rgba(167, 139, 250, 0.22) !important;
      overflow: hidden !important;
      box-shadow:
        0 14px 36px rgba(0,0,0,0.42),
        0 0 0 1px rgba(255,255,255,0.04) inset !important;
    }

    .foot-card {
      margin-top: 2.4rem;
      padding: 1.5rem 2.2rem;
      text-align: center;
      border-radius: 20px;
      background: linear-gradient(160deg, rgba(22, 13, 50, 0.88) 0%, rgba(15, 9, 34, 0.88) 100%);
      backdrop-filter: blur(18px);
      border: 1px solid rgba(167, 139, 250, 0.14);
      color: rgba(196, 181, 253, 0.92);
      font-size: 0.88rem;
      line-height: 1.8;
      box-shadow:
        0 0 0 1px rgba(255,255,255,0.03) inset,
        0 10px 30px rgba(0,0,0,0.3);
      position: relative;
      overflow: hidden;
    }
    .foot-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1.5px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(129, 140, 248, 0.6) 30%,
        rgba(192, 132, 252, 0.55) 50%,
        rgba(236, 72, 153, 0.5) 70%,
        transparent 100%
      );
    }
    .foot-card strong {
      background: linear-gradient(135deg, #c4b5fd, #f0abfc, #22d3ee);
      background-size: 200% 200%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800;
      animation: gradientShiftVibrant 6s ease infinite;
    }

    ::-webkit-scrollbar { width: 11px; height: 11px; }
    ::-webkit-scrollbar-track { background: rgba(10, 6, 22, 0.7); }
    ::-webkit-scrollbar-thumb {
      background: linear-gradient(180deg,
        #6366f1 0%,
        #8b5cf6 30%,
        #a855f7 60%,
        #ec4899 100%
      );
      border-radius: 6px;
      border: 2.5px solid transparent;
      background-clip: padding-box;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: linear-gradient(180deg,
        #818cf8 0%,
        #a78bfa 30%,
        #c084fc 60%,
        #f472b6 100%
      );
      border: 2.5px solid transparent;
      background-clip: padding-box;
    }

    @media (max-width: 1200px) {
      .info-strip { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 1024px) {
      [data-testid="stAppViewContainer"] > section[data-testid="stMain"] {
        padding: 1.6rem 1.6rem 3rem 1.6rem !important;
      }
      .hero-big { font-size: 2.4rem; }
      .hero-banner { padding: 2.4rem 2rem 2.6rem 2rem; }
      .input-panel { padding: 2rem 1.7rem 2.2rem 1.7rem; }
    }
    @media (max-width: 768px) {
      .info-strip { grid-template-columns: 1fr; }
      .stats-triplet { grid-template-columns: 1fr; }
      .hero-big { font-size: 1.9rem; }
      .result-figure { font-size: 3.4rem !important; }
      .hero-banner { padding: 2rem 1.4rem 2.2rem 1.4rem; }
      .result-hero { padding: 2.6rem 1.3rem 2.1rem 1.3rem; }
      .input-panel { padding: 1.6rem 1.2rem 1.8rem 1.2rem; border-radius: 22px; }
      .brand-title-main { font-size: 1.55rem; }
      .brand-logo-wrap { width: 58px; height: 58px; }
      .brand-logo { font-size: 1.9rem; }
      [data-testid="stAppViewContainer"] > section[data-testid="stMain"] {
        padding: 1.3rem 1rem 2.5rem 1rem !important;
      }
    }
    </style>

    <div class="floating-orb orb-1"></div>
    <div class="floating-orb orb-2"></div>
    <div class="floating-orb orb-3"></div>
"""), unsafe_allow_html=True)

st.markdown(flat_html("""
    <div class="app-hero-wrap">
      <div class="hero-banner">
        <div class="hero-border-top"></div>
        <div class="hero-top-row">
          <div class="brand-left-main">
            <div class="brand-logo-wrap">
              <span class="brand-logo">💸</span>
            </div>
            <div class="brand-text-main">
              <h1 class="brand-title-main">Profit Oracle</h1>
              <p class="brand-tagline-main">Launch Intelligence Suite · AI-Powered Profit Forecasting</p>
            </div>
          </div>
          <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
            <div class="hero-badge">
              <span class="hero-dot"></span>
              AI-DRIVEN · ML MODEL LIVE
            </div>
            <div class="hero-meta">
              <span>📊 Assignment 2 · AI & Python Programming</span>
            </div>
          </div>
        </div>
        <h1 class="hero-big">
          <span class="part1">Smart </span><span class="part2">Profit Forecaster</span>
        </h1>
        <p class="hero-sub">
          Configure your launch plan below and the system will estimate the
          <strong>First-Year Profit</strong> using a trained Linear Regression pipeline
          built on thousands of historical product launches. Predictions update instantly
          when you submit a new configuration.
        </p>
      </div>
    </div>
"""), unsafe_allow_html=True)

st.markdown(flat_html(f"""
    <div class="info-strip">
      <div class="info-card">
        <div class="info-icon-wrap">🧪</div>
        <div class="info-body">
          <div class="info-label">Model Architecture</div>
          <div class="info-value">Linear Regression Pipeline</div>
        </div>
      </div>
      <div class="info-card">
        <div class="info-icon-wrap">🏆</div>
        <div class="info-body">
          <div class="info-label">Test R² Score</div>
          <div class="accuracy-highlight">94.3% Accuracy</div>
        </div>
      </div>
    </div>
"""), unsafe_allow_html=True)

st.markdown('<div class="input-panel-wrap">', unsafe_allow_html=True)
with st.form("prediction_form"):
    st.markdown(flat_html("""
        <div class="input-panel">
          <div class="input-panel-header">
            <div class="input-panel-title-wrap">
              <div class="input-panel-icon">⚙️</div>
              <div>
                <h3 class="input-panel-title">Configure Your Product Launch</h3>
                <div class="input-panel-sub">Fill in the details below · All fields required</div>
              </div>
            </div>
            <div class="input-panel-meta">
              📋 3-Step Form · Step 01 → 02 → 03
            </div>
          </div>
    """), unsafe_allow_html=True)

    st.markdown(flat_html("""
          <div class="fs-section">
            <div class="fs-section-head">
              <span class="fs-step">Step 01</span>
              <div class="fs-section-title">
                <span class="fs-section-icon">🎯</span>
                Product &amp; Market Details
              </div>
            </div>
          </div>
    """), unsafe_allow_html=True)

    col1_pm, col2_pm = st.columns(2)
    with col1_pm:
        product_category = st.selectbox("Product Category", CATEGORY_OPTIONS, index=0)
    with col2_pm:
        launch_region = st.selectbox("Launch Region", REGION_OPTIONS, index=0)

    st.markdown(flat_html("""
          <div class="fs-section">
            <div class="fs-section-head">
              <span class="fs-step">Step 02</span>
              <div class="fs-section-title">
                <span class="fs-section-icon">💲</span>
                Cost &amp; Pricing Structure
              </div>
            </div>
          </div>
    """), unsafe_allow_html=True)

    col1_cp, col2_cp = st.columns(2)
    with col1_cp:
        development_cost = st.number_input(
            "Development Cost ($)", min_value=0.0, max_value=5_000_000.0,
            value=150_000.0, step=5_000.0,
            help="Total R&D / engineering cost to develop the product.",
        )

        unit_production_cost = st.number_input(
            "Unit Production Cost ($)", min_value=0.01, max_value=100_000.0,
            value=60.0, step=1.0,
            help="Cost to manufacture a single unit.",
        )
    with col2_cp:
        launch_marketing_spend = st.number_input(
            "Launch Marketing Spend ($)", min_value=0.0, max_value=5_000_000.0,
            value=100_000.0, step=5_000.0,
            help="Total marketing budget for the first-year launch.",
        )

        unit_selling_price = st.number_input(
            "Unit Selling Price ($)", min_value=0.01, max_value=100_000.0,
            value=150.0, step=1.0,
            help="Retail price per unit.",
        )

    st.markdown(flat_html("""
          <div class="fs-section">
            <div class="fs-section-head">
              <span class="fs-step">Step 03</span>
              <div class="fs-section-title">
                <span class="fs-section-icon">📈</span>
                Demand Forecast &amp; Market Reach
              </div>
            </div>
          </div>
    """), unsafe_allow_html=True)

    col1_dr, col2_dr, col3_dr = st.columns(3)
    with col1_dr:
        first_year_units_sold = st.number_input(
            "First-Year Units Sold", min_value=0, max_value=10_000_000,
            value=15_000, step=500,
            help="Forecasted number of units sold in the first year.",
        )
    with col2_dr:
        customer_rating = st.slider(
            "Expected Customer Rating", min_value=1.0, max_value=5.0,
            value=4.3, step=0.1,
        )
    with col3_dr:
        distribution_coverage = st.slider(
            "Distribution Coverage", min_value=0.0, max_value=1.0,
            value=0.80, step=0.01,
            help="Fraction of target retail/online channels the product will be available in.",
        )

    st.markdown('<div class="form-submit-wrap">', unsafe_allow_html=True)
    submitted = st.form_submit_button("✨  Predict First-Year Profit", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

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
        result_glow = "rgba(16, 185, 129, 0.25)" if is_profit else "rgba(239, 68, 68, 0.25)"

        result_html = flat_html(f"""
            <div class="result-hero" style="--result-glow: {result_glow};">
              <div class="result-tag {tag_cls}">
                <span class="hero-dot" style="background: linear-gradient(135deg, {'#10b981, #22d3ee' if is_profit else '#ef4444, #fb7185'});"></span>
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
        """)
        st.markdown(result_html, unsafe_allow_html=True)

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
            display_df = input_df.T.rename(columns={0: "Value"})
            display_df["Value"] = display_df["Value"].astype(str)
            st.dataframe(display_df, width="stretch")

st.markdown(flat_html("""
    <div class="foot-card">
      <strong>Model:</strong> Linear Regression pipeline (StandardScaler + OneHotEncoder) trained on
      historical New Product launches · Assignment 2 — AI Spreadsheets &amp; Python Programming
    </div>
"""), unsafe_allow_html=True)
