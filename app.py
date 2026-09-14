"""
New Product Profit Prediction — Streamlit Web Application
Model to Web Application

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

    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], [data-testid="block-container"], [data-testid="stHeader"] {
      font-size: 17.5px !important;
      line-height: 1.58;
      -webkit-text-size-adjust: 100%;
    }

    * {
      font-family: 'Inter', 'Poppins', 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    .stApp {
      background:
        radial-gradient(ellipse 95% 65% at 12% -22%, rgba(251, 146, 60, 0.22) 0%, rgba(168, 85, 247, 0.12) 40%, transparent 68%),
        radial-gradient(ellipse 75% 58% at 88% -8%, rgba(139, 92, 246, 0.48) 0%, rgba(59, 130, 246, 0.18) 45%, transparent 65%),
        radial-gradient(ellipse 60% 52% at 96% 30%, rgba(236, 72, 153, 0.3) 0%, rgba(251, 113, 133, 0.14) 48%, transparent 65%),
        radial-gradient(ellipse 85% 70% at 48% 132%, rgba(59, 130, 246, 0.32) 0%, rgba(168, 85, 247, 0.16) 45%, transparent 65%),
        radial-gradient(ellipse 55% 50% at 6% 88%, rgba(251, 113, 133, 0.24) 0%, rgba(251, 191, 36, 0.1) 50%, transparent 65%),
        radial-gradient(ellipse 50% 45% at 94% 82%, rgba(34, 211, 238, 0.26) 0%, rgba(16, 185, 129, 0.1) 52%, transparent 65%),
        linear-gradient(180deg, #12071e 0%, #1a0c2e 20%, #1c0e30 40%, #180b2b 60%, #140926 80%, #100620 100%);
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
        radial-gradient(circle at 22% 28%, rgba(251, 191, 36, 0.085) 0%, transparent 42%),
        radial-gradient(circle at 78% 72%, rgba(168, 85, 247, 0.075) 0%, transparent 42%),
        radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.04) 0%, transparent 60%),
        radial-gradient(circle at 8% 50%, rgba(251, 113, 133, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 92% 50%, rgba(34, 211, 238, 0.05) 0%, transparent 40%);
      background-size: auto;
      animation: auroraDrift 20s ease-in-out infinite alternate;
    }
    @keyframes auroraDrift {
      0% { transform: translate(0, 0) scale(1); }
      50% { transform: translate(2.5%, -1.5%) scale(1.05); }
      100% { transform: translate(-1.5%, 1.5%) scale(1.03); }
    }

    .stApp::after {
      content: '';
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 0;
      background-image:
        radial-gradient(2.4px 2.4px at 10% 15%, rgba(253, 224, 71, 0.82), transparent),
        radial-gradient(1.8px 1.8px at 26% 40%, rgba(196, 181, 253, 0.64), transparent),
        radial-gradient(2.2px 2.2px at 42% 10%, rgba(251, 146, 60, 0.58), transparent),
        radial-gradient(2px 2px at 60% 56%, rgba(167, 139, 250, 0.68), transparent),
        radial-gradient(2.4px 2.4px at 76% 20%, rgba(244, 114, 182, 0.54), transparent),
        radial-gradient(1.8px 1.8px at 88% 66%, rgba(34, 211, 238, 0.64), transparent),
        radial-gradient(2.2px 2.2px at 34% 80%, rgba(253, 186, 116, 0.58), transparent),
        radial-gradient(2px 2px at 56% 90%, rgba(129, 140, 248, 0.58), transparent),
        radial-gradient(1.8px 1.8px at 6% 58%, rgba(249, 168, 212, 0.54), transparent),
        radial-gradient(2.2px 2.2px at 72% 86%, rgba(134, 239, 172, 0.58), transparent),
        radial-gradient(1.8px 1.8px at 18% 70%, rgba(165, 180, 252, 0.54), transparent),
        radial-gradient(2.2px 2.2px at 50% 35%, rgba(253, 224, 71, 0.48), transparent),
        radial-gradient(2px 2px at 82% 45%, rgba(240, 171, 252, 0.58), transparent),
        radial-gradient(1.8px 1.8px at 30% 5%, rgba(192, 132, 252, 0.54), transparent),
        radial-gradient(2.2px 2.2px at 68% 4%, rgba(251, 146, 60, 0.52), transparent),
        radial-gradient(1.5px 1.5px at 15% 88%, rgba(251, 191, 36, 0.52), transparent),
        radial-gradient(1.5px 1.5px at 46% 68%, rgba(236, 72, 153, 0.5), transparent),
        radial-gradient(1.5px 1.5px at 94% 12%, rgba(99, 102, 241, 0.55), transparent),
        radial-gradient(1.4px 1.4px at 38% 48%, rgba(34, 211, 238, 0.48), transparent),
        radial-gradient(1.4px 1.4px at 66% 74%, rgba(251, 113, 133, 0.48), transparent),
        radial-gradient(1.3px 1.3px at 22% 22%, rgba(168, 85, 247, 0.45), transparent),
        radial-gradient(1.3px 1.3px at 86% 34%, rgba(16, 185, 129, 0.5), transparent);
      background-size: 100% 100%;
      animation: starTwinkle 7s ease-in-out infinite;
      opacity: 0.92;
    }
    @keyframes starTwinkle {
      0%, 100% { opacity: 0.55; filter: brightness(0.85); }
      50% { opacity: 1; filter: brightness(1.15); }
    }

    .stApp > [data-testid="stAppViewContainer"]::before {
      content: '';
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 2;
      background: radial-gradient(ellipse at center, transparent 45%, rgba(5, 2, 14, 0.55) 100%);
    }

    .floating-orb {
      position: fixed;
      border-radius: 50%;
      pointer-events: none;
      z-index: 1;
      filter: blur(95px);
      opacity: 0.68;
    }
    .orb-1 {
      width: 620px; height: 620px;
      background: radial-gradient(circle, rgba(251, 191, 36, 0.56) 0%, rgba(236, 72, 153, 0.36) 32%, rgba(139, 92, 246, 0.42) 58%, transparent 78%);
      top: -24%; left: -14%;
      animation: floatOrb1 22s ease-in-out infinite;
    }
    .orb-2 {
      width: 560px; height: 560px;
      background: radial-gradient(circle, rgba(236, 72, 153, 0.56) 0%, rgba(251, 146, 60, 0.36) 38%, rgba(139, 92, 246, 0.28) 66%, transparent 80%);
      top: 20%; right: -16%;
      animation: floatOrb2 26s ease-in-out infinite;
    }
    .orb-3 {
      width: 520px; height: 520px;
      background: radial-gradient(circle, rgba(34, 211, 238, 0.5) 0%, rgba(99, 102, 241, 0.4) 38%, rgba(168, 85, 247, 0.22) 62%, transparent 80%);
      bottom: -22%; left: 22%;
      animation: floatOrb3 30s ease-in-out infinite;
    }
    .orb-4 {
      width: 480px; height: 480px;
      background: radial-gradient(circle, rgba(251, 191, 36, 0.42) 0%, rgba(251, 113, 133, 0.36) 36%, rgba(168, 85, 247, 0.24) 66%, transparent 82%);
      top: 54%; left: -12%;
      animation: floatOrb4 28s ease-in-out infinite;
    }
    .orb-5 {
      width: 420px; height: 420px;
      background: radial-gradient(circle, rgba(16, 185, 129, 0.34) 0%, rgba(34, 211, 238, 0.3) 38%, rgba(129, 140, 248, 0.24) 66%, transparent 82%);
      top: 66%; right: -8%;
      animation: floatOrb5 32s ease-in-out infinite;
    }
    @keyframes floatOrb1 {
      0%, 100% { transform: translate(0, 0) scale(1); }
      33% { transform: translate(70px, 90px) scale(1.14); }
      66% { transform: translate(-45px, 60px) scale(0.92); }
    }
    @keyframes floatOrb2 {
      0%, 100% { transform: translate(0, 0) scale(1); }
      33% { transform: translate(-70px, 50px) scale(0.9); }
      66% { transform: translate(50px, -60px) scale(1.18); }
    }
    @keyframes floatOrb3 {
      0%, 100% { transform: translate(0, 0) scale(1); }
      50% { transform: translate(85px, -75px) scale(1.15); }
    }
    @keyframes floatOrb4 {
      0%, 100% { transform: translate(0, 0) scale(1) rotate(0deg); }
      50% { transform: translate(55px, -45px) scale(1.1) rotate(12deg); }
    }
    @keyframes floatOrb5 {
      0%, 100% { transform: translate(0, 0) scale(1); }
      50% { transform: translate(-45px, 35px) scale(1.08); }
    }

    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stAppViewContainer"] > section[data-testid="stMain"] {
      padding: 1.5rem 2.2rem 2.6rem 2.2rem !important;
      z-index: 5;
      position: relative;
      overflow-x: hidden;
    }
    [data-testid="stAppViewContainer"] > section[data-testid="stMain"] > [data-testid="block-container"] {
      padding-top: 0 !important;
      max-width: 1500px;
    }

    [data-testid="stHeader"] {
      background: linear-gradient(180deg, rgba(10, 5, 20, 0.975) 0%, rgba(10, 5, 20, 0.2) 100%) !important;
      backdrop-filter: saturate(240%) blur(36px);
      -webkit-backdrop-filter: saturate(240%) blur(36px);
      border-bottom: 1px solid rgba(251, 191, 36, 0.1);
      height: 56px;
      z-index: 50;
    }
    [data-testid="stToolbar"] { right: 1.25rem; top: 1rem; }

    .app-hero-wrap {
      position: relative;
      z-index: 6;
      animation: fadeSlideDown 1s cubic-bezier(0.22, 1, 0.36, 1) 0.02s both;
    }
    @keyframes fadeSlideDown {
      0% { opacity: 0; transform: translateY(-32px); }
      100% { opacity: 1; transform: translateY(0); }
    }

    .hero-banner {
      position: relative;
      padding: 2.2rem 2.6rem 2.3rem 2.6rem;
      border-radius: 28px;
      overflow: hidden;
      margin-bottom: 1.5rem;
      background:
        linear-gradient(145deg,
          rgba(52, 26, 96, 0.66) 0%,
          rgba(68, 32, 122, 0.52) 22%,
          rgba(42, 18, 82, 0.68) 52%,
          rgba(70, 28, 112, 0.56) 78%,
          rgba(28, 10, 56, 0.86) 100%);
      backdrop-filter: blur(52px) saturate(240%);
      -webkit-backdrop-filter: blur(52px) saturate(240%);
      border: 1.5px solid rgba(251, 191, 36, 0.12);
      box-shadow:
        0 56px 150px rgba(0, 0, 0, 0.68),
        0 0 0 1.5px rgba(255, 255, 255, 0.05) inset,
        0 0 180px rgba(139, 92, 246, 0.3),
        0 0 120px rgba(251, 146, 60, 0.1);
    }
    .hero-banner::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 36px;
      padding: 1.5px;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.6) 0%,
        rgba(236, 72, 153, 0.35) 25%,
        rgba(168, 85, 247, 0.4) 50%,
        rgba(99, 102, 241, 0.3) 75%,
        rgba(34, 211, 238, 0.5) 100%);
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      pointer-events: none;
      opacity: 0.65;
      animation: gradientBorderShift 10s ease infinite;
      background-size: 200% 200%;
    }
    @keyframes gradientBorderShift {
      0%, 100% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
    }
    .hero-banner::after {
      content: '';
      position: absolute;
      top: -65%;
      left: -22%;
      width: 72%;
      height: 230%;
      background: radial-gradient(ellipse,
        rgba(139, 92, 246, 0.42) 0%,
        rgba(59, 130, 246, 0.14) 48%,
        transparent 72%);
      transform: rotate(-15deg);
      pointer-events: none;
      animation: glowDrift1 14s ease-in-out infinite;
    }
    .hero-inner-glow {
      position: absolute;
      bottom: -65%;
      right: -12%;
      width: 76%;
      height: 230%;
      background: radial-gradient(ellipse,
        rgba(236, 72, 153, 0.26) 0%,
        rgba(251, 146, 60, 0.18) 35%,
        rgba(34, 211, 238, 0.13) 64%,
        transparent 80%);
      transform: rotate(22deg);
      pointer-events: none;
      animation: glowDrift2 16s ease-in-out infinite;
    }
    @keyframes glowDrift1 {
      0%, 100% { transform: rotate(-15deg) translateX(0); }
      50% { transform: rotate(-10deg) translateX(60px); }
    }
    @keyframes glowDrift2 {
      0%, 100% { transform: rotate(22deg) translateX(0); }
      50% { transform: rotate(17deg) translateX(-50px); }
    }

    .hero-border-top {
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3.2px;
      background: linear-gradient(90deg,
        transparent 0%,
        #fbbf24 7%,
        #f97316 17%,
        #fb7185 28%,
        #ec4899 38%,
        #a855f7 50%,
        #6366f1 63%,
        #06b6d4 77%,
        #22d3ee 87%,
        transparent 100%
      );
      box-shadow: 0 0 60px rgba(168, 85, 247, 0.75), 0 0 35px rgba(251, 146, 60, 0.4);
      animation: shimmer 5s linear infinite;
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
      gap: 0.8rem;
      margin-bottom: 1.2rem;
      position: relative;
      z-index: 2;
    }
    .brand-left-main {
      display: flex;
      align-items: center;
      gap: 1.1rem;
    }
    .brand-logo-wrap {
      flex-shrink: 0;
      width: 64px; height: 64px;
      border-radius: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.38) 0%,
        rgba(236, 72, 153, 0.42) 28%,
        rgba(139, 92, 246, 0.52) 56%,
        rgba(34, 211, 238, 0.42) 100%
      );
      border: 1.5px solid rgba(251, 191, 36, 0.32);
      box-shadow:
        0 22px 56px rgba(139, 92, 246, 0.6),
        0 0 0 1.5px rgba(255, 255, 255, 0.11) inset,
        0 0 90px rgba(236, 72, 153, 0.32),
        0 0 55px rgba(251, 146, 60, 0.18);
      backdrop-filter: blur(18px);
      position: relative;
      overflow: hidden;
      animation: logoBounce 6s ease-in-out infinite;
      transform-style: preserve-3d;
    }
    @keyframes logoBounce {
      0%, 100% { transform: translateY(0) rotate(0deg); }
      50% { transform: translateY(-6px) rotate(3deg); }
    }
    .brand-logo-wrap::before {
      content: '';
      position: absolute;
      inset: -1px;
      border-radius: 20px;
      padding: 2px;
      background: linear-gradient(135deg, #fbbf24, #fb7185, #ec4899, #a855f7, #6366f1, #22d3ee, #fbbf24);
      background-size: 300% 300%;
      animation: borderSpin 7s linear infinite;
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.78;
    }
    @keyframes borderSpin {
      0%, 100% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
    }
    .brand-logo-wrap::after {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 22%, rgba(255,255,255,0.32) 50%, transparent 78%);
      transform: translateX(-140%);
      animation: logoShine 7s ease-in-out infinite;
    }
    @keyframes logoShine {
      0%, 40%, 100% { transform: translateX(-140%); }
      60% { transform: translateX(140%); }
    }
    .brand-logo {
      font-size: 2.6rem; line-height: 1; margin: 0;
      filter: drop-shadow(0 3px 12px rgba(0,0,0,0.45));
      animation: logoFloat 4.5s ease-in-out infinite;
    }
    @keyframes logoFloat {
      0%, 100% { transform: translateY(0) rotate(-3deg); }
      50% { transform: translateY(-4px) rotate(3deg); }
    }
    .brand-text-main { display: flex; flex-direction: column; gap: 0.4rem; }
    .brand-title-main {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 800;
      font-size: 2.4rem;
      line-height: 1;
      margin: 0;
      letter-spacing: -0.045em;
      background: linear-gradient(135deg,
        #ffffff 0%,
        #fef3c7 12%,
        #fde68a 25%,
        #e0e7ff 42%,
        #ddd6fe 58%,
        #f5d0fe 72%,
        #f0abfc 84%,
        #a5f3fc 94%,
        #ffffff 100%
      );
      background-size: 340% 340%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: brandGradient 10s ease infinite;
      filter: drop-shadow(0 2px 22px rgba(168, 85, 247, 0.2));
    }
    @keyframes brandGradient {
      0%, 100% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
    }
    .brand-tagline-main {
      font-size: 1.08rem;
      color: rgba(221, 214, 254, 0.97);
      font-weight: 600;
      letter-spacing: 0.05em;
      margin: 0;
    }

    .hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.65rem;
      padding: 0.72rem 1.55rem;
      background:
        linear-gradient(135deg,
          rgba(139, 92, 246, 0.22) 0%,
          rgba(251, 146, 60, 0.14) 45%,
          rgba(236, 72, 153, 0.16) 100%);
      border: 1.5px solid rgba(251, 191, 36, 0.3);
      border-radius: 999px;
      font-size: 0.86rem;
      font-weight: 800;
      color: #fde68a;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      backdrop-filter: blur(16px);
      box-shadow:
        0 0 45px rgba(139, 92, 246, 0.22),
        0 0 0 1.5px rgba(255,255,255,0.05) inset;
      position: relative;
      overflow: hidden;
    }
    .hero-badge::before {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 25%, rgba(255,255,255,0.14) 50%, transparent 75%);
      transform: translateX(-100%);
      animation: badgeShine 5s ease-in-out infinite;
    }
    @keyframes badgeShine {
      0%, 45%, 100% { transform: translateX(-100%); }
      65% { transform: translateX(100%); }
    }
    .hero-dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      background: linear-gradient(135deg, #fbbf24, #fb7185, #ec4899, #a855f7);
      box-shadow: 0 0 18px #fbbf24, 0 0 36px rgba(236, 72, 153, 0.55);
      animation: pulseDotVibrant 1.8s ease-in-out infinite;
    }
    @keyframes pulseDotVibrant {
      0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 18px #fbbf24, 0 0 36px rgba(236, 72, 153, 0.55); }
      50% { opacity: 0.4; transform: scale(1.75); box-shadow: 0 0 28px #ec4899, 0 0 58px rgba(168, 85, 247, 0.65); }
    }

    .hero-meta {
      display: inline-flex;
      align-items: center;
      gap: 1.1rem;
      font-size: 0.95rem;
      color: rgba(221, 214, 254, 0.93);
      font-weight: 600;
    }
    .hero-meta span { display: inline-flex; align-items: center; gap: 0.42rem; }

    .hero-big {
      position: relative;
      z-index: 2;
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 800;
      font-size: 3.4rem;
      line-height: 1.05;
      margin: 1.1rem 0 0.85rem 0;
      letter-spacing: -0.05em;
      max-width: 980px;
    }
    .hero-big .part1 {
      background: linear-gradient(135deg,
        #ffffff 0%,
        #fff7ed 15%,
        #fef3c7 32%,
        #fde68a 48%,
        #e9d5ff 70%,
        #ddd6fe 100%);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      filter: drop-shadow(0 2px 18px rgba(251, 191, 36, 0.1));
    }
    .hero-big .part2 {
      background: linear-gradient(135deg,
        #fbbf24 0%,
        #f97316 10%,
        #fb7185 24%,
        #f472b6 38%,
        #e879f9 52%,
        #c084fc 64%,
        #a78bfa 76%,
        #818cf8 86%,
        #22d3ee 94%,
        #67e8f9 100%
      );
      background-size: 360% 360%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: gradientShiftVibrant 7s ease infinite;
      display: inline;
      filter: drop-shadow(0 0 38px rgba(168, 85, 247, 0.4));
    }
    @keyframes gradientShiftVibrant {
      0%, 100% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
    }
    .hero-sub {
      position: relative;
      z-index: 2;
      font-size: 1.16rem;
      line-height: 1.7;
      color: rgba(226, 232, 240, 0.95);
      max-width: 840px;
      font-weight: 400;
      margin: 0;
    }
    .hero-sub strong {
      color: #e0e7ff;
      font-weight: 700;
      background: linear-gradient(135deg, #fde68a, #f9a8d4, #f0abfc, #a5f3fc);
      background-size: 250% 250%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: gradientShiftVibrant 8s ease infinite;
    }

    .info-strip {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1.1rem;
      margin-bottom: 1.5rem;
      z-index: 6;
      position: relative;
    }
    .info-card {
      background: linear-gradient(145deg,
        rgba(48, 26, 92, 0.74) 0%,
        rgba(34, 18, 68, 0.82) 50%,
        rgba(26, 12, 52, 0.9) 100%);
      backdrop-filter: blur(30px) saturate(200%);
      -webkit-backdrop-filter: blur(30px) saturate(200%);
      border: 1.5px solid rgba(251, 191, 36, 0.11);
      border-radius: 22px;
      padding: 1.2rem 1.25rem;
      transition: all 0.48s cubic-bezier(0.22, 1, 0.36, 1);
      display: flex;
      align-items: flex-start;
      gap: 1.3rem;
      position: relative;
      overflow: hidden;
      box-shadow:
        0 22px 56px rgba(0, 0, 0, 0.48),
        0 0 0 1.5px rgba(255, 255, 255, 0.045) inset,
        0 0 80px rgba(99, 102, 241, 0.1),
        0 0 45px rgba(251, 146, 60, 0.05);
      animation: staggerCardIn 0.8s cubic-bezier(0.22, 1, 0.36, 1) both;
    }
    .info-card:nth-child(1) { animation-delay: 0.14s; }
    .info-card:nth-child(2) { animation-delay: 0.28s; }
    @keyframes staggerCardIn {
      0% { opacity: 0; transform: translateY(28px) scale(0.95); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    .info-card::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 26px;
      padding: 1.5px;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.5) 0%,
        rgba(236, 72, 153, 0.3) 30%,
        rgba(129, 140, 248, 0.4) 60%,
        rgba(34, 211, 238, 0.45) 100%);
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0;
      transition: opacity 0.5s ease;
      pointer-events: none;
    }
    .info-card::after {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2.5px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(251, 191, 36, 0.75) 12%,
        rgba(236, 72, 153, 0.62) 38%,
        rgba(129, 140, 248, 0.78) 64%,
        rgba(34, 211, 238, 0.58) 86%,
        transparent 100%
      );
      opacity: 0.9;
    }
    .info-card::after {
      z-index: 2;
    }
    .info-hover-glow {
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at var(--mx, 50%) var(--my, 0%),
        rgba(251, 191, 36, 0.1) 0%,
        rgba(139, 92, 246, 0.22) 42%,
        transparent 65%
      );
      opacity: 0;
      transition: opacity 0.48s ease;
      pointer-events: none;
    }
    .info-card:hover {
      transform: translateY(-10px) scale(1.018);
      border-color: rgba(251, 191, 36, 0.28);
      box-shadow:
        0 40px 85px rgba(0, 0, 0, 0.6),
        0 0 95px rgba(139, 92, 246, 0.34),
        0 0 55px rgba(251, 146, 60, 0.15),
        0 0 0 1.5px rgba(251, 191, 36, 0.12) inset;
    }
    .info-card:hover::before { opacity: 0.7; }
    .info-card:hover .info-hover-glow { opacity: 1; }
    .info-icon-wrap {
      flex-shrink: 0;
      width: 62px; height: 62px;
      border-radius: 20px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.3) 0%,
        rgba(236, 72, 153, 0.28) 32%,
        rgba(168, 85, 247, 0.34) 66%,
        rgba(34, 211, 238, 0.26) 100%
      );
      border: 1.5px solid rgba(251, 191, 36, 0.32);
      font-size: 1.7rem;
      box-shadow:
        0 12px 32px rgba(139, 92, 246, 0.45),
        0 0 0 1.5px rgba(255, 255, 255, 0.075) inset,
        0 0 34px rgba(251, 146, 60, 0.14);
      position: relative;
      transition: transform 0.48s cubic-bezier(0.22, 1, 0.36, 1);
      z-index: 2;
    }
    .info-icon-wrap::after {
      content: '';
      position: absolute;
      inset: -1px;
      border-radius: 20px;
      padding: 1.5px;
      background: linear-gradient(135deg, #fbbf24, #ec4899, #a855f7, #22d3ee);
      background-size: 200% 200%;
      animation: borderSpin 8s linear infinite;
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.5;
    }
    .info-card:hover .info-icon-wrap {
      transform: rotate(-10deg) scale(1.16);
    }
    .info-body { flex: 1; min-width: 0; z-index: 2; position: relative; }
    .info-label {
      font-size: 0.71rem;
      font-weight: 800;
      letter-spacing: 0.17em;
      text-transform: uppercase;
      color: rgba(251, 191, 36, 0.97);
      margin-bottom: 0.5rem;
    }
    .info-value {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 700;
      font-size: 1.38rem;
      line-height: 1.16;
      color: #f8fafc;
    }
    .accuracy-highlight {
      background: linear-gradient(135deg,
        #6ee7b7 0%,
        #34d399 20%,
        #22d3ee 42%,
        #818cf8 62%,
        #a78bfa 80%,
        #fbbf24 100%);
      background-size: 260% 260%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      font-size: 1.55rem;
      animation: gradientShiftVibrant 7.5s ease infinite;
      filter: drop-shadow(0 0 14px rgba(16, 185, 129, 0.2));
    }

    .input-panel-wrap {
      animation: fadeSlideUp 1s cubic-bezier(0.22, 1, 0.36, 1) 0.4s both;
      z-index: 6;
      position: relative;
    }
    @keyframes fadeSlideUp {
      0% { opacity: 0; transform: translateY(36px); }
      100% { opacity: 1; transform: translateY(0); }
    }

    .input-panel {
      position: relative;
      padding: 2.0rem 2.2rem 2.1rem 2.2rem;
      border-radius: 26px;
      overflow: hidden;
      margin-bottom: 1.6rem;
      background: linear-gradient(180deg,
        rgba(46, 24, 88, 0.74) 0%,
        rgba(34, 18, 68, 0.82) 50%,
        rgba(26, 12, 52, 0.92) 100%
      );
      backdrop-filter: blur(40px) saturate(220%);
      -webkit-backdrop-filter: blur(40px) saturate(220%);
      border: 1.5px solid rgba(251, 191, 36, 0.11);
      box-shadow:
        0 44px 110px rgba(0, 0, 0, 0.62),
        0 0 0 1.5px rgba(255, 255, 255, 0.045) inset,
        0 0 130px rgba(99, 102, 241, 0.16),
        0 0 70px rgba(251, 146, 60, 0.07);
    }
    .input-panel::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3.2px;
      background: linear-gradient(90deg,
        transparent 0%,
        #fbbf24 8%,
        #f97316 22%,
        #fb7185 36%,
        #ec4899 48%,
        #a855f7 60%,
        #6366f1 74%,
        #06b6d4 86%,
        #22d3ee 94%,
        transparent 100%
      );
      background-size: 200% 100%;
      animation: shimmer 6s linear infinite;
      opacity: 0.95;
      z-index: 3;
    }
    .input-panel::after {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 26px;
      padding: 1.5px;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.4) 0%,
        rgba(236, 72, 153, 0.25) 30%,
        rgba(129, 140, 248, 0.3) 60%,
        rgba(34, 211, 238, 0.38) 100%);
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.5;
      pointer-events: none;
    }
    .input-panel-header {
      position: relative;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 0.8rem;
      margin-bottom: 1.55rem;
      padding-bottom: 1.2rem;
      border-bottom: 1px solid rgba(251, 191, 36, 0.08);
    }
    .input-panel-title-wrap { display: flex; align-items: center; gap: 1rem; }
    .input-panel-icon {
      width: 48px; height: 48px;
      border-radius: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.7rem;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.28) 0%,
        rgba(236, 72, 153, 0.28) 35%,
        rgba(168, 85, 247, 0.34) 68%,
        rgba(34, 211, 238, 0.26) 100%
      );
      border: 1.5px solid rgba(251, 191, 36, 0.3);
      box-shadow:
        0 12px 34px rgba(139, 92, 246, 0.42),
        0 0 0 1.5px rgba(255, 255, 255, 0.08) inset,
        0 0 36px rgba(251, 146, 60, 0.12);
      position: relative;
    }
    .input-panel-icon::after {
      content: '';
      position: absolute;
      inset: -1px;
      border-radius: 16px;
      padding: 1.5px;
      background: linear-gradient(135deg, #fbbf24, #ec4899, #a855f7, #22d3ee);
      background-size: 200% 200%;
      animation: borderSpin 9s linear infinite;
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.5;
    }
    .input-panel-title {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 700;
      font-size: 1.35rem;
      background: linear-gradient(135deg, #fef3c7 0%, #f8fafc 25%, #e0e7ff 55%, #ddd6fe 100%);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      margin: 0;
      letter-spacing: -0.028em;
    }
    .input-panel-sub {
      font-size: 0.74rem;
      color: rgba(221, 214, 254, 0.9);
      font-weight: 600;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      margin-top: 0.2rem;
    }
    .input-panel-meta {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.45rem 1rem;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.16) 0%,
        rgba(168, 85, 247, 0.15) 50%,
        rgba(236, 72, 153, 0.11) 100%);
      border: 1.5px solid rgba(251, 191, 36, 0.24);
      border-radius: 999px;
      font-size: 0.68rem;
      font-weight: 800;
      color: #fde68a;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      box-shadow: 0 0 40px rgba(139, 92, 246, 0.16);
      backdrop-filter: blur(10px);
    }

    .fs-section { position: relative; z-index: 2; margin-bottom: 1.2rem; }
    .fs-section-head { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.85rem; }
    .fs-step {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 700;
      font-size: 0.71rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: #180730;
      padding: 0.44rem 1rem;
      background: linear-gradient(135deg,
        #fbbf24 0%,
        #f97316 20%,
        #fb7185 42%,
        #ec4899 62%,
        #a855f7 100%);
      border-radius: 13px;
      box-shadow:
        0 6px 20px rgba(236, 72, 153, 0.5),
        0 0 0 1.5px rgba(255,255,255,0.12) inset,
        0 0 30px rgba(251, 146, 60, 0.22);
      position: relative;
      overflow: hidden;
    }
    .fs-step::after {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 28%, rgba(255,255,255,0.42) 50%, transparent 72%);
      transform: translateX(-100%);
      animation: stepShine 4s ease-in-out infinite;
    }
    @keyframes stepShine {
      0%, 48%, 100% { transform: translateX(-100%); }
      68% { transform: translateX(100%); }
    }
    .fs-section-title {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 600;
      font-size: 1.12rem;
      color: #e0e7ff;
      letter-spacing: -0.018em;
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .fs-section-icon {
      width: 38px; height: 38px;
      border-radius: 13px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-size: 1.05rem;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.24) 0%,
        rgba(168, 85, 247, 0.26) 100%
      );
      border: 1.5px solid rgba(251, 191, 36, 0.28);
      color: #fde68a;
      box-shadow:
        0 0 20px rgba(139, 92, 246, 0.18),
        0 0 0 1px rgba(255,255,255,0.05) inset;
    }

    .input-card {
      background: linear-gradient(160deg,
        rgba(52, 28, 96, 0.72) 0%,
        rgba(38, 20, 74, 0.82) 50%,
        rgba(30, 15, 58, 0.9) 100%);
      border-radius: 18px;
      padding: 0.95rem 1rem 1rem 1rem;
      border: 1.5px solid rgba(251, 191, 36, 0.09);
      box-shadow:
        0 10px 30px rgba(0, 0, 0, 0.42),
        0 0 0 1.5px rgba(255, 255, 255, 0.04) inset;
      transition: all 0.42s cubic-bezier(0.22, 1, 0.36, 1);
      position: relative;
      overflow: hidden;
    }
    .input-card::before {
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at var(--mx, 50%) var(--my, 50%),
        rgba(251, 191, 36, 0.08) 0%,
        rgba(139, 92, 246, 0.2) 52%,
        transparent 70%
      );
      opacity: 0;
      transition: opacity 0.42s ease;
      pointer-events: none;
    }
    .input-card::after {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1.8px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(251, 191, 36, 0.5) 22%,
        rgba(236, 72, 153, 0.44) 48%,
        rgba(129, 140, 248, 0.55) 74%,
        transparent 100%
      );
      opacity: 0.65;
    }
    .input-card:hover {
      border-color: rgba(251, 191, 36, 0.24);
      box-shadow:
        0 18px 46px rgba(0, 0, 0, 0.54),
        0 0 0 1.5px rgba(251, 191, 36, 0.1),
        0 0 58px rgba(99, 102, 241, 0.2),
        0 0 32px rgba(251, 146, 60, 0.08),
        0 0 0 1.5px rgba(255, 255, 255, 0.05) inset;
      transform: translateY(-5px);
    }
    .input-card:hover::before { opacity: 1; }

    label, .stSelectbox label p, .stNumberInput label, .stSlider label p {
      color: #e0e7ff !important;
      font-weight: 600 !important;
      font-size: 1.02rem !important;
      margin-bottom: 0.55rem !important;
      letter-spacing: 0.018em;
    }
    .stSelectbox label, .stNumberInput label, .stSlider label {
      margin-bottom: 0.65rem !important;
      padding-bottom: 0 !important;
    }

    .stSelectbox [data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input {
      background: linear-gradient(180deg,
        rgba(24, 12, 50, 0.96) 0%,
        rgba(18, 8, 40, 0.98) 100%) !important;
      border: 1.5px solid rgba(251, 191, 36, 0.11) !important;
      border-radius: 12px !important;
      color: #f1f5f9 !important;
      font-weight: 600 !important;
      font-size: 1.05rem !important;
      min-height: 50px;
      transition: all 0.32s cubic-bezier(0.22, 1, 0.36, 1) !important;
      padding-left: 1rem !important;
      padding-right: 1rem !important;
      box-shadow:
        inset 0 2px 5px rgba(0, 0, 0, 0.4),
        0 0 0 0 rgba(251, 191, 36, 0) !important;
    }
    .stSelectbox [data-baseweb="select"] > div:hover,
    .stNumberInput input:hover,
    .stTextInput input:hover {
      border-color: rgba(251, 191, 36, 0.3) !important;
      box-shadow:
        inset 0 2.5px 6px rgba(0, 0, 0, 0.4),
        0 0 0 5.5px rgba(139, 92, 246, 0.13) !important;
    }
    .stSelectbox [data-baseweb="select"]:focus-within > div,
    .stNumberInput:focus-within input,
    .stTextInput:focus-within input {
      border-color: rgba(168, 85, 247, 0.65) !important;
      background: linear-gradient(180deg, rgba(30, 15, 60, 0.98) 0%, rgba(24, 12, 50, 0.99) 100%) !important;
      box-shadow:
        0 0 0 6.5px rgba(139, 92, 246, 0.2),
        0 0 42px rgba(168, 85, 247, 0.32),
        0 0 22px rgba(251, 191, 36, 0.1),
        inset 0 2.5px 6px rgba(0, 0, 0, 0.4) !important;
    }
    [data-baseweb="select"] svg { color: #c4b5fd !important; }
    [data-baseweb="popover"] {
      background: linear-gradient(180deg, #220e4a 0%, #190a38 100%) !important;
      border: 1.5px solid rgba(251, 191, 36, 0.17) !important;
      border-radius: 18px !important;
      box-shadow:
        0 36px 80px rgba(0,0,0,0.68),
        0 0 70px rgba(99, 102, 241, 0.26),
        0 0 35px rgba(251, 146, 60, 0.08) !important;
      backdrop-filter: blur(26px);
      padding: 0.4rem !important;
    }
    [data-baseweb="popover"] ul li {
      color: #e2e8f0 !important;
      font-size: 1.02rem !important;
      font-weight: 500 !important;
      padding: 0.85rem 1.15rem !important;
      transition: all 0.24s ease;
      border-radius: 10px;
      margin: 0.18rem 0.2rem;
    }
    [data-baseweb="popover"] ul li:hover {
      background: linear-gradient(90deg,
        rgba(251, 191, 36, 0.14) 0%,
        rgba(139, 92, 246, 0.28) 50%,
        rgba(168, 85, 247, 0.18) 100%) !important;
      color: #ffffff !important;
      transform: translateX(3px);
    }

    .stNumberInput button[aria-label="Increase"],
    .stNumberInput button[aria-label="Decrease"] {
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.2) 0%,
        rgba(168, 85, 247, 0.22) 50%,
        rgba(236, 72, 153, 0.18) 100%) !important;
      border: 1.5px solid rgba(251, 191, 36, 0.32) !important;
      color: #fde68a !important;
      font-size: 1.2rem !important;
      min-width: 44px !important;
      min-height: 50px !important;
      border-radius: 10px !important;
      transition: all 0.24s ease !important;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.05) inset;
    }
    .stNumberInput button[aria-label="Increase"]:hover,
    .stNumberInput button[aria-label="Decrease"]:hover {
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.38) 0%,
        rgba(168, 85, 247, 0.36) 50%,
        rgba(236, 72, 153, 0.32) 100%) !important;
      color: #ffffff !important;
      transform: scale(1.12);
      box-shadow:
        0 6px 16px rgba(139, 92, 246, 0.4),
        0 0 0 1px rgba(255,255,255,0.1) inset;
    }

    [data-baseweb="slider"] [data-testid="stTickBar"] > div {
      background: rgba(251, 191, 36, 0.22) !important;
      height: 3px !important;
    }
    [data-baseweb="slider"] > div > div > div {
      background: linear-gradient(90deg,
        #fbbf24 0%,
        #f97316 18%,
        #fb7185 36%,
        #ec4899 54%,
        #a855f7 72%,
        #6366f1 100%
      ) !important;
      height: 5.5px !important;
      border-radius: 999px;
      box-shadow: 0 0 20px rgba(168, 85, 247, 0.5), 0 0 10px rgba(251, 146, 60, 0.3);
    }
    [data-baseweb="slider"] [role="slider"] [data-testid="stTickBar"] { display: none !important; }
    [data-baseweb="slider"] [role="slider"] {
      z-index: 30 !important;
    }
    [data-baseweb="slider"] [role="slider"] > div > div {
      background: linear-gradient(135deg, #fef3c7, #f0abfc, #a5f3fc) !important;
      box-shadow:
        0 0 0 6px rgba(251, 191, 36, 0.18),
        0 0 26px rgba(168, 85, 247, 0.8),
        0 0 48px rgba(236, 72, 153, 0.4) !important;
      width: 22px !important;
      height: 22px !important;
      border: 2.5px solid #fff !important;
      transition: transform 0.22s cubic-bezier(0.22, 1, 0.36, 1);
    }
    [data-baseweb="slider"] [role="slider"]:hover > div > div {
      transform: scale(1.3);
    }
    [data-testid="stSliderThumbValue"] {
      background:
        linear-gradient(180deg,
          rgba(255, 255, 255, 0.22) 0%,
          rgba(255, 255, 255, 0) 50%,
          rgba(0, 0, 0, 0.25) 100%),
        linear-gradient(135deg,
          #fbbf24 0%,
          #f59e0b 14%,
          #fb923c 28%,
          #f97316 42%,
          #fb7185 58%,
          #ec4899 72%,
          #c026d3 86%,
          #a855f7 100%) !important;
      background-color: #fb923c !important;
      background-repeat: no-repeat, no-repeat !important;
      background-size: 100% 100%, 360% 360% !important;
      background-position: 0% 0%, 0% 50% !important;
      animation: gradientShiftVibrant 6.5s ease infinite !important;
      color: #0a0414 !important;
      font-weight: 900 !important;
      font-size: 0.92rem !important;
      min-width: 66px !important;
      max-width: none !important;
      width: max-content !important;
      height: 34px !important;
      min-height: 34px !important;
      max-height: 34px !important;
      padding: 0 0.85rem !important;
      margin: 0 !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      line-height: 34px !important;
      white-space: nowrap !important;
      box-sizing: border-box !important;
      border: 1.5px solid rgba(255, 255, 255, 0.48) !important;
      border-radius: 9px !important;
      top: -2.9rem !important;
      left: 50% !important;
      transform: translateX(-50%) !important;
      translate: none !important;
      box-shadow:
        0 9px 20px rgba(0, 0, 0, 0.5),
        0 0 0 1.5px rgba(10, 4, 20, 0.42),
        0 0 20px rgba(251, 146, 60, 0.42),
        0 0 36px rgba(168, 85, 247, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.42) !important;
      z-index: 80 !important;
      position: absolute !important;
      pointer-events: none !important;
      font-family: 'Space Grotesk', 'Poppins', sans-serif !important;
      letter-spacing: -0.02em !important;
      text-align: center !important;
      text-shadow: 0 1px 0 rgba(255, 255, 255, 0.42) !important;
      overflow: visible !important;
    }
    [data-testid="stSliderThumbValue"]::before {
      content: '' !important;
      display: block !important;
      visibility: visible !important;
      opacity: 1 !important;
      position: absolute !important;
      left: 50% !important;
      top: 50% !important;
      transform: translate(-50%, -50%) !important;
      width: calc(100% - 0.55rem) !important;
      max-width: calc(100% - 0.55rem) !important;
      min-width: 70% !important;
      height: calc(100% - 0.4rem) !important;
      max-height: calc(100% - 0.4rem) !important;
      min-height: 78% !important;
      background:
        linear-gradient(180deg,
          rgba(255, 255, 255, 0.98) 0%,
          rgba(240, 171, 252, 0.95) 50%,
          rgba(165, 243, 252, 0.95) 100%) !important;
      background-color: #ffffff !important;
      border: 2.5px solid #ffffff !important;
      border-radius: 5px !important;
      box-shadow:
        0 3px 10px rgba(12, 5, 24, 0.3),
        0 0 16px rgba(236, 72, 153, 0.25),
        0 0 26px rgba(168, 85, 247, 0.18),
        inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
      z-index: 1 !important;
      pointer-events: none !important;
      overflow: visible !important;
    }
    [data-testid="stSliderThumbValue"] > div,
    [data-testid="stSliderThumbValue"] > span,
    [data-testid="stSliderThumbValue"] > label {
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      width: 100% !important;
      height: 100% !important;
      min-width: 0 !important;
      min-height: 0 !important;
      padding: 0 !important;
      margin: 0 !important;
      background: transparent !important;
      background-image: none !important;
      background-color: transparent !important;
      border: none !important;
      box-shadow: none !important;
      line-height: 34px !important;
      text-align: center !important;
      position: relative !important;
      z-index: 5 !important;
      color: inherit !important;
      font-weight: inherit !important;
      font-size: inherit !important;
      font-family: inherit !important;
      letter-spacing: inherit !important;
      text-shadow: inherit !important;
    }
    [data-testid="stSliderThumbValue"] * {
      position: relative !important;
      z-index: 5 !important;
      background: transparent !important;
      background-color: transparent !important;
      background-image: none !important;
      border: none !important;
      box-shadow: none !important;
      outline: none !important;
      padding: 0 !important;
      margin: 0 !important;
      color: inherit !important;
      font-weight: inherit !important;
      font-size: inherit !important;
      font-family: inherit !important;
      line-height: inherit !important;
      letter-spacing: inherit !important;
      text-shadow: inherit !important;
      width: auto !important;
      height: auto !important;
      min-width: 0 !important;
      min-height: 0 !important;
      transform: none !important;
      translate: none !important;
      border-radius: 0 !important;
      text-align: center !important;
      visibility: visible !important;
      opacity: 1 !important;
      display: inline !important;
    }
    [data-testid="stSliderThumbValue"] [role="slider"],
    [data-testid="stSliderThumbValue"] [data-testid*="Thumb"],
    [data-testid="stSliderThumbValue"] [data-testid*="thumb"],
    [data-testid="stSliderThumbValue"] [data-testid*="Slider"],
    [data-testid="stSliderThumbValue"] [data-testid*="slider"],
    [data-testid="stSliderThumbValue"] [class*="thumb"],
    [data-testid="stSliderThumbValue"] [class*="Thumb"],
    [data-testid="stSliderThumbValue"] [class*="handle"],
    [data-testid="stSliderThumbValue"] [class*="Handle"],
    [data-testid="stSliderThumbValue"] [class*="knob"],
    [data-testid="stSliderThumbValue"] [class*="Knob"],
    [data-testid="stSliderThumbValue"] svg,
    [data-testid="stSliderThumbValue"] img,
    [data-testid="stSliderThumbValue"] canvas,
    [data-testid="stSliderThumbValue"] button,
    [data-testid="stSliderThumbValue"] input {
      display: none !important;
      visibility: hidden !important;
      width: 0 !important;
      height: 0 !important;
      min-width: 0 !important;
      min-height: 0 !important;
      max-width: 0 !important;
      max-height: 0 !important;
      opacity: 0 !important;
      pointer-events: none !important;
      position: absolute !important;
      left: -99999px !important;
      top: -99999px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
      background: transparent !important;
      background-image: none !important;
      border: none !important;
      box-shadow: none !important;
      padding: 0 !important;
      margin: 0 !important;
    }
    [data-testid="stSliderThumbValue"]::after,
    [data-testid="stSliderThumbValue"] > div::before,
    [data-testid="stSliderThumbValue"] > div::after,
    [data-testid="stSliderThumbValue"] > span::before,
    [data-testid="stSliderThumbValue"] > span::after,
    [data-testid="stSliderThumbValue"] > label::before,
    [data-testid="stSliderThumbValue"] > label::after,
    [data-testid="stSliderThumbValue"] *::before,
    [data-testid="stSliderThumbValue"] *::after {
      display: none !important;
      visibility: hidden !important;
      content: none !important;
      background: transparent !important;
      background-image: none !important;
      background-color: transparent !important;
      border: none !important;
      box-shadow: none !important;
      width: 0 !important;
      height: 0 !important;
      min-width: 0 !important;
      min-height: 0 !important;
      max-width: 0 !important;
      max-height: 0 !important;
      opacity: 0 !important;
      position: absolute !important;
      left: -99999px !important;
      top: -99999px !important;
      overflow: hidden !important;
      clip: rect(0, 0, 0, 0) !important;
    }

    .stNumberInput, .stSelectbox, .stSlider { margin-bottom: 0.5rem !important; }
    .stNumberInput > div, .stSelectbox > div, .stSlider > div {
      padding-bottom: 0 !important;
      margin-bottom: 0 !important;
    }

    .form-submit-wrap {
      position: relative;
      z-index: 2;
      margin-top: 1.4rem;
      padding-top: 1.2rem;
      border-top: 1px solid rgba(251, 191, 36, 0.08);
    }
    .stFormSubmitButton > button, button[kind="primary"] {
      position: relative !important;
      width: 100% !important;
      padding: 0.95rem 1.4rem !important;
      font-family: 'Space Grotesk', 'Poppins', sans-serif !important;
      font-size: 0.88rem !important;
      font-weight: 800 !important;
      color: #ffffff !important;
      letter-spacing: 0.08em !important;
      background: linear-gradient(135deg,
        #f59e0b 0%,
        #f97316 12%,
        #fb7185 28%,
        #ec4899 44%,
        #c026d3 60%,
        #9333ea 76%,
        #7c3aed 88%,
        #6366f1 100%
      ) !important;
      background-size: 320% 320% !important;
      border: none !important;
      border-radius: 15px !important;
      transition: all 0.45s cubic-bezier(0.22, 1, 0.36, 1) !important;
      box-shadow:
        0 16px 45px rgba(124, 58, 237, 0.58),
        0 0 65px rgba(168, 85, 247, 0.32),
        0 0 36px rgba(251, 146, 60, 0.2),
        0 0 0 1.5px rgba(251, 191, 36, 0.5) inset,
        0 -2.5px 0 rgba(255, 255, 255, 0.18) inset !important;
      overflow: hidden !important;
      text-transform: uppercase;
      min-height: 52px;
      animation: gradientShiftVibrant 8s ease infinite;
    }
    .stFormSubmitButton > button::before {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, transparent 18%, rgba(255, 255, 255, 0.32) 50%, transparent 82%);
      transform: translateX(-100%);
      transition: transform 0.9s ease;
    }
    .stFormSubmitButton > button::after {
      content: '';
      position: absolute;
      inset: -3px;
      border-radius: 21px;
      padding: 3px;
      background: linear-gradient(135deg,
        #fbbf24, #f97316, #fb7185, #ec4899,
        #a855f7, #6366f1, #22d3ee, #fbbf24);
      background-size: 350% 350%;
      animation: borderSpin 5s linear infinite;
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.75;
      pointer-events: none;
    }
    .stFormSubmitButton > button:hover {
      transform: translateY(-5px) scale(1.008) !important;
      background-size: 360% 360% !important;
      background-position: 100% 100% !important;
      box-shadow:
        0 32px 75px rgba(99, 102, 241, 0.72),
        0 0 110px rgba(168, 85, 247, 0.48),
        0 0 65px rgba(251, 146, 60, 0.28),
        0 0 0 1.5px rgba(253, 224, 71, 0.65) inset,
        0 -2.5px 0 rgba(255, 255, 255, 0.22) inset !important;
    }
    .stFormSubmitButton > button:hover::before { transform: translateX(100%); }
    .stFormSubmitButton > button:active { transform: translateY(-1px) scale(0.992) !important; }

    .result-hero {
      position: relative;
      padding: 2.3rem 2.2rem 2.1rem 2.2rem;
      border-radius: 26px;
      overflow: hidden;
      text-align: center;
      background:
        linear-gradient(160deg,
          rgba(44, 20, 88, 0.88) 0%,
          rgba(60, 28, 112, 0.76) 28%,
          rgba(36, 16, 74, 0.88) 58%,
          rgba(52, 22, 96, 0.82) 78%,
          rgba(22, 10, 48, 0.97) 100%);
      backdrop-filter: blur(50px) saturate(220%);
      -webkit-backdrop-filter: blur(50px) saturate(220%);
      border: 1.5px solid rgba(251, 191, 36, 0.13);
      box-shadow:
        0 38px 100px rgba(0, 0, 0, 0.68),
        0 0 110px rgba(99, 102, 241, 0.22),
        0 0 0 1.5px rgba(255, 255, 255, 0.05) inset;
      animation: resultBounceIn 1s cubic-bezier(0.34, 1.56, 0.64, 1);
      margin-bottom: 1.3rem;
    }
    @keyframes resultBounceIn {
      0% { opacity: 0; transform: scale(0.88) translateY(36px); }
      60% { opacity: 1; transform: scale(1.03) translateY(-6px); }
      100% { opacity: 1; transform: scale(1) translateY(0); }
    }
    .result-hero::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 26px;
      padding: 1.5px;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.6) 0%,
        rgba(236, 72, 153, 0.35) 28%,
        rgba(168, 85, 247, 0.4) 52%,
        rgba(99, 102, 241, 0.32) 76%,
        rgba(34, 211, 238, 0.55) 100%);
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      pointer-events: none;
      opacity: 0.65;
      animation: gradientBorderShift 10s ease infinite;
      background-size: 200% 200%;
    }
    .result-hero::after {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3.5px;
      background: linear-gradient(90deg,
        transparent 0%,
        #fbbf24 8%,
        #f97316 22%,
        #ec4899 40%,
        #a855f7 56%,
        #6366f1 72%,
        #06b6d4 86%,
        #22d3ee 94%,
        transparent 100%
      );
      background-size: 200% 100%;
      animation: shimmer 4.5s linear infinite;
    }
    .result-glow {
      position: absolute;
      top: -35%;
      left: 50%;
      transform: translateX(-50%);
      width: 85%;
      height: 85%;
      background: radial-gradient(ellipse, var(--result-glow, rgba(16, 185, 129, 0.28)) 0%, transparent 68%);
      pointer-events: none;
      animation: resultGlow 3.5s ease-in-out infinite;
    }
    @keyframes resultGlow {
      0%, 100% { opacity: 0.55; transform: translateX(-50%) scale(1); }
      50% { opacity: 1; transform: translateX(-50%) scale(1.12); }
    }
    .result-tag {
      display: inline-flex;
      align-items: center;
      gap: 0.6rem;
      padding: 0.45rem 1.2rem;
      border-radius: 999px;
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 800;
      font-size: 0.7rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      margin-bottom: 1.25rem;
      backdrop-filter: blur(18px);
      position: relative;
      overflow: hidden;
      animation: tagPulse 2.4s ease-in-out infinite;
      z-index: 2;
    }
    @keyframes tagPulse {
      0%, 100% { transform: scale(1); }
      50% { transform: scale(1.04); }
    }
    .result-tag.profit {
      background: linear-gradient(135deg,
        rgba(16, 185, 129, 0.24) 0%,
        rgba(34, 211, 238, 0.16) 50%,
        rgba(251, 191, 36, 0.12) 100%);
      border: 1.5px solid rgba(52, 211, 153, 0.55);
      color: #6ee7b7;
      box-shadow:
        0 0 50px rgba(16, 185, 129, 0.25),
        0 0 0 1.5px rgba(255,255,255,0.06) inset;
    }
    .result-tag.loss {
      background: linear-gradient(135deg,
        rgba(239, 68, 68, 0.26) 0%,
        rgba(251, 113, 133, 0.18) 50%,
        rgba(251, 191, 36, 0.1) 100%);
      border: 1.5px solid rgba(248, 113, 113, 0.58);
      color: #fecaca;
      box-shadow:
        0 0 50px rgba(239, 68, 68, 0.26),
        0 0 0 1.5px rgba(255,255,255,0.06) inset;
    }
    .result-figure {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 900;
      font-size: 5.4rem;
      line-height: 1;
      margin: 0 0 0.95rem 0;
      letter-spacing: -0.05em;
      position: relative;
      z-index: 2;
    }
    .result-figure.profit {
      background: linear-gradient(135deg,
        #d1fae5 0%,
        #6ee7b7 14%,
        #34d399 28%,
        #10b981 42%,
        #22d3ee 58%,
        #67e8f9 72%,
        #a5f3fc 86%,
        #fde68a 100%
      );
      background-size: 300% 300%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: valuePop 1.3s cubic-bezier(0.34, 1.56, 0.64, 1) 0.22s both, gradientShiftVibrant 5.5s ease infinite;
      filter: drop-shadow(0 0 50px rgba(16, 185, 129, 0.5));
    }
    .result-figure.loss {
      background: linear-gradient(135deg,
        #fee2e2 0%,
        #fecaca 14%,
        #fca5a5 28%,
        #f87171 42%,
        #fb7185 58%,
        #f472b6 72%,
        #fbbf24 86%,
        #fde68a 100%
      );
      background-size: 300% 300%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: valuePop 1.3s cubic-bezier(0.34, 1.56, 0.64, 1) 0.22s both, gradientShiftVibrant 5.5s ease infinite;
      filter: drop-shadow(0 0 50px rgba(239, 68, 68, 0.5));
    }
    @keyframes valuePop {
      0% { opacity: 0; transform: scale(0.45); filter: blur(30px) drop-shadow(0 0 0 transparent); }
      60% { opacity: 1; transform: scale(1.15); filter: blur(0); }
      100% { transform: scale(1); }
    }
    .result-caption {
      font-size: 1.18rem;
      font-weight: 600;
      color: rgba(226, 232, 240, 0.96);
      margin: 0 0 1.7rem 0;
      position: relative;
      z-index: 2;
    }

    .stats-triplet {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 1rem;
      position: relative;
      z-index: 2;
    }
    .stat-pill {
      padding: 1.1rem 1.05rem;
      border-radius: 18px;
      background: linear-gradient(160deg,
        rgba(34, 16, 72, 0.8) 0%,
        rgba(26, 12, 56, 0.86) 100%);
      border: 1.5px solid rgba(251, 191, 36, 0.12);
      backdrop-filter: blur(20px);
      transition: all 0.48s cubic-bezier(0.22, 1, 0.36, 1);
      position: relative;
      overflow: hidden;
      text-align: left;
      box-shadow:
        0 0 0 1.5px rgba(255, 255, 255, 0.05) inset,
        0 8px 24px rgba(0,0,0,0.38);
      animation: statIn 0.8s cubic-bezier(0.34, 1.56, 0.64, 1) both;
    }
    .stat-pill:nth-child(1) { animation-delay: 0.48s; }
    .stat-pill:nth-child(2) { animation-delay: 0.58s; }
    .stat-pill:nth-child(3) { animation-delay: 0.68s; }
    @keyframes statIn {
      0% { opacity: 0; transform: translateY(22px) scale(0.94); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    .stat-pill::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1.8px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(251, 191, 36, 0.55) 28%,
        rgba(236, 72, 153, 0.48) 52%,
        rgba(129, 140, 248, 0.6) 74%,
        transparent 100%
      );
    }
    .stat-pill::after {
      content: '';
      position: absolute;
      inset: 0;
      background: radial-gradient(circle at var(--mx, 50%) var(--my, 0%),
        rgba(251, 191, 36, 0.06) 0%,
        rgba(139, 92, 246, 0.16) 50%,
        transparent 70%
      );
      opacity: 0;
      transition: opacity 0.48s ease;
      pointer-events: none;
    }
    .stat-pill:hover {
      transform: translateY(-7px) scale(1.03);
      border-color: rgba(251, 191, 36, 0.28);
      box-shadow:
        0 24px 52px rgba(0,0,0,0.55),
        0 0 58px rgba(99, 102, 241, 0.26),
        0 0 32px rgba(251, 146, 60, 0.1),
        0 0 0 1.5px rgba(255,255,255,0.07) inset;
    }
    .stat-pill:hover::after { opacity: 1; }
    .stat-top { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.65rem; }
    .stat-circle {
      flex-shrink: 0;
      width: 42px; height: 42px;
      border-radius: 13px;
      border: 1.5px solid rgba(251, 191, 36, 0.32);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.05rem;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.26) 0%,
        rgba(236, 72, 153, 0.22) 35%,
        rgba(168, 85, 247, 0.28) 70%,
        rgba(34, 211, 238, 0.2) 100%
      );
      box-shadow:
        0 0 18px rgba(139, 92, 246, 0.32),
        0 0 0 1.5px rgba(255,255,255,0.06) inset;
      transition: transform 0.48s cubic-bezier(0.22, 1, 0.36, 1);
    }
    .stat-pill:hover .stat-circle {
      transform: rotate(-10deg) scale(1.14);
    }
    .stat-k {
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 800;
      color: rgba(251, 191, 36, 0.95);
      margin-bottom: 0.32rem;
    }
    .stat-v {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 800;
      font-size: 1.5rem;
      line-height: 1.14;
      background: linear-gradient(135deg, #ffffff 0%, #fef3c7 30%, #e0e7ff 60%, #ddd6fe 100%);
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .pm-wrap {
      position: relative;
      z-index: 6;
      margin-top: 1.3rem;
      animation: pmFadeIn 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.3s both;
    }
    @keyframes pmFadeIn {
      0% { opacity: 0; transform: translateY(26px); }
      100% { opacity: 1; transform: translateY(0); }
    }
    .pm-head {
      display: flex;
      align-items: center;
      gap: 0.9rem;
      margin-bottom: 0.9rem;
      padding-left: 0.15rem;
    }
    .pm-bar {
      width: 6px;
      height: 48px;
      border-radius: 999px;
      background: linear-gradient(180deg, #fbbf24 0%, #f97316 20%, #fb7185 40%, #ec4899 55%, #a855f7 72%, #6366f1 88%, #22d3ee 100%);
      box-shadow: 0 0 24px rgba(168, 85, 247, 0.55), 0 0 12px rgba(251, 191, 36, 0.35);
      flex-shrink: 0;
    }
    .pm-head-text h2 {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 800;
      font-size: 1.72rem;
      line-height: 1;
      margin: 0 0 0.28rem 0;
      letter-spacing: -0.038em;
      background: linear-gradient(135deg,
        #ffffff 0%,
        #fef3c7 14%,
        #fde68a 28%,
        #f9a8d4 48%,
        #f0abfc 62%,
        #ddd6fe 78%,
        #c4b5fd 90%,
        #a5f3fc 100%);
      background-size: 320% 320%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: gradientShiftVibrant 8s ease infinite;
      filter: drop-shadow(0 2px 20px rgba(168, 85, 247, 0.2));
    }
    .pm-head-text p {
      margin: 0;
      font-size: 0.8rem;
      color: rgba(221, 214, 254, 0.9);
      font-weight: 500;
      letter-spacing: 0.015em;
    }
    .pm-single-pane {
      position: relative;
      border-radius: 26px;
      overflow: hidden;
      background:
        radial-gradient(ellipse at 12% 0%, rgba(251, 191, 36, 0.18) 0%, transparent 48%),
        radial-gradient(ellipse at 88% 100%, rgba(34, 211, 238, 0.16) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(168, 85, 247, 0.1) 0%, transparent 60%),
        linear-gradient(160deg,
          rgba(58, 28, 108, 0.82) 0%,
          rgba(44, 20, 86, 0.78) 28%,
          rgba(36, 16, 72, 0.86) 62%,
          rgba(30, 13, 62, 0.92) 100%);
      backdrop-filter: blur(46px) saturate(230%);
      -webkit-backdrop-filter: blur(46px) saturate(230%);
      border: 1.6px solid rgba(251, 191, 36, 0.12);
      box-shadow:
        0 30px 80px rgba(0, 0, 0, 0.62),
        0 0 0 1.6px rgba(255, 255, 255, 0.05) inset,
        0 0 120px rgba(99, 102, 241, 0.12),
        0 0 80px rgba(251, 146, 60, 0.07);
      animation: pmPaneIn 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.25s both;
    }
    @keyframes pmPaneIn {
      0% { opacity: 0; transform: translateY(30px) scale(0.97); }
      100% { opacity: 1; transform: translateY(0) scale(1); }
    }
    .pm-single-pane::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 26px;
      padding: 1.6px;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.55) 0%,
        rgba(236, 72, 153, 0.35) 25%,
        rgba(168, 85, 247, 0.42) 52%,
        rgba(129, 140, 248, 0.4) 75%,
        rgba(34, 211, 238, 0.5) 100%);
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.55;
      pointer-events: none;
      z-index: 2;
    }
    .pm-single-pane::after {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2.4px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(251, 191, 36, 0.8) 15%,
        rgba(236, 72, 153, 0.65) 35%,
        rgba(168, 85, 247, 0.75) 55%,
        rgba(129, 140, 248, 0.7) 75%,
        rgba(34, 211, 238, 0.75) 88%,
        transparent 100%
      );
      opacity: 1;
      z-index: 3;
    }
    .pm-pane-inner {
      position: relative;
      z-index: 4;
      padding: 1.5rem 1.6rem 1.4rem 1.6rem;
    }
    .pm-steps {
      display: flex;
      flex-direction: column;
      gap: 0.95rem;
      margin-top: 0.2rem;
    }
    .pm-step {
      display: flex;
      align-items: flex-start;
      gap: 0.95rem;
      padding: 0.85rem 0.95rem;
      border-radius: 16px;
      background: linear-gradient(135deg,
        rgba(255, 255, 255, 0.035) 0%,
        rgba(255, 255, 255, 0.015) 100%);
      border: 1px solid rgba(255, 255, 255, 0.05);
      transition: all 0.45s cubic-bezier(0.22, 1, 0.36, 1);
      position: relative;
      overflow: hidden;
    }
    .pm-step::after {
      content: '';
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 3.5px;
      background: var(--step-accent, #fbbf24);
      box-shadow: 0 0 14px var(--step-accent, #fbbf24);
      opacity: 0.85;
    }
    .pm-step:hover {
      background: linear-gradient(135deg,
        rgba(255, 255, 255, 0.06) 0%,
        rgba(255, 255, 255, 0.025) 100%);
      border-color: rgba(255, 255, 255, 0.09);
      transform: translateX(4px);
    }
    .pm-step-icon {
      width: 44px;
      height: 44px;
      flex-shrink: 0;
      border-radius: 14px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.35rem;
      background: var(--step-bg, linear-gradient(135deg, rgba(251,191,36,0.3), rgba(251,113,133,0.3)));
      border: 1.5px solid var(--step-border, rgba(251,191,36,0.35));
      box-shadow:
        0 8px 22px var(--step-shadow, rgba(251,191,36,0.2)),
        0 0 0 1.5px rgba(255,255,255,0.06) inset;
      position: relative;
    }
    .pm-step-icon::after {
      content: '';
      position: absolute;
      inset: -1px;
      border-radius: 14px;
      padding: 1.5px;
      background: var(--step-border-grad, linear-gradient(135deg, #fbbf24, #ec4899));
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.55;
    }
    .pm-step-body { flex: 1; min-width: 0; }
    .pm-step-label {
      display: flex;
      align-items: baseline;
      gap: 0.55rem;
      margin-bottom: 0.25rem;
    }
    .pm-step-num {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 800;
      font-size: 0.72rem;
      padding: 0.12rem 0.48rem;
      border-radius: 999px;
      background: var(--step-pill-bg, rgba(251,191,36,0.18));
      color: var(--step-pill-text, #fde68a);
      letter-spacing: 0.04em;
      line-height: 1;
      border: 1px solid var(--step-pill-border, rgba(251,191,36,0.3));
    }
    .pm-step-title {
      font-family: 'Space Grotesk', 'Poppins', sans-serif;
      font-weight: 750;
      font-size: 0.94rem;
      line-height: 1.15;
      margin: 0;
      background: var(--step-title-grad, linear-gradient(135deg, #fef3c7, #fde68a, #f9a8d4));
      background-size: 220% 220%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: gradientShiftVibrant 9s ease infinite;
    }
    .pm-step-text {
      font-size: 0.8rem;
      line-height: 1.52;
      color: rgba(226, 232, 240, 0.93);
      margin: 0;
      font-weight: 450;
    }
    .pm-step-text strong {
      color: var(--step-strong, #fde68a);
      font-weight: 720;
    }
    .pm-section-gap { margin-top: 1.1rem; }
    .pm-inputs-pane {
      position: relative;
      border-radius: 24px;
      overflow: hidden;
      background:
        radial-gradient(ellipse at 88% 0%, rgba(236, 72, 153, 0.15) 0%, transparent 48%),
        radial-gradient(ellipse at 12% 100%, rgba(34, 211, 238, 0.13) 0%, transparent 50%),
        linear-gradient(160deg,
          rgba(46, 22, 88, 0.8) 0%,
          rgba(36, 16, 72, 0.78) 32%,
          rgba(28, 12, 58, 0.86) 66%,
          rgba(24, 10, 50, 0.92) 100%);
      backdrop-filter: blur(44px) saturate(225%);
      -webkit-backdrop-filter: blur(44px) saturate(225%);
      border: 1.5px solid rgba(168, 85, 247, 0.12);
      box-shadow:
        0 26px 72px rgba(0, 0, 0, 0.58),
        0 0 0 1.5px rgba(255, 255, 255, 0.05) inset,
        0 0 100px rgba(168, 85, 247, 0.1),
        0 0 60px rgba(236, 72, 153, 0.06);
      animation: pmPaneIn 0.9s cubic-bezier(0.22, 1, 0.36, 1) 0.38s both;
    }
    .pm-inputs-pane::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 24px;
      padding: 1.5px;
      background: linear-gradient(135deg,
        rgba(236, 72, 153, 0.52) 0%,
        rgba(168, 85, 247, 0.38) 30%,
        rgba(99, 102, 241, 0.42) 58%,
        rgba(59, 130, 246, 0.38) 80%,
        rgba(34, 211, 238, 0.48) 100%);
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.55;
      pointer-events: none;
      z-index: 2;
    }
    .pm-inputs-pane::after {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2.2px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(236, 72, 153, 0.75) 15%,
        rgba(168, 85, 247, 0.7) 40%,
        rgba(99, 102, 241, 0.72) 62%,
        rgba(34, 211, 238, 0.7) 85%,
        transparent 100%
      );
      opacity: 1;
      z-index: 3;
    }
    .pm-inputs-inner {
      position: relative;
      z-index: 4;
      padding: 1.25rem 1.5rem 1.3rem 1.5rem;
    }
    .pm-inputs-head {
      display: flex;
      align-items: center;
      gap: 0.8rem;
      margin-bottom: 0.9rem;
    }
    .pm-inputs-icon {
      width: 40px; height: 40px;
      border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.2rem;
      background: linear-gradient(135deg, rgba(236,72,153,0.3), rgba(99,102,241,0.3));
      border: 1.3px solid rgba(236,72,153,0.35);
      box-shadow: 0 7px 20px rgba(168,85,247,0.32), 0 0 0 1.3px rgba(255,255,255,0.07) inset;
      position: relative;
      flex-shrink: 0;
    }
    .pm-inputs-icon::after {
      content: '';
      position: absolute;
      inset: -1px;
      border-radius: 12px;
      padding: 1.3px;
      background: linear-gradient(135deg,#ec4899,#a855f7,#6366f1);
      -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor; mask-composite: exclude;
      opacity: 0.55;
    }
    .pm-inputs-title {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 800;
      font-size: 1.08rem;
      line-height: 1;
      margin: 0 0 0.18rem 0;
      background: linear-gradient(135deg,#fce7f3,#f9a8d4,#f0abfc,#ddd6fe,#c4b5fd);
      background-size: 240% 240%;
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: gradientShiftVibrant 9s ease infinite;
    }
    .pm-inputs-sub {
      margin: 0;
      font-size: 0.72rem;
      color: rgba(221, 214, 254, 0.85);
      font-weight: 500;
    }
    .pm-inputs-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.55rem 0.8rem;
    }
    .pm-input-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.52rem 0.75rem;
      border-radius: 12px;
      background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.015));
      border: 1px solid rgba(255,255,255,0.05);
      transition: all 0.35s ease;
    }
    .pm-input-row:hover {
      background: linear-gradient(135deg, rgba(255,255,255,0.065), rgba(255,255,255,0.025));
      border-color: rgba(236, 72, 153, 0.18);
      transform: translateY(-1.5px);
    }
    .pm-input-k {
      font-size: 0.68rem;
      color: rgba(226, 232, 240, 0.78);
      font-weight: 500;
      letter-spacing: 0.02em;
      text-transform: capitalize;
    }
    .pm-input-v {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 750;
      font-size: 0.76rem;
      color: #f9a8d4;
      background: linear-gradient(135deg, #fde68a, #f9a8d4, #a5f3fc);
      background-size: 200% 200%;
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: gradientShiftVibrant 8s ease infinite;
      white-space: nowrap;
    }

    .pm-conclusion-pane {
      position: relative;
      border-radius: 26px;
      overflow: hidden;
      margin-top: 1.1rem;
      background:
        radial-gradient(ellipse at 50% 0%, rgba(251, 191, 36, 0.2) 0%, transparent 55%),
        radial-gradient(ellipse at 0% 100%, rgba(236, 72, 153, 0.16) 0%, transparent 50%),
        radial-gradient(ellipse at 100% 100%, rgba(34, 211, 238, 0.16) 0%, transparent 50%),
        linear-gradient(160deg,
          rgba(58, 28, 108, 0.86) 0%,
          rgba(44, 20, 86, 0.82) 30%,
          rgba(36, 16, 72, 0.9) 65%,
          rgba(30, 13, 62, 0.94) 100%);
      backdrop-filter: blur(48px) saturate(240%);
      -webkit-backdrop-filter: blur(48px) saturate(240%);
      border: 1.8px solid rgba(251, 191, 36, 0.14);
      box-shadow:
        0 32px 88px rgba(0, 0, 0, 0.65),
        0 0 0 1.8px rgba(255, 255, 255, 0.055) inset,
        0 0 140px rgba(251, 191, 36, 0.08),
        0 0 90px rgba(236, 72, 153, 0.07);
      animation: pmPaneIn 0.95s cubic-bezier(0.22, 1, 0.36, 1) 0.48s both;
    }
    .pm-conclusion-pane::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 26px;
      padding: 1.8px;
      background: linear-gradient(135deg,
        rgba(251, 191, 36, 0.62) 0%,
        rgba(245, 158, 11, 0.48) 18%,
        rgba(236, 72, 153, 0.4) 38%,
        rgba(168, 85, 247, 0.46) 58%,
        rgba(99, 102, 241, 0.44) 78%,
        rgba(34, 211, 238, 0.56) 100%);
      -webkit-mask:
        linear-gradient(#fff 0 0) content-box,
        linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      opacity: 0.6;
      pointer-events: none;
      z-index: 2;
    }
    .pm-conclusion-pane::after {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 2.6px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(251, 191, 36, 0.88) 12%,
        rgba(245, 158, 11, 0.78) 28%,
        rgba(236, 72, 153, 0.72) 46%,
        rgba(168, 85, 247, 0.78) 62%,
        rgba(99, 102, 241, 0.75) 80%,
        rgba(34, 211, 238, 0.8) 90%,
        transparent 100%
      );
      opacity: 1;
      z-index: 3;
    }
    .pm-conclusion-inner {
      position: relative;
      z-index: 4;
      padding: 1.45rem 1.6rem 1.4rem 1.6rem;
    }
    .pm-conc-head {
      display: flex;
      align-items: center;
      gap: 0.9rem;
      margin-bottom: 1rem;
    }
    .pm-conc-icon-big {
      width: 54px; height: 54px;
      border-radius: 16px;
      display: flex; align-items: center; justify-content: center;
      font-size: 1.65rem;
      background: var(--conc-icon-bg, linear-gradient(135deg, rgba(251,191,36,0.38), rgba(236,72,153,0.32)));
      border: 1.8px solid var(--conc-icon-border, rgba(251,191,36,0.45));
      box-shadow:
        0 12px 32px var(--conc-icon-shadow, rgba(251,191,36,0.32)),
        0 0 0 1.8px rgba(255,255,255,0.08) inset;
      position: relative;
      flex-shrink: 0;
    }
    .pm-conc-icon-big::after {
      content: '';
      position: absolute;
      inset: -1px;
      border-radius: 16px;
      padding: 1.8px;
      background: var(--conc-icon-grad, linear-gradient(135deg,#fbbf24,#ec4899,#a855f7));
      -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
      -webkit-mask-composite: xor; mask-composite: exclude;
      opacity: 0.6;
    }
    .pm-conc-title {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 800;
      font-size: 1.55rem;
      line-height: 1;
      margin: 0 0 0.22rem 0;
      letter-spacing: -0.035em;
      background: var(--conc-title-grad, linear-gradient(135deg,#ffffff,#fef3c7,#fde68a,#f9a8d4,#f0abfc,#a5f3fc));
      background-size: 320% 320%;
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: gradientShiftVibrant 8s ease infinite;
      filter: drop-shadow(0 2px 22px rgba(168,85,247,0.22));
    }
    .pm-conc-sub {
      margin: 0;
      font-size: 0.82rem;
      color: rgba(221, 214, 254, 0.88);
      font-weight: 500;
    }
    .pm-verdict-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 0.95rem 0.5rem 0.65rem;
      border-radius: 999px;
      background: var(--conc-badge-bg, linear-gradient(135deg, rgba(251,191,36,0.22), rgba(236,72,153,0.2)));
      border: 1.3px solid var(--conc-badge-border, rgba(251,191,36,0.4));
      box-shadow: 0 6px 20px var(--conc-badge-shadow, rgba(251,191,36,0.2));
      margin-bottom: 0.85rem;
    }
    .pm-verdict-dot {
      width: 10px; height: 10px;
      border-radius: 999px;
      background: var(--conc-dot-grad, linear-gradient(135deg,#fbbf24,#ec4899));
      box-shadow: 0 0 12px var(--conc-dot-glow, rgba(251,191,36,0.7));
      animation: concPulse 1.8s ease-in-out infinite;
    }
    @keyframes concPulse {
      0%,100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.25); opacity: 0.78; }
    }
    .pm-verdict-text {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 800;
      font-size: 0.9rem;
      letter-spacing: 0.01em;
      color: var(--conc-verdict-text, #fde68a);
      text-transform: uppercase;
    }
    .pm-conc-body {
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 1rem;
    }
    .pm-conc-block {
      padding: 0.85rem 1rem;
      border-radius: 16px;
      background: linear-gradient(135deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
      border: 1px solid rgba(255,255,255,0.06);
      position: relative;
      overflow: hidden;
    }
    .pm-conc-block::before {
      content: '';
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 3.5px;
      background: var(--conc-block-accent, #fbbf24);
      box-shadow: 0 0 14px var(--conc-block-accent, #fbbf24);
      opacity: 0.85;
    }
    .pm-conc-block h4 {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 780;
      font-size: 0.9rem;
      line-height: 1.1;
      margin: 0 0 0.4rem 0;
      padding-left: 0.3rem;
      background: var(--conc-block-title-grad, linear-gradient(135deg,#fef3c7,#fde68a,#f9a8d4));
      background-size: 220% 220%;
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: gradientShiftVibrant 9s ease infinite;
    }
    .pm-conc-block p {
      margin: 0;
      padding-left: 0.3rem;
      font-size: 0.8rem;
      line-height: 1.58;
      color: rgba(226, 232, 240, 0.94);
      font-weight: 450;
    }
    .pm-conc-block p strong {
      color: var(--conc-block-strong, #fde68a);
      font-weight: 720;
    }
    .pm-quick-stats {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.5rem;
      margin-top: 0.4rem;
    }
    .pm-mini-stat {
      padding: 0.45rem 0.65rem;
      border-radius: 10px;
      background: linear-gradient(135deg, rgba(251,191,36,0.1), rgba(236,72,153,0.08));
      border: 1px solid rgba(251,191,36,0.16);
    }
    .pm-mini-k {
      font-size: 0.6rem;
      color: rgba(226, 232, 240, 0.72);
      font-weight: 500;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      margin: 0 0 0.1rem 0;
    }
    .pm-mini-v {
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 800;
      font-size: 0.88rem;
      margin: 0;
      background: linear-gradient(135deg,#fde68a,#f9a8d4,#a5f3fc);
      background-size: 220% 220%;
      -webkit-background-clip: text; background-clip: text;
      -webkit-text-fill-color: transparent;
      animation: gradientShiftVibrant 8s ease infinite;
    }

    .streamlit-expanderHeader {
      background: linear-gradient(160deg,
        rgba(42, 20, 82, 0.88) 0%,
        rgba(30, 14, 64, 0.92) 100%) !important;
      backdrop-filter: blur(20px);
      border: 1.5px solid rgba(251, 191, 36, 0.12) !important;
      border-radius: 17px !important;
      padding: 0.9rem 1.2rem !important;
      transition: all 0.35s cubic-bezier(0.22, 1, 0.36, 1) !important;
      color: #e0e7ff !important;
      font-weight: 700 !important;
      font-size: 0.9rem !important;
      margin-bottom: 0.8rem !important;
      box-shadow:
        0 8px 24px rgba(0,0,0,0.42),
        0 0 0 1.5px rgba(255,255,255,0.05) inset !important;
      font-family: 'Space Grotesk', sans-serif !important;
      position: relative;
      overflow: hidden;
    }
    .streamlit-expanderHeader::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1.8px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(251, 191, 36, 0.55) 25%,
        rgba(236, 72, 153, 0.45) 50%,
        rgba(129, 140, 248, 0.55) 75%,
        transparent 100%
      );
    }
    .streamlit-expanderHeader:hover {
      border-color: rgba(251, 191, 36, 0.3) !important;
      box-shadow:
        0 18px 42px rgba(0,0,0,0.54),
        0 0 42px rgba(99, 102, 241, 0.22),
        0 0 0 1.5px rgba(255,255,255,0.06) inset !important;
      transform: translateY(-3px);
    }
    [data-testid="stExpander"] { border: none !important; }
    [data-testid="stExpanderDetails"] {
      background: linear-gradient(180deg, rgba(22, 10, 48, 0.82) 0%, rgba(18, 8, 40, 0.88) 100%) !important;
      backdrop-filter: blur(18px);
      border: 1.5px solid rgba(251, 191, 36, 0.1) !important;
      border-top: none !important;
      border-radius: 0 0 17px 17px !important;
      padding: 1.15rem !important;
      margin: -0.8rem 0 0.9rem 0 !important;
      color: #cbd5e1 !important;
      line-height: 1.7;
      font-size: 0.89rem;
      box-shadow: 0 10px 24px rgba(0,0,0,0.35);
    }
    [data-testid="stExpanderDetails"] strong {
      background: linear-gradient(135deg, #fde68a, #f9a8d4, #f0abfc, #a5f3fc);
      background-size: 220% 220%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800 !important;
      animation: gradientShiftVibrant 8s ease infinite;
    }
    [data-testid="stExpanderDetails"] li { color: #cbd5e1 !important; margin-bottom: 0.4rem; }

    [data-testid="stAlertContainer"] { margin-bottom: 1.2rem !important; }
    [data-testid="stAlert"] {
      background: linear-gradient(160deg,
        rgba(40, 14, 30, 0.96) 0%,
        rgba(30, 10, 22, 0.98) 100%) !important;
      backdrop-filter: blur(20px) !important;
      border: 1.5px solid rgba(248, 113, 113, 0.5) !important;
      border-left: 6px solid #f43f5e !important;
      border-radius: 18px !important;
      box-shadow:
        0 16px 42px rgba(0, 0, 0, 0.48),
        0 0 50px rgba(244, 63, 94, 0.16) !important;
      color: #fecaca !important;
      font-weight: 500 !important;
      padding: 1.2rem 1.35rem !important;
      position: relative;
      overflow: hidden;
    }
    [data-testid="stAlert"]::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1.5px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(248, 113, 113, 0.6) 30%,
        rgba(251, 191, 36, 0.4) 60%,
        transparent 100%
      );
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
      color: #fecaca !important;
      font-size: 0.94rem;
      line-height: 1.75;
    }
    [data-testid="stAlert"][kind="warning"] {
      background: linear-gradient(160deg,
        rgba(48, 30, 10, 0.96) 0%,
        rgba(36, 22, 8, 0.98) 100%) !important;
      border: 1.5px solid rgba(251, 191, 36, 0.55) !important;
      border-left: 6px solid #f59e0b !important;
      color: #fde68a !important;
      box-shadow:
        0 16px 42px rgba(0, 0, 0, 0.48),
        0 0 50px rgba(245, 158, 11, 0.16) !important;
    }
    [data-testid="stAlert"][kind="warning"]::before {
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(251, 191, 36, 0.65) 30%,
        rgba(251, 113, 133, 0.4) 60%,
        transparent 100%
      );
    }
    [data-testid="stAlert"][kind="warning"] [data-testid="stMarkdownContainer"] { color: #fde68a !important; }

    [data-testid="stDataFrame"] {
      background: linear-gradient(160deg,
        rgba(30, 14, 64, 0.8) 0%,
        rgba(22, 10, 48, 0.86) 100%) !important;
      border-radius: 20px !important;
      border: 1.5px solid rgba(251, 191, 36, 0.13) !important;
      overflow: hidden !important;
      box-shadow:
        0 16px 42px rgba(0,0,0,0.5),
        0 0 0 1.5px rgba(255,255,255,0.05) inset !important;
    }

    .foot-card {
      margin-top: 1.6rem;
      padding: 1.05rem 1.8rem;
      text-align: center;
      border-radius: 18px;
      background: linear-gradient(160deg,
        rgba(30, 14, 64, 0.82) 0%,
        rgba(20, 10, 44, 0.88) 100%);
      backdrop-filter: blur(20px);
      border: 1.5px solid rgba(251, 191, 36, 0.09);
      color: rgba(221, 214, 254, 0.94);
      font-size: 0.82rem;
      line-height: 1.65;
      box-shadow:
        0 0 0 1.5px rgba(255,255,255,0.035) inset,
        0 10px 28px rgba(0,0,0,0.36);
      position: relative;
      overflow: hidden;
    }
    .foot-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1.8px;
      background: linear-gradient(90deg,
        transparent 0%,
        rgba(251, 191, 36, 0.55) 25%,
        rgba(236, 72, 153, 0.45) 50%,
        rgba(192, 132, 252, 0.5) 72%,
        transparent 100%
      );
    }
    .foot-card strong {
      background: linear-gradient(135deg, #fde68a, #f9a8d4, #f0abfc, #a5f3fc, #fbbf24);
      background-size: 260% 260%;
      -webkit-background-clip: text;
      background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800;
      animation: gradientShiftVibrant 8s ease infinite;
    }

    ::-webkit-scrollbar { width: 12px; height: 12px; }
    ::-webkit-scrollbar-track { background: rgba(10, 5, 20, 0.75); }
    ::-webkit-scrollbar-thumb {
      background: linear-gradient(180deg,
        #fbbf24 0%,
        #f97316 18%,
        #fb7185 36%,
        #ec4899 52%,
        #a855f7 70%,
        #6366f1 100%
      );
      border-radius: 7px;
      border: 3px solid transparent;
      background-clip: padding-box;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: linear-gradient(180deg,
        #fde047 0%,
        #fb923c 18%,
        #f472b6 36%,
        #f472b6 52%,
        #c084fc 70%,
        #818cf8 100%
      );
      border: 3px solid transparent;
      background-clip: padding-box;
    }

    @media (max-width: 1200px) {
      .info-strip { grid-template-columns: repeat(2, 1fr); }
    }
    @media (max-width: 1024px) {
      [data-testid="stAppViewContainer"] > section[data-testid="stMain"] {
        padding: 1.3rem 1.5rem 2.2rem 1.5rem !important;
      }
      .hero-big { font-size: 2.2rem; }
      .result-figure { font-size: 3.6rem !important; }
      .hero-banner { padding: 1.9rem 1.7rem 2rem 1.7rem; border-radius: 24px; }
      .input-panel { padding: 1.6rem 1.4rem 1.7rem 1.4rem; border-radius: 22px; }
    }
    @media (max-width: 768px) {
      .info-strip { grid-template-columns: 1fr; }
      .stats-triplet { grid-template-columns: 1fr; }
      .hero-big { font-size: 1.8rem; }
      .result-figure { font-size: 2.8rem !important; }
      .hero-banner { padding: 1.5rem 1.1rem 1.6rem 1.1rem; border-radius: 20px; }
      .result-hero { padding: 2rem 1.1rem 1.7rem 1.1rem; border-radius: 22px; }
      .input-panel { padding: 1.3rem 1rem 1.4rem 1rem; border-radius: 18px; }
      .brand-title-main { font-size: 1.4rem; }
      .brand-logo-wrap { width: 50px; height: 50px; border-radius: 16px; }
      .brand-logo { font-size: 1.7rem; }
      [data-testid="stAppViewContainer"] > section[data-testid="stMain"] {
        padding: 1.1rem 0.9rem 2rem 0.9rem !important;
      }
      .pm-grid { grid-template-columns: 1fr 1fr !important; }
    }
    @media (max-width: 540px) {
      .pm-grid { grid-template-columns: 1fr !important; }
    }
    </style>

    <div class="floating-orb orb-1"></div>
    <div class="floating-orb orb-2"></div>
    <div class="floating-orb orb-3"></div>
    <div class="floating-orb orb-4"></div>
    <div class="floating-orb orb-5"></div>
"""), unsafe_allow_html=True)

st.markdown(flat_html("""
    <div class="app-hero-wrap">
      <div class="hero-banner">
        <div class="hero-border-top"></div>
        <div class="hero-inner-glow"></div>
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
              <span>📊 AI &amp; Python Programming</span>
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
        <div class="info-hover-glow"></div>
        <div class="info-icon-wrap">🧪</div>
        <div class="info-body">
          <div class="info-label">Model Architecture</div>
          <div class="info-value">Linear Regression Pipeline</div>
        </div>
      </div>
      <div class="info-card">
        <div class="info-hover-glow"></div>
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
        result_glow = "rgba(16, 185, 129, 0.28)" if is_profit else "rgba(239, 68, 68, 0.28)"
        dot_gradient = '#10b981, #22d3ee, #fbbf24' if is_profit else '#ef4444, #fb7185, #fbbf24'

        result_html = flat_html(f"""
            <div class="result-hero" style="--result-glow: {result_glow};">
              <div class="result-glow"></div>
              <div class="result-tag {tag_cls}">
                <span class="hero-dot" style="background: linear-gradient(135deg, {dot_gradient});"></span>
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

        margin_pct = (unit_margin / unit_selling_price) * 100 if unit_selling_price > 0 else 0
        break_even_units = upfront / unit_margin if unit_margin > 0 else float('inf')
        break_even_pct = (break_even_units / first_year_units_sold) * 100 if first_year_units_sold > 0 else 0
        roi_pct = (prediction / upfront) * 100 if upfront > 0 else 0
        dev_pct = (development_cost / upfront) * 100 if upfront > 0 else 0
        mkt_pct = (launch_marketing_spend / upfront) * 100 if upfront > 0 else 0
        payback_months = (upfront / (margin_x_units / 12)) if margin_x_units > 0 else float('inf')
        payback_text = f" 💰 You get all your pocket money back in ~{payback_months:.1f} months" if payback_months != float('inf') and payback_months > 0 else ""

        pricing_tone = "healthy" if margin_pct >= 40 else ("solid" if margin_pct >= 25 else "tight")
        pricing_icon = "💎" if margin_pct >= 40 else ("💰" if margin_pct >= 25 else "⚖️")
        roi_icon = "📈" if roi_pct >= 50 else ("✅" if roi_pct >= 15 else ("⚠️" if roi_pct >= 0 else "🚨"))
        roi_tone = "exceptional" if roi_pct >= 100 else ("strong" if roi_pct >= 50 else ("positive" if roi_pct >= 15 else ("marginal" if roi_pct >= 0 else "negative")))
        gtm_icon = "🎯" if mkt_pct >= 40 else ("🛠️" if dev_pct >= 60 else "⚡")
        vol_icon = "📦" if break_even_pct <= 50 else ("🎯" if break_even_pct <= 80 else "🔴")
        roi_article = "an" if roi_tone[0].lower() in "aeiou" else "a"

        step1_simple = (
            f"Imagine you sell lemonade for <strong>${unit_selling_price:.2f}</strong> a cup. "
            f"It costs you <strong>${unit_production_cost:,.2f}</strong> to make each cup. "
            f"So every time you sell one, you put <strong>${unit_margin:,.2f}</strong> "
            f"(that's <strong>{margin_pct:.1f}%</strong>) in your piggy bank. "
            f"That's a <strong>{pricing_tone}</strong> deal! "
            f"If people love your lemonade, charging a tiny bit more (like <strong>5–10%</strong>) means you fill your piggy bank much faster."
        )
        gtm_simple = (
            f"You're taking <strong>${upfront:,.0f}</strong> from your big savings jar to start this. "
            f"<strong>{dev_pct:.0f}%</strong> goes to making the product great (building the lemonade stand, squeezing the lemons), "
            f"and <strong>{mkt_pct:.0f}%</strong> goes to telling people about it (making signs, yelling 'fresh lemonade!'). "
            + (f"Since you're spending a lot on signs, first make sure people actually stop to buy before you make all the signs." if mkt_pct > dev_pct else f"Since you're spending a lot on building, make sure you know exactly what your stand needs — don't build a fancy slide if people just want lemonade!")
        )
        volume_simple = (
            f"You need to sell <strong>{break_even_units:,.0f}</strong> cups of lemonade before your piggy bank has all the money back that you took from savings. "
            f"That's <strong>{break_even_pct:.1f}%</strong> of all the cups you hope to sell in the first year. "
            + (f"Great news — that's not very many! Even if fewer people come than you hope, you'll still get your money back." if break_even_pct <= 50 else
              (f"That's a reasonable number. Watch the first few months closely — if lots of people come right away, you're golden." if break_even_pct <= 80 else
               f"Whoa, that's a lot of cups! You might need to go to more streets (that's <strong>{distribution_coverage * 100:.0f}%</strong> distribution coverage) or make your lemonade extra special so more people buy."))
        )
        roi_simple = (
            f"After one year, you'll have <strong>{roi_pct:+.1f}%</strong> more money in your piggy bank than what you started with — "
            f"that's <strong>{roi_tone}</strong>!{payback_text}. "
            + (f"🎉 DO IT! You'll make way more money than you spend. Start building the stand and keep checking your piggy bank every 3 months." if roi_pct >= 30 else
              (f"🤔 Maybe do it! First double-check your math — what if 15% fewer people come? Or what if they love it and 15% more come? Then decide." if roi_pct >= 0 else
               f"🚫 Wait! Don't do it yet. Try making the cups cheaper to make, charging a bit more per cup, or building a smaller stand first. Then try the math again."))
        )

        pm_insights_html = flat_html(f"""
            <div class="pm-wrap">
              <div class="pm-head">
                <div class="pm-bar"></div>
                <div class="pm-head-text">
                  <h2>What This Means for Product Managers</h2>
                  <p>Your launch forecast, explained in 4 simple steps · no jargon, promise</p>
                </div>
              </div>
              <div class="pm-single-pane">
                <div class="pm-pane-inner">
                  <div class="pm-steps">

                    <div class="pm-step" style="--step-accent:#fbbf24; --step-bg:linear-gradient(135deg,rgba(251,191,36,0.32),rgba(251,113,133,0.28)); --step-border:rgba(251,191,36,0.38); --step-shadow:rgba(251,191,36,0.22); --step-border-grad:linear-gradient(135deg,#fbbf24,#ec4899); --step-pill-bg:rgba(251,191,36,0.18); --step-pill-text:#fde68a; --step-pill-border:rgba(251,191,36,0.32); --step-title-grad:linear-gradient(135deg,#fef3c7,#fde68a,#f9a8d4); --step-strong:#fde68a;">
                      <div class="pm-step-icon">{pricing_icon}</div>
                      <div class="pm-step-body">
                        <div class="pm-step-label">
                          <span class="pm-step-num">STEP 1</span>
                          <h3 class="pm-step-title">How much you earn on every sale</h3>
                        </div>
                        <p class="pm-step-text">{step1_simple}</p>
                      </div>
                    </div>

                    <div class="pm-step" style="--step-accent:#ec4899; --step-bg:linear-gradient(135deg,rgba(236,72,153,0.3),rgba(168,85,247,0.28)); --step-border:rgba(236,72,153,0.36); --step-shadow:rgba(236,72,153,0.2); --step-border-grad:linear-gradient(135deg,#ec4899,#a855f7); --step-pill-bg:rgba(236,72,153,0.17); --step-pill-text:#f9a8d4; --step-pill-border:rgba(236,72,153,0.3); --step-title-grad:linear-gradient(135deg,#fce7f3,#f9a8d4,#f0abfc); --step-strong:#f9a8d4;">
                      <div class="pm-step-icon">{gtm_icon}</div>
                      <div class="pm-step-body">
                        <div class="pm-step-label">
                          <span class="pm-step-num">STEP 2</span>
                          <h3 class="pm-step-title">Where your startup money goes</h3>
                        </div>
                        <p class="pm-step-text">{gtm_simple}</p>
                      </div>
                    </div>

                    <div class="pm-step" style="--step-accent:#a855f7; --step-bg:linear-gradient(135deg,rgba(168,85,247,0.3),rgba(99,102,241,0.28)); --step-border:rgba(168,85,247,0.36); --step-shadow:rgba(168,85,247,0.22); --step-border-grad:linear-gradient(135deg,#a855f7,#6366f1); --step-pill-bg:rgba(168,85,247,0.17); --step-pill-text:#ddd6fe; --step-pill-border:rgba(168,85,247,0.3); --step-title-grad:linear-gradient(135deg,#ede9fe,#ddd6fe,#c4b5fd); --step-strong:#ddd6fe;">
                      <div class="pm-step-icon">{vol_icon}</div>
                      <div class="pm-step-body">
                        <div class="pm-step-label">
                          <span class="pm-step-num">STEP 3</span>
                          <h3 class="pm-step-title">How many sales until you win</h3>
                        </div>
                        <p class="pm-step-text">{volume_simple}</p>
                      </div>
                    </div>

                    <div class="pm-step" style="--step-accent:#22d3ee; --step-bg:linear-gradient(135deg,rgba(34,211,238,0.3),rgba(59,130,246,0.28)); --step-border:rgba(34,211,238,0.36); --step-shadow:rgba(34,211,238,0.22); --step-border-grad:linear-gradient(135deg,#22d3ee,#3b82f6); --step-pill-bg:rgba(34,211,238,0.17); --step-pill-text:#a5f3fc; --step-pill-border:rgba(34,211,238,0.3); --step-title-grad:linear-gradient(135deg,#cffafe,#a5f3fc,#7dd3fc); --step-strong:#a5f3fc;">
                      <div class="pm-step-icon">{roi_icon}</div>
                      <div class="pm-step-body">
                        <div class="pm-step-label">
                          <span class="pm-step-num">STEP 4</span>
                          <h3 class="pm-step-title">The big question: Should you do it?</h3>
                        </div>
                        <p class="pm-step-text">{roi_simple}</p>
                      </div>
                    </div>

                  </div>
                </div>
              </div>
            </div>
        """)
        st.markdown(pm_insights_html, unsafe_allow_html=True)

        inputs_html = flat_html(f"""
            <div class="pm-section-gap"></div>
            <div class="pm-inputs-pane">
              <div class="pm-inputs-inner">
                <div class="pm-inputs-head">
                  <div class="pm-inputs-icon">🔍</div>
                  <div>
                    <h3 class="pm-inputs-title">Exact Inputs Sent to the Model</h3>
                    <p class="pm-inputs-sub">8 features the AI used to calculate your forecast</p>
                  </div>
                </div>
                <div class="pm-inputs-grid">
                  <div class="pm-input-row">
                    <span class="pm-input-k">Development Cost</span>
                    <span class="pm-input-v">${development_cost:,.0f}</span>
                  </div>
                  <div class="pm-input-row">
                    <span class="pm-input-k">Launch Marketing</span>
                    <span class="pm-input-v">${launch_marketing_spend:,.0f}</span>
                  </div>
                  <div class="pm-input-row">
                    <span class="pm-input-k">Unit Margin</span>
                    <span class="pm-input-v">${unit_margin:,.2f}</span>
                  </div>
                  <div class="pm-input-row">
                    <span class="pm-input-k">Margin × Units</span>
                    <span class="pm-input-v">${margin_x_units:,.0f}</span>
                  </div>
                  <div class="pm-input-row">
                    <span class="pm-input-k">Distribution Coverage</span>
                    <span class="pm-input-v">{distribution_coverage * 100:.0f}%</span>
                  </div>
                  <div class="pm-input-row">
                    <span class="pm-input-k">Customer Rating</span>
                    <span class="pm-input-v">{customer_rating:.1f} / 5.0</span>
                  </div>
                  <div class="pm-input-row">
                    <span class="pm-input-k">Product Category</span>
                    <span class="pm-input-v">{product_category}</span>
                  </div>
                  <div class="pm-input-row">
                    <span class="pm-input-k">Launch Region</span>
                    <span class="pm-input-v">{launch_region}</span>
                  </div>
                  <div class="pm-input-row">
                    <span class="pm-input-k">Upfront Total</span>
                    <span class="pm-input-v">${upfront:,.0f}</span>
                  </div>
                </div>
              </div>
            </div>
        """)
        st.markdown(inputs_html, unsafe_allow_html=True)

        if roi_pct >= 50:
            conc_verdict = "🚀 GO FOR LAUNCH"
            conc_verdict_text = "Green Light · Exceptional Profile"
            conc_icon = "🚀"
            conc_style = """
              --conc-icon-bg:linear-gradient(135deg,rgba(34,211,238,0.38),rgba(16,185,129,0.32));
              --conc-icon-border:rgba(34,211,238,0.45);
              --conc-icon-shadow:rgba(34,211,238,0.34);
              --conc-icon-grad:linear-gradient(135deg,#22d3ee,#10b981,#a855f7);
              --conc-title-grad:linear-gradient(135deg,#ffffff,#cffafe,#a5f3fc,#6ee7b7,#bbf7d0,#ddd6fe);
              --conc-badge-bg:linear-gradient(135deg,rgba(16,185,129,0.24),rgba(34,211,238,0.22));
              --conc-badge-border:rgba(16,185,129,0.45);
              --conc-badge-shadow:rgba(16,185,129,0.25);
              --conc-dot-grad:linear-gradient(135deg,#10b981,#22d3ee);
              --conc-dot-glow:rgba(16,185,129,0.75);
              --conc-verdict-text:#6ee7b7;
              --conc-block-accent:#10b981;
              --conc-block-title-grad:linear-gradient(135deg,#d1fae5,#6ee7b7,#a5f3fc);
              --conc-block-strong:#6ee7b7;
            """
            conc_summary = f"The math is overwhelmingly positive — a <strong>{roi_pct:+.0f}%</strong> Year-1 return with payback in ~{payback_months:.0f} months sits well above the cost of capital. Gross margin at <strong>{margin_pct:.0f}%</strong> means each sale compounds quickly, and the break-even volume of <strong>{break_even_units:,.0f}</strong> units is comfortably within reach."
            conc_rec = f"<strong>Move to launch readiness immediately.</strong> Lock MVP scope to hit the <strong>{dev_pct:.0f}%</strong> R&D budget, then phase the <strong>{mkt_pct:.0f}%</strong> marketing spend in waves after the first cohort validates demand. Set up quarterly tracking against the <strong>{first_year_units_sold:,.0f}</strong> unit forecast and revisit pricing if distribution exceeds <strong>{distribution_coverage * 100:.0f}%</strong>."
        elif roi_pct >= 15:
            conc_verdict = "✅ CONDITIONAL GO"
            conc_verdict_text = "Strong Profile · Sensitivity Check"
            conc_icon = "✅"
            conc_style = """
              --conc-icon-bg:linear-gradient(135deg,rgba(168,85,247,0.38),rgba(99,102,241,0.32));
              --conc-icon-border:rgba(168,85,247,0.45);
              --conc-icon-shadow:rgba(168,85,247,0.34);
              --conc-icon-grad:linear-gradient(135deg,#a855f7,#6366f1,#22d3ee);
              --conc-title-grad:linear-gradient(135deg,#ffffff,#ede9fe,#ddd6fe,#c4b5fd,#a5f3fc);
              --conc-badge-bg:linear-gradient(135deg,rgba(168,85,247,0.24),rgba(99,102,241,0.22));
              --conc-badge-border:rgba(168,85,247,0.45);
              --conc-badge-shadow:rgba(168,85,247,0.25);
              --conc-dot-grad:linear-gradient(135deg,#a855f7,#6366f1);
              --conc-dot-glow:rgba(168,85,247,0.75);
              --conc-verdict-text:#ddd6fe;
              --conc-block-accent:#a855f7;
              --conc-block-title-grad:linear-gradient(135deg,#ede9fe,#ddd6fe,#c4b5fd);
              --conc-block-strong:#ddd6fe;
            """
            conc_summary = f"Returns are solid at <strong>{roi_pct:+.0f}%</strong> ROI with payback in ~{payback_months:.0f} months, but there's not enough buffer to be careless. Break-even at <strong>{break_even_pct:.0f}%</strong> of Year-1 volume leaves room, yet a material miss on demand or a cost overrun could flip the result."
            conc_rec = f"<strong>Approve with two quick guardrails:</strong> First, run sensitivity on <strong>±15% volume</strong> and <strong>±0.3 rating</strong> to confirm downside stays above water. Second, hold <strong>20–25%</strong> of marketing spend in reserve until the first 60 days of real sales data lands. Keep the <strong>{margin_pct:.0f}%</strong> gross margin target locked — don't dilute it for distribution."
        elif roi_pct >= 0:
            conc_verdict = "⚠️ HOLD — ITERATE FIRST"
            conc_verdict_text = "Marginal · Needs Optimization"
            conc_icon = "⚠️"
            conc_style = """
              --conc-icon-bg:linear-gradient(135deg,rgba(251,191,36,0.38),rgba(245,158,11,0.32));
              --conc-icon-border:rgba(251,191,36,0.45);
              --conc-icon-shadow:rgba(251,191,36,0.34);
              --conc-icon-grad:linear-gradient(135deg,#fbbf24,#f59e0b,#ec4899);
              --conc-title-grad:linear-gradient(135deg,#ffffff,#fef3c7,#fde68a,#fcd34d,#f9a8d4);
              --conc-badge-bg:linear-gradient(135deg,rgba(251,191,36,0.25),rgba(245,158,11,0.22));
              --conc-badge-border:rgba(251,191,36,0.45);
              --conc-badge-shadow:rgba(251,191,36,0.26);
              --conc-dot-grad:linear-gradient(135deg,#fbbf24,#f59e0b);
              --conc-dot-glow:rgba(251,191,36,0.75);
              --conc-verdict-text:#fde68a;
              --conc-block-accent:#fbbf24;
              --conc-block-title-grad:linear-gradient(135deg,#fef3c7,#fde68a,#fcd34d);
              --conc-block-strong:#fde68a;
            """
            conc_summary = f"Returns are positive at <strong>{roi_pct:+.0f}%</strong> but thin — barely covering the cost of capital. Break-even at <strong>{break_even_pct:.0f}%</strong> of forecast and a <strong>{margin_pct:.0f}%</strong> margin mean any slip on units or rating pushes you underwater."
            conc_rec = f"<strong>Do not sign off yet — rework the plan on at least two of these levers:</strong> (1) Trim unit production cost to push margin up <strong>5–8 pts</strong>. (2) Reposition for a <strong>5–10% higher price</strong> if category allows. (3) Narrow launch scope — cut <strong>15–20%</strong> of upfront spend (either delay features or reduce marketing). Then re-run the model."
        else:
            conc_verdict = "🚫 STOP — DO NOT LAUNCH"
            conc_verdict_text = "Loss Projected · Rework Required"
            conc_icon = "🚫"
            conc_style = """
              --conc-icon-bg:linear-gradient(135deg,rgba(239,68,68,0.38),rgba(236,72,153,0.32));
              --conc-icon-border:rgba(239,68,68,0.45);
              --conc-icon-shadow:rgba(239,68,68,0.34);
              --conc-icon-grad:linear-gradient(135deg,#ef4444,#ec4899,#f59e0b);
              --conc-title-grad:linear-gradient(135deg,#ffffff,#fee2e2,#fecaca,#f9a8d4,#fde68a);
              --conc-badge-bg:linear-gradient(135deg,rgba(239,68,68,0.25),rgba(236,72,153,0.22));
              --conc-badge-border:rgba(239,68,68,0.45);
              --conc-badge-shadow:rgba(239,68,68,0.26);
              --conc-dot-grad:linear-gradient(135deg,#ef4444,#ec4899);
              --conc-dot-glow:rgba(239,68,68,0.75);
              --conc-verdict-text:#fecaca;
              --conc-block-accent:#ef4444;
              --conc-block-title-grad:linear-gradient(135deg,#fee2e2,#fecaca,#f9a8d4);
              --conc-block-strong:#fecaca;
            """
            conc_summary = f"The model predicts a <strong>{roi_pct:+.0f}%</strong> first-year loss — spending more than you'll earn back. At <strong>{margin_pct:.0f}%</strong> margin and break-even at <strong>{break_even_pct:.0f}%</strong> of the forecast, the economics simply don't close under realistic demand."
            conc_rec = f"<strong>Hard stop — go back and rebuild at least three levers at once:</strong> <strong>(a)</strong> Can production cost drop enough to push gross margin above <strong>25%</strong>? <strong>(b)</strong> Can price rise <strong>10–15%</strong> without killing units? <strong>(c)</strong> Can upfront spend drop by <strong>25%+</strong> with a narrower scope? <strong>(d)</strong> Is there a higher-confidence region or category with better rating support? Only re-submit after the model crosses into positive ROI territory."

        payback_display = f"~{payback_months:.0f} mo" if payback_months != float('inf') and payback_months > 0 else "N/A"
        conclusion_html = flat_html(f"""
            <div class="pm-conclusion-pane" style="{conc_style}">
              <div class="pm-conclusion-inner">
                <div class="pm-conc-head">
                  <div class="pm-conc-icon-big">{conc_icon}</div>
                  <div>
                    <h2 class="pm-conc-title">Final Verdict &amp; Recommendation</h2>
                    <p class="pm-conc-sub">Based on all 8 model inputs · machine + human synthesis</p>
                  </div>
                </div>
                <div class="pm-verdict-badge">
                  <span class="pm-verdict-dot"></span>
                  <span class="pm-verdict-text">{conc_verdict} · {conc_verdict_text}</span>
                </div>
                <div class="pm-conc-body">
                  <div class="pm-conc-block">
                    <h4>🧠 What the numbers say</h4>
                    <p>{conc_summary}</p>
                    <div class="pm-quick-stats">
                      <div class="pm-mini-stat">
                        <p class="pm-mini-k">1st-Year ROI</p>
                        <p class="pm-mini-v">{roi_pct:+.1f}%</p>
                      </div>
                      <div class="pm-mini-stat">
                        <p class="pm-mini-k">Payback</p>
                        <p class="pm-mini-v">{payback_display}</p>
                      </div>
                      <div class="pm-mini-stat">
                        <p class="pm-mini-k">Gross Margin</p>
                        <p class="pm-mini-v">{margin_pct:.1f}%</p>
                      </div>
                      <div class="pm-mini-stat">
                        <p class="pm-mini-k">Break-Even</p>
                        <p class="pm-mini-v">{break_even_units:,.0f}u</p>
                      </div>
                    </div>
                  </div>
                  <div class="pm-conc-block" style="--conc-block-accent:#ec4899; --conc-block-title-grad:linear-gradient(135deg,#fce7f3,#f9a8d4,#f0abfc); --conc-block-strong:#f9a8d4;">
                    <h4>🎯 Recommended next steps</h4>
                    <p>{conc_rec}</p>
                  </div>
                </div>
              </div>
            </div>
        """)
        st.markdown(conclusion_html, unsafe_allow_html=True)

st.markdown(flat_html("""
    <div class="foot-card">
      <strong>Model:</strong> Linear Regression pipeline (StandardScaler + OneHotEncoder) trained on
      historical New Product launches · AI Spreadsheets &amp; Python Programming
    </div>
"""), unsafe_allow_html=True)
