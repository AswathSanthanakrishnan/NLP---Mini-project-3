
import os
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch

# --- Configuration ---
st.set_page_config(page_title="Tasker.ai", layout="wide")
LOGO_PATH = os.path.join("assets", "tasker_logo.png")

# Initialize model cache in session state
if "text_generator" not in st.session_state:
    st.session_state["text_generator"] = None
    st.session_state["model_loaded"] = False


def get_palette(theme):
    """Return color palette for light/dark themes."""
    if theme == "dark":
        return {
            "bg": "#0c1021",
            "grad1": "#1b1035",
            "grad2": "#0f172a",
            "card": "rgba(17, 24, 39, 0.92)",
            "glass": "rgba(17, 24, 39, 0.9)",
            "border": "rgba(148, 163, 184, 0.18)",
            "accent": "#7c3aed",
            "accent2": "#22d3ee",
            "text": "#e5e7eb",
            "muted": "#94a3b8",
            "sidebar_from": "#0f172a",
            "sidebar_to": "#111827",
            "chip_bg": "rgba(124, 58, 237, 0.16)",
            "pill_bg": "rgba(34, 211, 238, 0.14)",
            "pill_span": "#0ea5e9",
            "badge_bg": "rgba(124, 58, 237, 0.2)",
            "badge_text": "#e0e7ff",
            "landing_bg": "linear-gradient(135deg, #111827 0%, #0b1220 40%, #111827 100%)",
            "glow1": "rgba(124, 58, 237, 0.32)",
            "glow2": "rgba(34, 211, 238, 0.26)",
            "button_shadow": "rgba(124, 58, 237, 0.28)",
            "button_hover_shadow": "rgba(34, 211, 238, 0.35)",
            "input_bg": "#0f172a",
            "input_border": "rgba(226, 232, 240, 0.18)",
            "placeholder": "rgba(148, 163, 184, 0.9)",
        }
    return {
        "bg": "#f8f6ff",
        "grad1": "#ffe3ec",
        "grad2": "#e7d6ff",
        "card": "#ffffff",
        "glass": "rgba(255, 255, 255, 0.82)",
        "border": "rgba(15, 23, 42, 0.08)",
        "accent": "#e11d48",
        "accent2": "#5b21b6",
        "text": "#111827",
        "muted": "#4b5563",
        "sidebar_from": "#1f2937",
        "sidebar_to": "#312e81",
        "chip_bg": "rgba(225, 29, 72, 0.12)",
        "pill_bg": "rgba(91, 33, 182, 0.14)",
        "pill_span": "#1f2937",
        "badge_bg": "rgba(91, 33, 182, 0.14)",
        "badge_text": "#312e81",
        "landing_bg": "linear-gradient(135deg, #fef2f2 0%, #ffffff 35%, #f3e8ff 100%)",
        "glow1": "rgba(225, 29, 72, 0.28)",
        "glow2": "rgba(91, 33, 182, 0.24)",
        "button_shadow": "rgba(91, 33, 182, 0.3)",
        "button_hover_shadow": "rgba(225, 29, 72, 0.38)",
        "input_bg": "#ffffff",
        "input_border": "rgba(15, 23, 42, 0.08)",
        "placeholder": "#94a3b8",
    }


def inject_global_styles(palette):
    """Apply a brighter, modern UI theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

        :root {
            --bg: #f8f6ff;
            --card: #ffffff;
            --glass: rgba(255, 255, 255, 0.82);
            --border: rgba(15, 23, 42, 0.08);
            --shadow: 0 20px 50px rgba(15, 23, 42, 0.12);
            --accent: #e11d48;      /* ruby */
            --accent-2: #5b21b6;    /* violet */
            --text: #111827;
            --muted: #4b5563;
        }

        .stApp {
            background:
                radial-gradient(120% 120% at 15% 15%, #ffe3ec 0%, transparent 35%),
                radial-gradient(120% 120% at 85% 10%, #e7d6ff 0%, transparent 32%),
                var(--bg);
            color: var(--text);
        }

        .block-container {
            padding: 1.5rem 2.5rem 3rem;
        }

        h1, h2, h3, h4, h5 {
            font-family: 'Space Grotesk', 'Manrope', sans-serif;
            letter-spacing: -0.4px;
        }

        .stMarkdown, p, label, .stTextInput, .stTextArea, .stSelectbox, .stCaption, .stRadio {
            font-family: 'Manrope', sans-serif;
            color: var(--text);
        }

        code {
            background: rgba(15, 23, 42, 0.05);
            padding: 2px 6px;
            border-radius: 6px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1f2937 0%, #312e81 100%);
            color: #f3f4f6;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: #f3f4f6 !important;
        }
        [data-testid="stSidebar"] .stTextInput input, [data-testid="stSidebar"] .stFileUploader {
            background: rgba(255, 255, 255, 0.08);
        }

        /* Cards */
        .glass-card {
            background: var(--glass);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
        }
        .hero-card {
            margin-bottom: 18px;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 4px;
        }
        .chip-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        .chip {
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(225, 29, 72, 0.12);
            color: var(--text);
            font-weight: 600;
            font-size: 13px;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(91, 33, 182, 0.14);
            color: var(--text);
            font-weight: 600;
            letter-spacing: 0.1px;
        }
        .pill span {
            background: #1f2937;
            color: #fff;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 12px;
        }
        .badge {
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(91, 33, 182, 0.14);
            color: #312e81;
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.2px;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            padding: 14px 18px;
            font-weight: 600;
            color: var(--muted);
            border-radius: 14px 14px 0 0;
            border: 1px solid transparent;
            background: rgba(255, 255, 255, 0.7);
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent) !important;
            color: #fff !important;
            box-shadow: 0 12px 30px rgba(225, 29, 72, 0.35);
        }

        /* Buttons */
        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            border: none;
            background: linear-gradient(120deg, #e11d48, #5b21b6);
            color: #fff;
            font-weight: 700;
            box-shadow: 0 12px 30px rgba(91, 33, 182, 0.3);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 40px rgba(225, 29, 72, 0.38);
        }
        div.stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: #fff;
            color: var(--text);
            font-weight: 700;
        }

        /* Inputs */
        .stTextInput input, .stTextArea textarea, .stSelectbox select {
            background: #fff;
            border-radius: 12px;
            border: 1.5px solid rgba(15, 23, 42, 0.08);
            color: var(--text);
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {
            color: #94a3b8;
        }
        .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {
            border: 1.5px solid rgba(14, 165, 233, 0.6);
            box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.12);
            outline: none;
        }

        /* Dataframes */
        .stDataFrame {
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }
        .stDataFrame table {
            border-collapse: collapse !important;
        }

        /* Landing */
        .landing-hero {
            position: relative;
            overflow: hidden;
            padding: 32px;
            border-radius: 20px;
            border: 1px solid var(--border);
            background: linear-gradient(135deg, #fef2f2 0%, #ffffff 35%, #f3e8ff 100%);
            box-shadow: var(--shadow);
            margin-bottom: 18px;
        }
        .landing-hero .glow {
            position: absolute;
            filter: blur(32px);
            opacity: 0.65;
            z-index: 0;
        }
        .landing-hero .glow1 {
            width: 180px;
            height: 180px;
            background: rgba(225, 29, 72, 0.28);
            top: -40px;
            left: -40px;
        }
        .landing-hero .glow2 {
            width: 200px;
            height: 200px;
            background: rgba(91, 33, 182, 0.24);
            bottom: -60px;
            right: -60px;
        }
        .hero-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 20px;
            align-items: center;
        }
        .hero-right {
            display: grid;
            gap: 12px;
        }
        .stat-card {
            background: #fff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }
        .stat-card h4 {
            margin: 0 0 4px 0;
        }
        .stat-value {
            font-size: 20px;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 4px;
        }
        .stat-hint {
            color: var(--muted);
            font-size: 13px;
            margin: 0;
        }
        .feature-grid {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
            margin-top: 12px;
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }
        .feature-card h4 {
            margin: 0 0 6px 0;
        }
        .feature-card p {
            margin: 0;
            color: var(--muted);
        }
        @media (max-width: 900px) {
            .hero-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Sidebar specific */
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stTextArea textarea,
        [data-testid="stSidebar"] .stSelectbox select {
            color: #e5e7eb;
        }
        [data-testid="stSidebar"] .stTextInput input::placeholder,
        [data-testid="stSidebar"] .stTextArea textarea::placeholder {
            color: rgba(229, 231, 235, 0.7);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Theme-aware override
def inject_global_styles(palette):
    """Apply theme styles from palette."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

        :root {{
            --bg: {palette["bg"]};
            --card: {palette["card"]};
            --glass: {palette["glass"]};
            --border: {palette["border"]};
            --shadow: 0 20px 50px rgba(15, 23, 42, 0.12);
            --accent: {palette["accent"]};
            --accent-2: {palette["accent2"]};
            --text: {palette["text"]};
            --muted: {palette["muted"]};
            --chip-bg: {palette["chip_bg"]};
            --pill-bg: {palette["pill_bg"]};
            --pill-span: {palette["pill_span"]};
            --badge-bg: {palette["badge_bg"]};
            --badge-text: {palette["badge_text"]};
            --landing-bg: {palette["landing_bg"]};
            --glow1: {palette["glow1"]};
            --glow2: {palette["glow2"]};
            --button-shadow: {palette["button_shadow"]};
            --button-hover-shadow: {palette["button_hover_shadow"]};
            --input-bg: {palette["input_bg"]};
            --input-border: {palette["input_border"]};
            --placeholder: {palette["placeholder"]};
        }}

        .stApp {{
            background:
                radial-gradient(120% 120% at 15% 15%, {palette["grad1"]} 0%, transparent 35%),
                radial-gradient(120% 120% at 85% 10%, {palette["grad2"]} 0%, transparent 32%),
                var(--bg);
            color: var(--text);
        }}

        .block-container {{
            padding: 1.5rem 2.5rem 3rem;
        }}

        h1, h2, h3, h4, h5 {{
            font-family: 'Space Grotesk', 'Manrope', sans-serif;
            letter-spacing: -0.4px;
        }}

        .stMarkdown, p, label, .stTextInput, .stTextArea, .stSelectbox, .stCaption, .stRadio {{
            font-family: 'Manrope', sans-serif;
            color: var(--text);
        }}

        code {{
            background: rgba(15, 23, 42, 0.05);
            padding: 2px 6px;
            border-radius: 6px;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {palette["sidebar_from"]} 0%, {palette["sidebar_to"]} 100%);
            color: #f3f4f6;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}
        [data-testid="stSidebar"] * {{
            color: #f3f4f6 !important;
        }}
        [data-testid="stSidebar"] .stTextInput input, [data-testid="stSidebar"] .stFileUploader {{
            background: rgba(255, 255, 255, 0.08);
        }}

        /* Cards */
        .glass-card {{
            background: var(--glass);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: var(--shadow);
        }}
        .hero-card {{
            margin-bottom: 18px;
        }}
        .section-title {{
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .chip-row {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }}
        .chip {{
            padding: 8px 12px;
            border-radius: 999px;
            background: var(--chip-bg);
            color: var(--text);
            font-weight: 600;
            font-size: 13px;
        }}
        .pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: var(--pill-bg);
            color: var(--text);
            font-weight: 600;
            letter-spacing: 0.1px;
        }}
        .pill span {{
            background: var(--pill-span);
            color: #fff;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 12px;
        }}
        .badge {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: var(--badge-bg);
            color: var(--badge-text);
            font-weight: 700;
            font-size: 12px;
            letter-spacing: 0.2px;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab"] {{
            padding: 14px 18px;
            font-weight: 600;
            color: var(--muted);
            border-radius: 14px 14px 0 0;
            border: 1px solid transparent;
            background: rgba(255, 255, 255, 0.7);
        }}
        .stTabs [aria-selected="true"] {{
            background: var(--accent) !important;
            color: #fff !important;
            box-shadow: 0 12px 30px var(--button-shadow);
        }}

        /* Buttons */
        div.stButton > button {{
            width: 100%;
            border-radius: 12px;
            border: none;
            background: linear-gradient(120deg, var(--accent), var(--accent-2));
            color: #fff;
            font-weight: 700;
            box-shadow: 0 12px 30px var(--button-shadow);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }}
        div.stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 18px 40px var(--button-hover-shadow);
        }}
        div.stDownloadButton > button {{
            border-radius: 12px;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: #fff;
            color: var(--text);
            font-weight: 700;
        }}

        /* Inputs */
        .stTextInput input, .stTextArea textarea, .stSelectbox select {{
            background: var(--input-bg);
            border-radius: 12px;
            border: 1.5px solid var(--input-border);
            color: var(--text);
            box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.05);
        }}
        .stTextInput input::placeholder, .stTextArea textarea::placeholder {{
            color: var(--placeholder);
        }}
        .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {{
            border: 1.5px solid rgba(14, 165, 233, 0.6);
            box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.12);
            outline: none;
        }}

        /* Dataframes */
        .stDataFrame {{
            border-radius: 14px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }}
        .stDataFrame table {{
            border-collapse: collapse !important;
        }}

        /* Sidebar specific */
        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stTextArea textarea,
        [data-testid="stSidebar"] .stSelectbox select {{
            color: #f3f4f6;
        }}
        [data-testid="stSidebar"] .stTextInput input::placeholder,
        [data-testid="stSidebar"] .stTextArea textarea::placeholder {{
            color: rgba(243, 244, 246, 0.7);
        }}

        /* Landing */
        .landing-hero {{
            position: relative;
            overflow: hidden;
            padding: 32px;
            border-radius: 20px;
            border: 1px solid var(--border);
            background: var(--landing-bg);
            box-shadow: var(--shadow);
            margin-bottom: 18px;
        }}
        .landing-hero .glow {{
            position: absolute;
            filter: blur(32px);
            opacity: 0.65;
            z-index: 0;
        }}
        .landing-hero .glow1 {{
            width: 180px;
            height: 180px;
            background: var(--glow1);
            top: -40px;
            left: -40px;
        }}
        .landing-hero .glow2 {{
            width: 200px;
            height: 200px;
            background: var(--glow2);
            bottom: -60px;
            right: -60px;
        }}
        .hero-grid {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 20px;
            align-items: center;
        }}
        .hero-right {{
            display: grid;
            gap: 12px;
        }}
        .stat-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.06);
        }}
        .stat-card h4 {{
            margin: 0 0 4px 0;
        }}
        .stat-value {{
            font-size: 20px;
            font-weight: 800;
            color: var(--text);
            margin-bottom: 4px;
        }}
        .stat-hint {{
            color: var(--muted);
            font-size: 13px;
            margin: 0;
        }}
        .feature-grid {{
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 12px;
            margin-top: 12px;
        }}
        .feature-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 14px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
        }}
        .feature-card h4 {{
            margin: 0 0 6px 0;
        }}
        .feature-card p {{
            margin: 0;
            color: var(--muted);
        }}
        @media (max-width: 900px) {{
            .hero-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(prefer_logo=True):
    """Render centered logo on landing, fallback to text header."""
    logo_source = None
    if prefer_logo:
        if os.path.exists(LOGO_PATH):
            logo_source = LOGO_PATH
        else:
            # Optional: allow setting a URL via env or secrets
            env_logo = os.getenv("TASKER_LOGO_URL")
            secret_logo = getattr(st.secrets, "logo_url", None) if hasattr(st, "secrets") else None
            logo_source = env_logo or secret_logo

    if prefer_logo and logo_source:
        left, center, right = st.columns([1, 1.2, 1])
        with center:
            st.image(logo_source, use_column_width=True)
    else:
        st.markdown(
            '<h1 style="text-align:center; margin-top: -8px; margin-bottom: 12px;">Tasker.ai</h1>',
            unsafe_allow_html=True,
        )

# --- Main App ---
def main():
    if "show_landing" not in st.session_state:
        st.session_state["show_landing"] = True
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"

    top_cols = st.columns([5, 1.5])
    with top_cols[1]:
        dark_on = st.toggle(
            "🌗 Dark mode",
            value=st.session_state["theme"] == "dark",
            help="Toggle theme",
            label_visibility="visible",
        )
    st.session_state["theme"] = "dark" if dark_on else "light"
    palette = get_palette(st.session_state["theme"])

    inject_global_styles(palette)
    render_header(prefer_logo=st.session_state["show_landing"])

    # Landing page
    if st.session_state.get("show_landing", True):
        st.markdown(
            """
            <div class="landing-hero">
                <div class="glow glow1"></div>
                <div class="glow glow2"></div>
                <div class="hero-grid">
                    <div class="hero-left">
                        <div class="pill" style="margin-bottom: 10px;"><span>AI</span>Planning workspace</div>
                        <h2 style="margin: 4px 0 10px 0;">Plan, generate, and assign with clarity.</h2>
                        <p style="color: #475569; margin-bottom: 10px; max-width: 720px;">
                            Transform a project brief into a structured PRD, create tasks, and match them to your team without leaving one screen.
                        </p>
                        <div class="chip-row" style="margin-top: 12px;">
                            <div class="chip">PRD → Task list</div>
                            <div class="chip">Skill-based assignment</div>
                            <div class="chip">Email-ready reporting</div>
                        </div>
                    </div>
                    <div class="hero-right">
                        <div class="stat-card">
                            <div class="badge">Auto PRD</div>
                            <div class="stat-value">Structured in seconds</div>
                            <p class="stat-hint">Clear sections, ready to skim.</p>
                        </div>
                        <div class="stat-card">
                            <div class="badge">Smart match</div>
                            <div class="stat-value">Tasks → people</div>
                            <p class="stat-hint">Matches by skills with confidence.</p>
                        </div>
                        <div class="stat-card">
                            <div class="badge">Send fast</div>
                            <div class="stat-value">Email-ready</div>
                            <p class="stat-hint">Drop into your client and ship.</p>
                        </div>
                    </div>
                </div>
                <div class="feature-grid">
                    <div class="feature-card">
                        <h4>Built for focus</h4>
                        <p>One flow from idea to assignment—no extra dashboards.</p>
                    </div>
                    <div class="feature-card">
                        <h4>Concise tasks</h4>
                        <p>Clean, deduped outputs that stay readable.</p>
                    </div>
                    <div class="feature-card">
                        <h4>Exportable</h4>
                        <p>Download assignments or copy email drafts instantly.</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cta_col1, cta_col2, cta_col3 = st.columns([1, 1, 1])
        with cta_col2:
            if st.button("🚀 Open workspace", type="primary", use_container_width=True):
                st.session_state["show_landing"] = False
                st.rerun()
        st.stop()

    # --- Sidebar ---
    st.sidebar.title("Controls")
    st.sidebar.subheader("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload Employee CSV", type=["csv"])
    if uploaded_file:
        try:
            employees_df = pd.read_csv(uploaded_file)
            st.session_state["employees_df"] = employees_df
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {e}")

    # --- Main Content ---
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Project & PRD", "Task Generation", "Employees & Assignments", "Email Reports"]
    )

    # --- Project & PRD Tab ---
    with tab1:
        st.header("📋 Project & PRD")
        st.markdown("---")
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Set the vision</div>
                <p style="color: #475569; margin-bottom: 6px;">
                    Start from a curated template or write your own brief. Tasker.ai converts it into a structured PRD with features, tools, and milestones.
                </p>
                <div class="chip-row">
                    <div class="chip">Templates for common projects</div>
                    <div class="chip">Custom project briefs</div>
                    <div class="chip">One-click PRD</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([2, 1])
        
        with col1:
            default_projects = {
                "Select a project": "",
                "AI-Powered Customer Support Chatbot": "Develop a chatbot that can answer customer questions and resolve common issues. The chatbot should be able to understand natural language and provide personalized responses.",
                "E-commerce Website Redesign": "Redesign an e-commerce website to improve the user experience and increase sales. The project will involve updating the UI, improving navigation, and adding new features.",
                "Mobile App for Task Management": "Create a mobile app that helps users organize their tasks and stay productive. The app should have features like task creation, deadlines, and reminders.",
                "Create New Project": "",
            }
            project_choice = st.selectbox(
                "Choose a default project or create new", 
                list(default_projects.keys()),
                key="project_select"
            )
        
        with col2:
            if project_choice == "Create New Project":
                st.info("💡 Fill in the fields below to create your custom project")
        
        project_name = st.text_input(
            "📝 Project Name",
            value=project_choice if project_choice != "Select a project" and project_choice != "Create New Project" else "",
            placeholder="Enter your project name here...",
        )
        project_description = st.text_area(
            "📄 Project Description",
            value=default_projects[project_choice]
            if project_choice != "Select a project" and project_choice != "Create New Project"
            else "",
            height=200,
            placeholder="Describe your project in detail. Include features, goals, and requirements...",
        )

        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🚀 Generate PRD", type="primary", use_container_width=True):
                if not project_name or not project_description:
                    st.error("⚠️ Please enter a project name and description.")
                else:
                    generate_prd(project_name, project_description)

        if "prd" in st.session_state:
            st.markdown("---")
            st.subheader("✨ Generated PRD")
            with st.container():
                st.markdown(st.session_state["prd"])

    # --- Task Generation Tab ---
    with tab2:
        st.header("✅ Task Generation")
        st.markdown("---")
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">From PRD to actionable tasks</div>
                <p style="color: #475569; margin-bottom: 6px;">
                    Paste or reuse the PRD, then generate a clean, deduped task list. Edit freely and rerun as you refine scope.
                </p>
                <div class="chip-row">
                    <div class="chip">Understands context</div>
                    <div class="chip">Keeps tasks concise</div>
                    <div class="chip">No duplicates</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        prd_input = st.text_area(
            "📋 PRD Document", 
            value=st.session_state.get("prd", ""), 
            height=400,
            placeholder="Your PRD will appear here automatically after generation, or paste a PRD manually...",
            help="The PRD from the previous tab is automatically loaded here. You can also paste a custom PRD."
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            if st.button("🎯 Generate Tasks", type="primary", use_container_width=True):
                if not prd_input:
                    st.error("⚠️ Please paste the PRD in the text area or generate one in the Project & PRD tab.")
                else:
                    generate_tasks_from_prd(prd_input)

        if "tasks" in st.session_state:
            st.markdown("---")
            st.subheader("📝 Generated Tasks")
            tasks = st.session_state["tasks"]
            if isinstance(tasks, list):
                for i, task in enumerate(tasks, 1):
                    st.markdown(f"**{i}.** {task}")
            else:
                st.write(tasks)

    # --- Employees & Assignments Tab ---
    with tab3:
        st.header("👥 Employees & Assignments")
        st.markdown("---")
        
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Match work to the right people</div>
                <p style="color: #475569; margin-bottom: 6px;">
                    Upload a simple CSV, preview your team, and let Tasker.ai suggest the best fit for every task using skill similarity.
                </p>
                <div class="chip-row">
                    <div class="chip">CSV upload</div>
                    <div class="chip">Similarity scoring</div>
                    <div class="chip">Instant CSV export</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        if "employees_df" in st.session_state:
            st.subheader("📊 Available Employees")
            st.dataframe(st.session_state["employees_df"], use_container_width=True, hide_index=True)

            if "tasks" in st.session_state:
                col_btn1, col_btn2 = st.columns([1, 4])
                with col_btn1:
                    if st.button("🎯 Assign Tasks", type="primary", use_container_width=True):
                        assign_tasks(
                            st.session_state["tasks"], st.session_state["employees_df"]
                        )
            else:
                st.info("ℹ️ Generate tasks first in the 'Task Generation' tab to assign them.")

            if "assignments_df" in st.session_state:
                st.markdown("---")
                st.subheader("✅ Task Assignments")
                st.dataframe(
                    st.session_state["assignments_df"], use_container_width=True, hide_index=True
                )
                csv = st.session_state["assignments_df"].to_csv(index=False)
                st.download_button(
                    "📥 Download Assignments as CSV",
                    csv,
                    "task_assignments.csv",
                    "text/csv",
                    type="primary",
                )
        else:
            st.warning("⚠️ Please upload an employee CSV file in the sidebar.")
            st.info("💡 The CSV should have columns: `name` and `skills`")

    # --- Email Reports Tab (draft only, no sending) ---
    with tab4:
        st.header("📧 Email Reports")
        st.markdown("---")
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Share clean updates</div>
                <p style="color: #475569; margin-bottom: 6px;">
                    Generate a concise status email with real assignment data. Edit in-place and send from your client with confidence.
                </p>
                <div class="chip-row">
                    <div class="chip">Uses real assignments</div>
                    <div class="chip">Project context included</div>
                    <div class="chip">Editable before sending</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "assignments_df" not in st.session_state:
            st.info("Generate and assign tasks first to draft an email report.")
        else:
            assignments_df = st.session_state["assignments_df"]

            st.subheader("Draft Email")
            st.caption(f"Project: {st.session_state.get('project_name', 'Not set')}")

            col_from, col_to = st.columns(2)
            with col_from:
                from_email = st.text_input(
                    "From",
                    value=st.session_state.get("email_from", "santhanakrishnan@cua.edu"),
                    help="Sender email address",
                )
            with col_to:
                to_email = st.text_input(
                    "To",
                    value=st.session_state.get("email_to", "aswath.mcs@gmail.com"),
                    help="Recipient email address",
                )

            subject = st.text_input(
                "Subject",
                value=st.session_state.get("email_subject", "Task Assignment Report"),
            )
            signature = st.text_area(
                "Signature / Company footer",
                value=st.session_state.get(
                    "email_signature", "Best regards,\nTasker.ai Team"
                ),
                height=80,
                help="Include your company name or personal sign-off here.",
            )

            col_gen, col_preview = st.columns([1, 2])
            with col_gen:
                if st.button("🪄 Generate Email Draft", type="primary", use_container_width=True):
                    email_body = generate_email_report(
                        assignments_df,
                        from_email,
                        to_email,
                        signature,
                        st.session_state.get("project_name", "Project"),
                        st.session_state.get("prd", ""),
                        st.session_state.get("tasks", []),
                    )
                    if email_body:
                        st.session_state["email_body"] = email_body
                        st.session_state["email_from"] = from_email
                        st.session_state["email_to"] = to_email
                        st.session_state["email_subject"] = subject
                        st.session_state["email_signature"] = signature
                    else:
                        st.error("Could not generate email. Please try again.")

            email_body = st.text_area(
                "Email Body",
                value=st.session_state.get("email_body", ""),
                height=260,
                help="Edit the draft before sending from your email client.",
            )
            st.session_state["email_body"] = email_body

            with st.expander("📄 Assignment summary used for the email"):
                st.dataframe(assignments_df, use_container_width=True, hide_index=True)

@st.cache_resource
def load_text_generator():
    """Load the text generation model (cached to avoid reloading)"""
    try:
        # Using distilgpt2 - smaller and faster than gpt2, good for basic text generation
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1 if not torch.cuda.is_available() else 0,  # Use CPU if no GPU
        )
        return generator
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def generate_from_model(prompt):
    """Generate text using local model"""
    try:
        # Load model if not already loaded
        if st.session_state["text_generator"] is None:
            with st.spinner("Loading AI model (first time only, this may take a moment)..."):
                generator = load_text_generator()
                if generator is None:
                    return None
                st.session_state["text_generator"] = generator
                st.session_state["model_loaded"] = True
        
        generator = st.session_state["text_generator"]

        tokenizer = generator.tokenizer
        # GPT-2 family supports ~1024 tokens context; keep prompt well under that.
        max_ctx = (
            getattr(generator.model.config, "n_positions", None)
            or getattr(generator.model.config, "max_position_embeddings", 1024)
            or 1024
        )
        max_new_tokens = 160
        max_prompt_tokens = max(64, max_ctx - max_new_tokens - 8)

        # Format prompt for better generation
        formatted_prompt = (prompt or "").strip()

        # Token-safe truncation from the end of the prompt
        encoded = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        if len(encoded) > max_prompt_tokens:
            encoded = encoded[-max_prompt_tokens:]
            formatted_prompt = tokenizer.decode(encoded, skip_special_tokens=True)
        formatted_prompt = formatted_prompt + "\n\n"

        # Generate text
        results = generator(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            truncation=True,
            return_full_text=True,
        )

        generated_text = results[0].get("generated_text", "")
        
        # Remove the original prompt from the generated text
        if generated_text.startswith(formatted_prompt):
            generated_text = generated_text[len(formatted_prompt):].strip()
        
        return generated_text
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return ""


def generate_prd(project_name, project_description):
    with st.spinner("🤖 Generating comprehensive PRD document..."):
        # Generate multiple sections using the model
        # Overview section
        overview_prompt = f"Product Requirements Document for {project_name}. Overview: {project_description}"
        overview_text = generate_from_model(overview_prompt)
        st.session_state["project_name"] = project_name
        st.session_state["project_description"] = project_description
        
        # Features section - extract from description
        features_prompt = f"Key features and functionalities for {project_name}: {project_description}"
        features_text = generate_from_model(features_prompt)
        
        # Tools and technologies section
        tools_prompt = f"Technologies, tools, and frameworks needed for {project_name}: {project_description}"
        tools_text = generate_from_model(tools_prompt)
        
        # Parse generated content
        def extract_bullet_points(text, max_items=8):
            """Extract meaningful bullet points from generated text"""
            if not text:
                return []
            items = []
            # Split by sentences and common separators
            sentences = text.replace('\n', ' ').split('.')
            for sent in sentences:
                sent = sent.strip()
                # Filter meaningful sentences
                if len(sent) > 15 and len(sent) < 150:
                    # Remove common prefixes
                    for prefix in ['The', 'This', 'It', 'A', 'An']:
                        if sent.startswith(prefix + ' '):
                            sent = sent[len(prefix) + 1:].strip()
                            break
                    if sent and sent[0].isupper():
                        items.append(sent)
                if len(items) >= max_items:
                    break
            return items
        
        # Extract features from description and generated text
        features_list = []
        desc_lower = project_description.lower()
        
        # Extract explicit features mentioned in description
        desc_sentences = project_description.split('.')
        for sent in desc_sentences:
            sent = sent.strip()
            if any(keyword in sent.lower() for keyword in ['feature', 'should', 'must', 'need', 'include', 'have']):
                if len(sent) > 10 and len(sent) < 200:
                    features_list.append(sent)
        
        # Add generated features
        if features_text:
            gen_features = extract_bullet_points(features_text, max_items=6)
            features_list.extend(gen_features)
        
        # Remove duplicates and limit
        seen = set()
        unique_features = []
        for feat in features_list:
            feat_lower = feat.lower()[:50]  # Use first 50 chars for comparison
            if feat_lower not in seen:
                seen.add(feat_lower)
                unique_features.append(feat)
        features_list = unique_features[:10]
        
        # If still no features, create intelligent defaults based on keywords
        if not features_list:
            if 'chatbot' in desc_lower or 'ai' in desc_lower:
                features_list = [
                    "Natural language processing and understanding capabilities",
                    "Conversational user interface with context awareness",
                    "Integration with knowledge base and FAQ system",
                    "Multi-channel support (web, mobile, API)"
                ]
            elif 'e-commerce' in desc_lower or 'shopping' in desc_lower:
                features_list = [
                    "Product catalog with search and filtering",
                    "Shopping cart and checkout system",
                    "Secure payment gateway integration",
                    "Order management and tracking system"
                ]
            elif 'ios' in desc_lower or 'iphone' in desc_lower or 'ipad' in desc_lower:
                features_list = [
                    "Native iOS application with Swift/SwiftUI",
                    "iOS Human Interface Guidelines compliance",
                    "App Store integration and submission",
                    "Core Data or CloudKit for data persistence",
                    "Push notifications via APNs"
                ]
            elif 'android' in desc_lower and ('ios' not in desc_lower and 'iphone' not in desc_lower):
                features_list = [
                    "Native Android application with Kotlin/Java",
                    "Material Design guidelines compliance",
                    "Google Play Store integration",
                    "Room or SQLite for local database",
                    "Firebase Cloud Messaging for push notifications"
                ]
            elif 'mobile' in desc_lower or 'app' in desc_lower:
                # Check if it's cross-platform or native
                if 'native' in desc_lower or 'swift' in desc_lower or 'kotlin' in desc_lower:
                    # Native development
                    if 'ios' in desc_lower or 'iphone' in desc_lower:
                        features_list = [
                            "Native iOS application with Swift/SwiftUI",
                            "iOS Human Interface Guidelines compliance",
                            "App Store integration",
                            "Core Data for local storage"
                        ]
                    else:
                        features_list = [
                            "Native mobile application",
                            "Platform-specific UI/UX",
                            "App store integration",
                            "Local data persistence"
                        ]
                else:
                    # Cross-platform
                    features_list = [
                        "Cross-platform mobile application (iOS/Android)",
                        "Offline functionality and data synchronization",
                        "Push notifications for updates and reminders",
                        "User authentication and profile management"
                    ]
            else:
                features_list = [
                    "User-friendly and intuitive interface",
                    "Core functionality as per requirements",
                    "Data management and storage system",
                    "Security and authentication mechanisms"
                ]
        
        # Extract tools and technologies
        tools_list = []
        if tools_text:
            tools_list = extract_bullet_points(tools_text, max_items=8)
        
        # Intelligent tool detection based on project type - CHECK iOS/Android FIRST
        if not tools_list:
            # iOS Native Development
            if 'ios' in desc_lower or 'iphone' in desc_lower or 'ipad' in desc_lower or ('native' in desc_lower and 'ios' in project_name.lower()):
                tools_list = [
                    "Xcode - Apple's integrated development environment (IDE)",
                    "Swift programming language for iOS development",
                    "SwiftUI or UIKit for user interface development",
                    "Core Data or CloudKit for data persistence",
                    "CocoaPods or Swift Package Manager for dependency management",
                    "TestFlight for beta testing",
                    "App Store Connect for app distribution"
                ]
            # Android Native Development
            elif 'android' in desc_lower and ('ios' not in desc_lower and 'iphone' not in desc_lower) or ('native' in desc_lower and 'android' in project_name.lower()):
                tools_list = [
                    "Android Studio - Official Android IDE",
                    "Kotlin or Java programming language",
                    "Jetpack Compose or XML layouts for UI",
                    "Room or SQLite for local database",
                    "Gradle for build automation and dependency management",
                    "Google Play Console for app distribution",
                    "Firebase for backend services (optional)"
                ]
            # Cross-platform Mobile (React Native/Flutter)
            elif ('react native' in desc_lower or 'flutter' in desc_lower or 'cross-platform' in desc_lower) and ('native' not in desc_lower):
                tools_list = [
                    "React Native or Flutter for cross-platform development",
                    "Firebase or AWS for backend services",
                    "SQLite or Realm for local database",
                    "RESTful API for server communication"
                ]
            # Generic Mobile App (assume cross-platform if not specified)
            elif 'mobile' in desc_lower or 'app' in desc_lower:
                # Check for native keywords
                if 'native' in desc_lower or 'swift' in desc_lower or 'xcode' in desc_lower:
                    tools_list = [
                        "Xcode and Swift for iOS development",
                        "SwiftUI or UIKit framework",
                        "Core Data for local storage",
                        "App Store Connect for distribution"
                    ]
                elif 'kotlin' in desc_lower or 'android studio' in desc_lower:
                    tools_list = [
                        "Android Studio and Kotlin for Android development",
                        "Jetpack Compose or XML layouts",
                        "Room database for local storage",
                        "Google Play Console for distribution"
                    ]
                else:
                    tools_list = [
                        "React Native or Flutter for cross-platform development",
                        "Firebase or AWS for backend services",
                        "SQLite or Realm for local database",
                        "RESTful API for server communication"
                    ]
            elif 'web' in desc_lower or 'website' in desc_lower:
                tools_list = [
                    "React.js or Vue.js for frontend framework",
                    "Node.js or Python Django/Flask for backend",
                    "PostgreSQL or MongoDB for database",
                    "Docker for containerization and deployment"
                ]
            elif 'ai' in desc_lower or 'chatbot' in desc_lower:
                tools_list = [
                    "Python with TensorFlow or PyTorch for ML models",
                    "NLTK or spaCy for NLP processing",
                    "FastAPI or Flask for API development",
                    "Vector database (Pinecone/Weaviate) for embeddings"
                ]
            else:
                tools_list = [
                    "Modern web framework (React/Vue/Angular)",
                    "Backend API framework (Node.js/Python/Java)",
                    "Database system (PostgreSQL/MySQL/MongoDB)",
                    "Cloud hosting platform (AWS/Azure/GCP)"
                ]
        
        # Build comprehensive PRD
        full_prd = f"""# Product Requirements Document: {project_name}

## 1. Overview
{project_description}

{overview_text[:300] if overview_text else ''}

## 2. Core Features
{chr(10).join(f'- {feat}' for feat in features_list)}

## 3. Technical Requirements

### Tools & Technologies
{chr(10).join(f'- {tool}' for tool in tools_list)}

### Architecture
- **Frontend**: Modern framework based on project requirements
- **Backend**: Scalable API architecture
- **Database**: Appropriate database solution (relational or NoSQL)
- **Deployment**: Cloud-based infrastructure with CI/CD pipeline
- **Security**: Authentication, authorization, and data encryption

## 4. Success Metrics
- User engagement and satisfaction rates
- Performance benchmarks (response time, uptime)
- Scalability and load handling capabilities
- Feature adoption and usage analytics

## 5. Timeline & Milestones
- Phase 1: Planning and Design
- Phase 2: Core Development
- Phase 3: Testing and Quality Assurance
- Phase 4: Deployment and Monitoring
"""
        st.session_state["prd"] = full_prd
        st.success("✅ PRD generated successfully!")

def generate_tasks_from_prd(prd_input):
    with st.spinner("🤖 Generating comprehensive task list from PRD..."):
        # Generate tasks using model based on PRD content
        task_prompt = f"Generate a detailed task list for this project PRD:\n\n{prd_input[:1000]}\n\nTasks:"
        tasks_text = generate_from_model(task_prompt)
        
        tasks = []
        prd_lower = prd_input.lower()
        project_name_lower = prd_lower.split('product requirements document:')[1].split('\n')[0].strip().lower() if 'product requirements document:' in prd_lower else ""
        
        # Detect platform type
        is_ios = 'ios' in prd_lower or 'iphone' in prd_lower or 'ipad' in prd_lower or 'xcode' in prd_lower or 'swift' in prd_lower or 'ios' in project_name_lower
        is_android = ('android' in prd_lower or 'kotlin' in prd_lower or 'android studio' in prd_lower) and not is_ios
        is_native_mobile = is_ios or is_android
        is_cross_platform = ('react native' in prd_lower or 'flutter' in prd_lower or 'cross-platform' in prd_lower) and not is_native_mobile
        
        # Phase 1: Planning and Setup
        tasks.append("Review and analyze PRD requirements thoroughly")
        tasks.append("Create detailed technical design document")
        
        # Platform-specific setup tasks
        if is_ios:
            tasks.append("Install and configure Xcode development environment")
            tasks.append("Set up Apple Developer account and certificates")
            tasks.append("Create new Xcode project with Swift/SwiftUI")
            tasks.append("Configure project settings (bundle ID, version, etc.)")
        elif is_android:
            tasks.append("Install and configure Android Studio")
            tasks.append("Set up Android SDK and required tools")
            tasks.append("Create new Android project with Kotlin/Java")
            tasks.append("Configure app manifest and build.gradle")
        elif is_cross_platform:
            tasks.append("Set up React Native or Flutter development environment")
            tasks.append("Initialize cross-platform project structure")
            tasks.append("Configure platform-specific settings")
        else:
            tasks.append("Set up development environment and tools")
        
        tasks.append("Initialize project repository and version control")
        
        # Extract features from PRD and create tasks
        lines = prd_input.split('\n')
        feature_section = False
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if 'feature' in line_lower and ('##' in line or '###' in line):
                feature_section = True
                continue
            if feature_section and ('##' in line or '###' in line) and 'feature' not in line_lower:
                feature_section = False
            if feature_section and '- ' in line:
                feature = line.split('- ', 1)[1].strip()
                if len(feature) > 5 and len(feature) < 150:
                    tasks.append(f"Implement feature: {feature}")
        
        # Extract tools and create setup tasks
        tools_section = False
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if 'tool' in line_lower or 'technolog' in line_lower:
                tools_section = True
                continue
            if tools_section and ('##' in line or '###' in line):
                tools_section = False
            if tools_section and '- ' in line:
                tool = line.split('- ', 1)[1].strip()
                if len(tool) > 5:
                    # Skip generic setup if it's iOS/Android specific
                    if is_ios and 'xcode' in tool.lower():
                        continue  # Already added above
                    if is_android and 'android studio' in tool.lower():
                        continue  # Already added above
                    tasks.append(f"Set up and configure {tool}")
        
        # Platform-specific development tasks
        if is_ios:
            tasks.append("Design iOS UI/UX following Human Interface Guidelines")
            tasks.append("Implement SwiftUI views or UIKit components")
            tasks.append("Set up Core Data or CloudKit for data persistence")
            tasks.append("Configure App Store Connect and app metadata")
            tasks.append("Implement push notifications using APNs")
            tasks.append("Add app icons and launch screens for all device sizes")
        elif is_android:
            tasks.append("Design Android UI/UX following Material Design guidelines")
            tasks.append("Implement Jetpack Compose or XML layouts")
            tasks.append("Set up Room database or SQLite for local storage")
            tasks.append("Configure Google Play Console and app listing")
            tasks.append("Implement Firebase Cloud Messaging for push notifications")
            tasks.append("Add app icons and adaptive icons for different densities")
        else:
            # Core development tasks based on PRD content (web/cross-platform)
            if 'frontend' in prd_lower or 'ui' in prd_lower or 'interface' in prd_lower:
                tasks.append("Design user interface mockups and wireframes")
                tasks.append("Implement responsive frontend components")
                tasks.append("Integrate frontend with backend APIs")
        
        if 'backend' in prd_lower or 'api' in prd_lower:
            tasks.append("Design and develop RESTful API endpoints")
            tasks.append("Implement API authentication and authorization")
            tasks.append("Create API documentation")
        
        if 'database' in prd_lower or 'data' in prd_lower:
            tasks.append("Design database schema and relationships")
            tasks.append("Implement database migrations")
            tasks.append("Set up database indexing and optimization")
        
        if 'authentication' in prd_lower or 'security' in prd_lower:
            tasks.append("Implement user authentication system")
            tasks.append("Add security measures and data encryption")
            tasks.append("Set up role-based access control")
        
        # Parse generated tasks from model
        if tasks_text:
            gen_lines = tasks_text.split('\n')
            for line in gen_lines:
                line = line.strip()
                if line:
                    # Remove numbering
                    for prefix in ['1.', '2.', '3.', '4.', '5.', '-', '*']:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                            break
                    if len(line) > 10 and len(line) < 200:
                        # Avoid duplicates
                        if not any(line.lower() in existing.lower() or existing.lower() in line.lower() for existing in tasks):
                            tasks.append(line)
        
        # Testing and deployment phase - platform specific
        tasks.append("Write comprehensive unit tests")
        tasks.append("Implement integration tests")
        
        if is_ios:
            tasks.append("Test on iOS Simulator and physical devices")
            tasks.append("Configure TestFlight for beta testing")
            tasks.append("Submit app for App Store review")
            tasks.append("Set up App Store analytics and crash reporting")
        elif is_android:
            tasks.append("Test on Android emulator and physical devices")
            tasks.append("Set up internal testing track in Google Play Console")
            tasks.append("Submit app for Google Play Store review")
            tasks.append("Configure Google Play Console analytics")
        else:
            tasks.append("Perform code review and refactoring")
            tasks.append("Set up CI/CD pipeline")
            tasks.append("Deploy to staging environment")
            tasks.append("Perform user acceptance testing (UAT)")
            tasks.append("Deploy to production environment")
            tasks.append("Set up monitoring and logging")
        
        tasks.append("Create user documentation and guides")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tasks = []
        for task in tasks:
            task_lower = task.lower()[:60]  # Use first 60 chars for comparison
            if task_lower not in seen:
                seen.add(task_lower)
                unique_tasks.append(task)
        
        # Limit to reasonable number but keep important ones
        if len(unique_tasks) > 25:
            # Keep first 5 (planning), middle tasks (development), and last 5 (deployment)
            unique_tasks = unique_tasks[:5] + unique_tasks[5:-5][:15] + unique_tasks[-5:]
        
        st.session_state["tasks"] = unique_tasks
        st.success(f"✅ Generated {len(unique_tasks)} tasks successfully!")

def format_assignments_summary(assignments_df):
    """Create a concise summary of task->assignee mapping."""
    lines = []
    for _, row in assignments_df.iterrows():
        task = row.get("Task", "")
        assignee = row.get("Assigned To", "")
        skills = row.get("Skills", "")
        lines.append(f"- {task} -> {assignee} (Skills: {skills})")
    return "\n".join(lines)

def generate_email_report(assignments_df, from_email, to_email, signature, project_name, prd_text, tasks_list):
    """Use the local text generator to craft a concise status email, anchored on real assignment data."""
    summary = format_assignments_summary(assignments_df)
    task_lines = "\n".join(f"- {t}" for t in tasks_list[:12]) if tasks_list else ""
    prd_excerpt = (prd_text or "")[:400]

    prompt = (
        "Write a short, professional status email. Keep it concise and client-ready.\n"
        f"Project name: {project_name}\n"
        f"From: {from_email}\n"
        f"To: {to_email}\n"
        "Structure:\n"
        "1) Greeting and one-sentence status summary about assignments being created.\n"
        "2) Bullet list EXACTLY using the provided assignments list (do not invent or repeat). Keep them brief.\n"
        "3) One sentence tying back to scope/context from the PRD excerpt.\n"
        "4) Closing line that invites follow-up.\n"
        f"Signature block (use as-is):\n{signature}\n"
        "Assignments list (use these bullets verbatim, do not add new items):\n"
        f"{summary}\n"
        f"Tasks (raw list, optional to mention count):\n{task_lines}\n"
        f"PRD excerpt:\n{prd_excerpt}\n"
        "Return only the email body, no subject line."
    )

    generated = generate_from_model(prompt) or ""

    # If the model output lacks our required bullets, fall back to a deterministic version
    if summary not in generated:
        deterministic = (
            f"Hello,\n\nHere is the latest assignment update for {project_name}:\n"
            f"{summary}\n\n"
            f"Scope reference: {prd_excerpt[:200]}...\n\n"
            f"Please let me know if you need any changes or additional details.\n\n{signature}"
        )
        return deterministic

    return generated.strip()

def assign_tasks(tasks, employees_df):
    with st.spinner("🤖 Matching tasks to employees based on skills..."):
        assignments = []
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            employee_skills = employees_df["skills"].tolist()
            skill_embeddings = model.encode(employee_skills, convert_to_tensor=True)

            for task in tasks:
                task_embedding = model.encode(task, convert_to_tensor=True)
                cosine_scores = util.pytorch_cos_sim(task_embedding, skill_embeddings)
                
                # Convert tensor to CPU and then to numpy to avoid device error
                cosine_scores_cpu = cosine_scores.cpu().detach().numpy()
                best_employee_idx = np.argmax(cosine_scores_cpu)
                confidence_score = float(cosine_scores_cpu[0][best_employee_idx])
                
                assignments.append(
                    {
                        "Task": task,
                        "Assigned To": employees_df.iloc[best_employee_idx]["name"],
                        "Skills": employees_df.iloc[best_employee_idx]["skills"],
                        "Confidence": f"{confidence_score:.2%}",
                    }
                )
            assignments_df = pd.DataFrame(assignments)
            st.session_state["assignments_df"] = assignments_df
            st.success(f"✅ Successfully assigned {len(assignments)} tasks!")
        except Exception as e:
            st.error(f"❌ Error assigning tasks: {e}")
            import traceback
            st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
