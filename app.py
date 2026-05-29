# =============================================================================
# app.py — Rashomon Set Visualizer
# =============================================================================
# Main Streamlit web application. Renders seven interactive screens that
# together tell the full story of model multiplicity in machine learning.
#
# SCREEN MAP:
#   1. Home         — Hero introduction, visitor guide, three key findings
#   2. Dataset      — Raw heart disease data, feature dictionary, distributions
#   3. Build the Set— Scatter of 3360 models with live epsilon slider
#   4. Patient View — Per-patient donut chart showing model disagreement
#   5. Features     — Feature importance mean/std across all Rashomon models
#   6. Epsilon      — Line chart showing set size vs tolerance
#   7. About        — Methodology, research papers, limitations, tech stack
#
# HOW IT WORKS (high level):
#   1. On first load, load_everything() trains 3360 decision trees,
#      evaluates them with 5-Fold CV, builds the Rashomon Set,
#      and runs all four analyses. This takes ~2 minutes.
#   2. @st.cache_data stores the result in memory. Every subsequent page
#      click is instant — no retraining needed.
#   3. st.session_state["page"] tracks which screen the user is on.
#      Sidebar buttons update this and call st.rerun() to re-render.
#   4. main() reads session_state["page"] and calls the correct render_*()
#      function.
#
# DESIGN:
#   Theme:  Dark scientific dashboard (navy + teal + coral + gold)
#   Fonts:  DM Serif Display (headings) · JetBrains Mono (data/code) · Inter (body)
#   Charts: Plotly with custom dark theme applied via apply_theme()
# =============================================================================

import streamlit as st      # web app framework — renders UI, handles state
import pandas as pd          # data tables — used for the patient dataframe
import numpy as np           # numerical operations — arrays, arange, random
import plotly.graph_objects as go  # low-level Plotly — full chart control
import plotly.express as px  # high-level Plotly — quick charts (imported but available)
import sys
import os

# Add the project root to Python's module search path.
# This lets us do "from rashomon.train import ..." even when Streamlit
# runs from a different working directory.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our two core modules:
#   train.py   — downloads data, builds the Rashomon Set (3360 models)
#   analyze.py — computes the four analyses on the Rashomon Set
from rashomon.train   import load_data, build_rashomon_set
from rashomon.analyze import run_all_analyses


# =============================================================================
# PAGE CONFIG
# =============================================================================
# st.set_page_config() MUST be the very first Streamlit call in the script.
# Any other st.* call before this will raise an error.
# layout="wide" uses the full browser width instead of the narrow default.

st.set_page_config(
    page_title="Rashomon Set Visualizer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# GLOBAL CSS
# =============================================================================
# We inject a <style> block into the HTML that Streamlit serves.
# unsafe_allow_html=True is required to inject raw HTML/CSS.
#
# Structure:
#   - Google Fonts import (DM Serif Display, JetBrains Mono, Inter)
#   - CSS variables (:root) for colors, fonts — change one variable to
#     retheme the entire app
#   - Background / sidebar overrides to apply the dark navy theme
#   - Reusable component classes: .card, .metric-box, .callout, .tag, etc.
#   - Streamlit widget overrides (selectbox, slider, tabs)
#
# WHY CSS VARIABLES?
#   If Prof. Semenova or anyone else wants to change the accent color from
#   teal (#00d4aa) to blue, they change one line in :root. Everything else
#   updates automatically.

st.markdown("""
<style>
/* Google Fonts — three complementary typefaces */
/* DM Serif Display: elegant serif for headings  */
/* JetBrains Mono:   monospace for numbers/code  */
/* Inter:            clean sans-serif for body   */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600&display=swap');

/* Design Tokens — change here to retheme the entire app */
:root {
    --bg-primary:    #0a0e1a;   /* page background — very dark navy      */
    --bg-secondary:  #111827;   /* sidebar background                    */
    --bg-card:       #1a2235;   /* card/panel background                 */
    --border:        #2a3a55;   /* default border color                  */
    --border-bright: #3a5080;   /* scrollbar thumb color                 */
    --teal:          #00d4aa;   /* primary accent                        */
    --teal-dim:      #00a882;   /* dimmer teal for card borders          */
    --coral:         #ff6b6b;   /* disease/danger color                  */
    --gold:          #ffd166;   /* epsilon/warning color                 */
    --purple:        #a29bfe;   /* about/meta color                      */
    --text-primary:  #e8edf5;   /* main readable text                    */
    --text-secondary:#8899bb;   /* secondary descriptive text            */
    --text-muted:    #4a5a75;   /* footnotes/labels                      */
    --font-display:  'DM Serif Display', Georgia, serif;
    --font-mono:     'JetBrains Mono', 'Courier New', monospace;
    --font-body:     'Inter', sans-serif;
}

/* Global background — override Streamlit's nested divs */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

/* Hide Streamlit's default chrome (hamburger menu, footer, header bar) */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Card components — four color variants */
/* .card          — neutral dark panel                */
/* .card-highlight — teal-accented for positive findings */
/* .card-coral     — coral-accented for warnings      */
/* .card-gold      — gold-accented for epsilon notes  */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-highlight {
    background: linear-gradient(135deg, #0d1f35, #0a1828);
    border: 1px solid var(--teal-dim);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-coral {
    background: linear-gradient(135deg, #1f0d0d, #180a0a);
    border: 1px solid #cc5555;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-gold {
    background: linear-gradient(135deg, #1f1a0d, #18150a);
    border: 1px solid #aa8833;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Metric box — large number + small uppercase label */
.metric-box {
    flex: 1;
    min-width: 140px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-family: var(--font-mono);
    font-size: 28px;
    font-weight: 700;
    color: var(--teal);
    display: block;
    line-height: 1.1;
}
.metric-label {
    font-family: var(--font-body);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-secondary);
    margin-top: 6px;
    display: block;
}

/* Typography — page headings and section markers */
.page-title {
    font-family: var(--font-display);
    font-size: 42px;
    color: var(--text-primary);
    line-height: 1.1;
    margin-bottom: 6px;
}
.page-subtitle {
    font-family: var(--font-body);
    font-size: 15px;
    color: var(--text-secondary);
    margin-bottom: 32px;
    line-height: 1.6;
}
.section-label {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--teal);
    margin-bottom: 12px;
    display: block;
}

/* Callout boxes — left-bordered insight panels */
/* .callout        — teal  (general insight)     */
/* .callout.coral  — coral (warning/problem)     */
/* .callout.gold   — gold  (opportunity/note)    */
/* .callout.purple — purple (academic context)   */
.callout {
    background: rgba(0,212,170,0.06);
    border-left: 3px solid var(--teal);
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary);
}
.callout.coral  { background: rgba(255,107,107,0.06);  border-left-color: var(--coral);  }
.callout.gold   { background: rgba(255,209,102,0.06);  border-left-color: var(--gold);   }
.callout.purple { background: rgba(162,155,254,0.06);  border-left-color: var(--purple); }

/* Patient diagnosis badge pills */
.patient-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 500;
}
.badge-disease   { background: rgba(255,107,107,0.15); color: var(--coral);  border: 1px solid rgba(255,107,107,0.3); }
.badge-nodisease { background: rgba(0,212,170,0.15);   color: var(--teal);   border: 1px solid rgba(0,212,170,0.3);  }
.badge-uncertain { background: rgba(255,209,102,0.15); color: var(--gold);   border: 1px solid rgba(255,209,102,0.3);}

/* Feature measurement bars in Patient View */
/* .feature-bar-fill width is set inline from Python (0% to 100%) */
.feature-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    font-size: 13px;
}
.feature-name    { width: 80px; font-family: var(--font-mono); color: var(--text-secondary); font-size: 12px; }
.feature-bar-bg  { flex: 1; height: 6px; background: var(--border); border-radius: 3px; }
.feature-bar-fill{ height: 6px; border-radius: 3px; background: var(--teal); }
.feature-val     { width: 60px; text-align: right; font-family: var(--font-mono); font-size: 12px; }

/* Override Streamlit widget defaults to match dark theme */
div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}
[data-testid="stTabs"] button {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    color: var(--text-secondary) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--teal) !important;
    border-bottom-color: var(--teal) !important;
}

/* Custom scrollbar */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }

/* Tag pills — used in About page for paper venues and stack items */
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    margin: 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.tag-teal   { background: rgba(0,212,170,0.12);   color: var(--teal);   border: 1px solid rgba(0,212,170,0.25);   }
.tag-coral  { background: rgba(255,107,107,0.12);  color: var(--coral);  border: 1px solid rgba(255,107,107,0.25);  }
.tag-gold   { background: rgba(255,209,102,0.12);  color: var(--gold);   border: 1px solid rgba(255,209,102,0.25);  }
.tag-purple { background: rgba(162,155,254,0.12);  color: var(--purple); border: 1px solid rgba(162,155,254,0.25);  }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PLOTLY DARK THEME
# =============================================================================
# A shared theme dictionary applied to every chart via apply_theme(fig).
# Transparent backgrounds let the dark navy card color show through.
# All five charts use the same grid color, font, and hover style.

PLOTLY_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",   # transparent — shows card bg
    "plot_bgcolor":  "rgba(0,0,0,0)",   # transparent — shows card bg
    "font": {
        "family": "JetBrains Mono, monospace",
        "color":  "#8899bb",
        "size":   11
    },
    "xaxis": {
        "gridcolor": "#1e2a3d",   # very subtle grid lines
        "linecolor": "#2a3a55",   # axis line
        "tickcolor": "#2a3a55",
        "tickfont":  {"color": "#8899bb", "size": 10}
    },
    "yaxis": {
        "gridcolor": "#1e2a3d",
        "linecolor": "#2a3a55",
        "tickcolor": "#2a3a55",
        "tickfont":  {"color": "#8899bb", "size": 10}
    },
    # Default color sequence for auto-assigned series colors
    "colorway": ["#00d4aa", "#ff6b6b", "#ffd166", "#74b9ff", "#a29bfe"],
    # Tooltip (hover box) styling
    "hoverlabel": {
        "bgcolor":     "#1a2235",
        "bordercolor": "#2a3a55",
        "font": {"family": "JetBrains Mono", "color": "#e8edf5", "size": 11}
    }
}

def apply_theme(fig, height=420):
    """
    Apply the global dark Plotly theme to any figure.
    Called after every chart's traces and layout are defined.

    Args:
        fig    (go.Figure): the Plotly figure to style
        height (int):       chart height in pixels (default 420)

    Returns:
        fig — same figure returned for method chaining
    """
    fig.update_layout(
        paper_bgcolor = PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor  = PLOTLY_THEME["plot_bgcolor"],
        font          = PLOTLY_THEME["font"],
        hoverlabel    = PLOTLY_THEME["hoverlabel"],
        height        = height,
        margin        = dict(l=40, r=20, t=40, b=40),
        legend        = dict(
            bgcolor     = "rgba(0,0,0,0)",
            bordercolor = "#2a3a55",
            font        = {"color": "#8899bb", "size": 10}
        )
    )
    # Apply axis styles to every axis in the figure
    fig.update_xaxes(**PLOTLY_THEME["xaxis"])
    fig.update_yaxes(**PLOTLY_THEME["yaxis"])
    return fig


# =============================================================================
# DATA LOADING — cached so training only runs once per browser session
# =============================================================================

@st.cache_data(show_spinner=False)
def load_everything(epsilon=0.02):
    """
    Full data pipeline: download → train 3360 models → analyze → return.

    @st.cache_data means this runs ONCE on first page load (~2 minutes),
    then returns instantly from memory on every subsequent call within
    the same browser session. The cache is keyed on epsilon — changing
    epsilon would trigger a re-run (currently not exposed in the UI).

    Pipeline:
        load_data()           → downloads heart.csv → returns X, y, df
        build_rashomon_set()  → trains 3360 trees, finds L*, filters set
        run_all_analyses()    → patient disagreement, feature importance,
                                complexity stats, epsilon sweep

    Args:
        epsilon (float): tolerance below L* (default 0.02 = 2%)

    Returns:
        results  (dict): Rashomon Set output from train.py
        analyses (dict): four analysis outputs from analyze.py
        df       (DataFrame): full heart disease table for Dataset page
    """
    X, y, df  = load_data()
    results   = build_rashomon_set(X, y, epsilon=epsilon)
    analyses  = run_all_analyses(results)
    return results, analyses, df


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """
    Renders the persistent left sidebar on every page.

    The sidebar contains:
      - App title and dataset tagline
      - Navigation buttons (one per page)
        Clicking a button sets st.session_state["page"] and calls
        st.rerun() so the main area re-renders with the new page.
      - Live statistics once results are loaded into session_state
      - Footer attribution

    Why session_state?
    Streamlit re-runs the ENTIRE script on every user interaction.
    session_state is a persistent dict that survives re-runs within a
    session. We store the page name there so the sidebar and main()
    always agree on what to show.
    """
    with st.sidebar:

        # App title block
        st.markdown("""
        <div style="padding:8px 0 24px 0; border-bottom:1px solid #2a3a55; margin-bottom:20px;">
            <div style="font-family:'DM Serif Display',serif; font-size:22px;
                        color:#e8edf5; line-height:1.2;">
                Rashomon Set<br>Visualizer
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:10px;
                        color:#00d4aa; margin-top:6px; letter-spacing:0.1em;
                        text-transform:uppercase;">
                Heart Disease · UCI Dataset
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation buttons
        # Each tuple: (emoji, page_name, description)
        # page_name must exactly match the keys in main()'s if/elif chain
        pages = [
            ("🏠", "Home",           "Introduction & visitor guide"),
            ("🗄️", "Dataset",        "The heart disease data"),
            ("🔬", "Build the Set",  "Explore the model landscape"),
            ("👤", "Patient View",   "Individual prediction explorer"),
            ("⚖️", "Features",       "Feature importance across models"),
            ("📐", "Epsilon",        "How the set grows with tolerance"),
            ("📖", "About",          "Methodology & research context"),
        ]

        st.markdown('<span class="section-label">Navigation</span>',
                    unsafe_allow_html=True)

        for icon, name, desc in pages:
            # Active page button uses type="primary" (highlighted)
            active   = st.session_state.get("page", "Home") == name
            btn_type = "primary" if active else "secondary"

            if st.button(f"{icon}  {name}", key=f"nav_{name}",
                         use_container_width=True, type=btn_type):
                st.session_state["page"] = name
                st.rerun()   # force immediate re-render with the new page

        # Live stats — only shown once data has loaded
        # main() stores results in session_state after load_everything() runs
        st.markdown("""
        <div style="margin-top:28px; padding-top:20px; border-top:1px solid #2a3a55;">
            <span class="section-label">Live Stats</span>
        </div>
        """, unsafe_allow_html=True)

        if "results" in st.session_state:
            r   = st.session_state["results"]
            pct = r['rashomon_size'] / r['total_trained'] * 100
            st.markdown(f"""
            <div class="metric-box" style="margin-bottom:8px;">
                <span class="metric-value">{r['rashomon_size']}</span>
                <span class="metric-label">In Rashomon Set</span>
            </div>
            <div class="metric-box" style="margin-bottom:8px;">
                <span class="metric-value" style="color:#ffd166; font-size:22px;">{pct:.1f}%</span>
                <span class="metric-label">Of All Models</span>
            </div>
            <div class="metric-box">
                <span class="metric-value" style="color:#ff6b6b; font-size:22px;">{r['reference_accuracy']:.4f}</span>
                <span class="metric-label">Best Accuracy L*</span>
            </div>
            """, unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="margin-top:24px; padding-top:16px; border-top:1px solid #1e2a3d;">
            <div style="font-family:'Inter',sans-serif; font-size:10px;
                        color:#2a3a55; line-height:1.7;">
                Built to explore Prof. Lesia Semenova's<br>
                research on model multiplicity.<br><br>
                UCI Cleveland Heart Disease Dataset<br>
                297 patients · 13 features
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SCREEN 1 — HOME
# =============================================================================

def render_home(results, analyses):
    """
    The landing page. Five sections:

    1. Hero — Large serif title with one-sentence project summary.
    2. Five live metrics — models trained, set size, %, L*, patient count.
    3. Two-column body:
       Left:  Full written explanation of the Rashomon Effect story + three
              callouts (definition, problem, opportunity).
       Right: Overlapping accuracy histogram + 2×2 key numbers grid.
    4. Six-step visitor guide — numbered steps explaining what to do in order.
    5. Three key findings — color-coded cards summarizing the main results.

    Args:
        results  (dict): Rashomon Set output (all_models, rashomon_size, etc.)
        analyses (dict): four analyses (patient_disagreement, complexity, etc.)
    """

    # Hero: monospace tagline + large serif title + intro paragraph
    st.markdown("""
    <div style="padding:8px 0 32px 0; border-bottom:1px solid #1e2a3d; margin-bottom:36px;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:11px;
                    color:#00d4aa; text-transform:uppercase; letter-spacing:0.15em;
                    margin-bottom:12px;">
            Interactive Research Tool · UCI Heart Disease · Machine Learning
        </div>
        <div style="font-family:'DM Serif Display',serif; font-size:52px;
                    color:#e8edf5; line-height:1.05; margin-bottom:16px;">
            The Rashomon Effect<br><em>in Machine Learning</em>
        </div>
        <div style="font-size:16px; color:#8899bb; max-width:680px; line-height:1.8;">
            When you train a machine learning model, you don't get
            <em>one</em> best answer. You get hundreds of equally valid ones —
            and they can disagree on individual patients.
            This is the <strong style="color:#e8edf5;">Rashomon Effect</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Five live metrics — values come from results (not hardcoded)
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (f"{results['total_trained']:,}",                                "Models Trained",      "#00d4aa"),
        (f"{results['rashomon_size']}",                                   "In Rashomon Set",     "#00d4aa"),
        (f"{results['rashomon_size']/results['total_trained']*100:.1f}%", "Qualified at ε=0.02", "#ffd166"),
        (f"{results['reference_accuracy']:.4f}",                         "Best Accuracy L*",    "#ff6b6b"),
        ("297",                                                            "Patients · 13 Feats", "#a29bfe"),
    ]
    for col, (val, label, color) in zip([c1, c2, c3, c4, c5], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <span class="metric-value" style="color:{color}; font-size:24px;">{val}</span>
                <span class="metric-label">{label}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column layout: story text (left) + histogram and numbers (right)
    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<span class="section-label">The Story Behind the Name</span>',
                    unsafe_allow_html=True)

        # Card: Kurosawa film reference + ML parallel
        st.markdown("""
        <div class="card">
            <p style="font-family:'DM Serif Display',serif; font-size:21px;
                      color:#e8edf5; line-height:1.5; margin-bottom:16px;">
                In 1950, Akira Kurosawa released the film <em>Rashomon</em>.
                A samurai is found dead in a forest. Four witnesses describe what
                happened — and every account is different, yet equally believable.
            </p>
            <p style="font-size:14px; color:#8899bb; line-height:1.8; margin-bottom:14px;">
                No single account is <em>the truth</em>. They are all valid
                explanations of the same reality, shaped by the observer's
                perspective and what they chose to notice.
            </p>
            <p style="font-size:14px; color:#8899bb; line-height:1.8;">
                Machine learning has exactly this problem. Train the same
                algorithm on the same data with slightly different settings —
                and you get models that look at heart disease completely
                differently, yet are statistically indistinguishable
                in overall accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Three callouts: formal definition, the problem, the opportunity
        st.markdown("""
        <div class="callout">
            <strong>The formal definition:</strong><br><br>
            Given a dataset where the best model has accuracy L*, the
            Rashomon Set is all models within ε of that accuracy:<br><br>
            <code style="font-family:'JetBrains Mono'; font-size:14px; color:#00d4aa;">
                R(ε) = { all models f : accuracy(f) ≥ L* − ε }
            </code>
        </div>
        <div class="callout coral">
            <strong>Why it matters for healthcare:</strong><br><br>
            If Hospital A deploys model #7 and Hospital B deploys model #312
            — both equally accurate but disagreeing on specific patients —
            a patient's diagnosis depends on which hospital they visit.
            That is arbitrary model selection, not medicine.
        </div>
        <div class="callout gold">
            <strong>The opportunity:</strong><br><br>
            If many equally-good models exist, search the Rashomon Set for
            ones that are simpler (interpretable), fairer (unbiased), or
            more stable — without sacrificing accuracy.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<span class="section-label">Model Accuracy Distribution</span>',
                    unsafe_allow_html=True)

        # Overlapping histogram showing all 3360 models.
        # Grey bars = outside Rashomon Set.
        # Teal bars = inside Rashomon Set.
        # Red dashed line = ε threshold.
        in_accs  = [m['accuracy'] for m in results['all_models'] if     m['in_rashomon_set']]
        out_accs = [m['accuracy'] for m in results['all_models'] if not m['in_rashomon_set']]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=out_accs, name="Outside Set",
            nbinsx=30, marker_color="#2a3a55", opacity=0.8
        ))
        fig.add_trace(go.Histogram(
            x=in_accs, name="Rashomon Set",
            nbinsx=30, marker_color="#00d4aa", opacity=0.9
        ))
        fig.add_vline(
            x=results['threshold'],
            line_dash="dash", line_color="#ff6b6b", line_width=1.5,
            annotation_text=f"threshold={results['threshold']:.3f}",
            annotation_font_color="#ff6b6b", annotation_font_size=10
        )
        fig.update_layout(
            title=dict(text="All 3,360 Models by Accuracy",
                       font={"family": "DM Serif Display",
                             "color": "#e8edf5", "size": 15}),
            xaxis_title="CV Accuracy",
            yaxis_title="Count",
            barmode="overlay",
            showlegend=True
        )
        apply_theme(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)

        # 2×2 quick-reference grid of key numbers
        simplest = analyses['complexity_accuracy']['simplest_rashomon']
        unstable = [p for p in analyses['patient_disagreement']
                    if 0.30 <= p['disease_vote_pct'] <= 0.70]
        sweep    = analyses['epsilon_sweep']
        size_5   = next(s['rashomon_size'] for s in sweep
                        if abs(s['epsilon'] - 0.05) < 0.001)

        st.markdown(f"""
        <div class="card" style="margin-top:16px;">
            <span class="section-label">Key Numbers</span>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                <div style="text-align:center; padding:12px; background:#111827;
                            border-radius:8px; border:1px solid #1e2a3d;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:22px;
                                color:#00d4aa; font-weight:700;">{simplest['max_depth']}</div>
                    <div style="font-size:10px; color:#4a5a75; text-transform:uppercase;
                                letter-spacing:0.08em; margin-top:4px;">Min depth in set</div>
                </div>
                <div style="text-align:center; padding:12px; background:#111827;
                            border-radius:8px; border:1px solid #1e2a3d;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:22px;
                                color:#ff6b6b; font-weight:700;">{len(unstable)}</div>
                    <div style="font-size:10px; color:#4a5a75; text-transform:uppercase;
                                letter-spacing:0.08em; margin-top:4px;">Uncertain patients</div>
                </div>
                <div style="text-align:center; padding:12px; background:#111827;
                            border-radius:8px; border:1px solid #1e2a3d;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:22px;
                                color:#ffd166; font-weight:700;">{size_5:,}</div>
                    <div style="font-size:10px; color:#4a5a75; text-transform:uppercase;
                                letter-spacing:0.08em; margin-top:4px;">Models at ε=0.05</div>
                </div>
                <div style="text-align:center; padding:12px; background:#111827;
                            border-radius:8px; border:1px solid #1e2a3d;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:22px;
                                color:#a29bfe; font-weight:700;">13</div>
                    <div style="font-size:10px; color:#4a5a75; text-transform:uppercase;
                                letter-spacing:0.08em; margin-top:4px;">Input features</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Six-step visitor guide — tells a new visitor exactly what to do
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border-top:1px solid #1e2a3d; padding-top:32px; margin-top:8px;">
        <span class="section-label">Your Step-by-Step Guide to This App</span>
        <div style="font-family:'DM Serif Display',serif; font-size:26px;
                    color:#e8edf5; margin-bottom:20px;">
            Six things to explore, in order
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Each step: (icon, step_number, title, description, accent_color)
    steps = [
        ("🗄️", "01", "Explore the Dataset",
         "Start by meeting the data. The Dataset page shows all 297 patients, "
         "their 13 medical measurements, and whether they have heart disease. "
         "Understand what the model is learning from before you see what it "
         "does with that information.", "teal"),
        ("🔬", "02", "Build the Rashomon Set",
         "Go to Build the Set and drag the epsilon slider. Watch how the "
         "scatter plot changes — which models glow teal (qualify) and which "
         "go dark (excluded). Notice that depth-3 trees dominate while "
         "deeper trees mostly fail to qualify.", "teal"),
        ("👤", "03", "Find an Uncertain Patient",
         "Go to Patient View. The first dropdown entry is the most uncertain "
         "patient (marked ⚠️). The donut chart shows some models say DISEASE "
         "and others say NO DISEASE for the exact same person. Then switch "
         "to a stable patient and see unanimous agreement.", "coral"),
        ("⚖️", "04", "Examine Feature Importance",
         "Visit the Features page. The bar chart shows mean importance for "
         "each of the 13 features across all Rashomon Set models, with error "
         "bars showing disagreement. The CoV column reveals which features "
         "are stable predictors and which swing wildly.", "gold"),
        ("📐", "05", "Sweep Epsilon",
         "The Epsilon page shows how the set size changes as you relax the "
         "tolerance. At ε=0.005 only 248 models qualify. At ε=0.10 all 3360 "
         "qualify. This growth curve proves multiplicity is not a rare edge "
         "case — it is the default.", "gold"),
        ("📖", "06", "Read the Methodology",
         "The About page explains exactly how this was built, what decisions "
         "were made, what the limitations are, and how it connects to "
         "published research. Read this before discussing with a researcher.", "purple"),
    ]

    # Render two per row, alternating left/right
    col1, col2 = st.columns(2, gap="large")
    for i, (icon, num, title, desc, color) in enumerate(steps):
        border_c = {"teal":"#00a882","coral":"#cc5555","gold":"#aa8833","purple":"#7c78cc"}[color]
        num_c    = {"teal":"#00d4aa","coral":"#ff6b6b","gold":"#ffd166","purple":"#a29bfe"}[color]
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div style="display:flex; gap:16px; align-items:flex-start;
                        background:#1a2235; border:1px solid {border_c};
                        border-radius:10px; padding:20px; margin-bottom:12px;">
                <div style="font-size:28px; min-width:40px; line-height:1;">{icon}</div>
                <div>
                    <div style="font-family:'JetBrains Mono',monospace; font-size:10px;
                                color:{num_c}; text-transform:uppercase;
                                letter-spacing:0.12em; margin-bottom:4px;">Step {num}</div>
                    <div style="font-family:'DM Serif Display',serif; font-size:17px;
                                color:#e8edf5; margin-bottom:8px;">{title}</div>
                    <div style="font-size:13px; color:#8899bb; line-height:1.7;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Three key findings cards
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">The Three Core Findings</span>',
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown(f"""
        <div class="card-highlight">
            <span class="section-label">Finding 01 — Simplicity</span>
            <div style="font-family:'DM Serif Display',serif; font-size:20px;
                        color:#e8edf5; margin:8px 0; line-height:1.3;">
                A 3-Question Tree<br>Matches Deep Models
            </div>
            <p style="font-size:13px; color:#8899bb; line-height:1.7; margin:0;">
                Depth-{simplest['max_depth']} trees — asking just
                <strong style="color:#e8edf5;">{simplest['max_depth']} yes/no questions</strong>
                — qualify at a <strong style="color:#00d4aa;">100%</strong> rate.
                Deep trees qualify at only 3–9%.
                Complexity buys nothing here.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown(f"""
        <div class="card-coral">
            <span class="section-label" style="color:#ff6b6b;">Finding 02 — Instability</span>
            <div style="font-family:'DM Serif Display',serif; font-size:20px;
                        color:#e8edf5; margin:8px 0; line-height:1.3;">
                {len(unstable)} Patient{'s' if len(unstable)!=1 else ''} in the<br>
                Prediction Danger Zone
            </div>
            <p style="font-size:13px; color:#8899bb; line-height:1.7; margin:0;">
                <strong style="color:#e8edf5;">{len(unstable)} out of 297 patients</strong>
                receive conflicting predictions from equally-accurate models.
                Their diagnosis depends on which model was arbitrarily deployed.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with f3:
        st.markdown(f"""
        <div class="card-gold">
            <span class="section-label" style="color:#ffd166;">Finding 03 — Multiplicity</span>
            <div style="font-family:'DM Serif Display',serif; font-size:20px;
                        color:#e8edf5; margin:8px 0; line-height:1.3;">
                The Set Explodes<br>With Small ε Changes
            </div>
            <p style="font-size:13px; color:#8899bb; line-height:1.7; margin:0;">
                Relaxing ε from 0.02 to 0.05 grows the set from
                <strong style="color:#e8edf5;">{results['rashomon_size']}</strong>
                to <strong style="color:#ffd166;">{size_5:,}</strong> models.
                Model multiplicity is the default condition.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SCREEN 2 — DATASET
# =============================================================================

def render_dataset(df):
    """
    Shows the actual UCI Heart Disease dataset with full explanations.

    Sections:
      1. Four summary metrics (total patients, disease split, feature count).
      2. Two columns:
         Left:  Feature dictionary — every column name explained in plain
                English with accent-colored tags.
         Right: Scrollable st.dataframe() showing all 297 rows with formatted
                columns and a readable 'diagnosis' emoji column.
      3. Four overlapping histograms — one per key feature, split by diagnosis.
         Clean separation between groups = that feature is a strong predictor.

    Args:
        df (DataFrame): full heart.csv table including the target column
    """

    st.markdown("""
    <div class="page-title">The Heart Disease<br><em>Dataset</em></div>
    <div class="page-subtitle">
        UCI Cleveland Heart Disease dataset — 297 patients, 13 features,
        collected at the Cleveland Clinic Foundation. One of the most
        studied datasets in interpretable ML research.
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-box">
            <span class="metric-value">{len(df)}</span>
            <span class="metric-label">Total Patients</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-box">
            <span class="metric-value" style="color:#ff6b6b;">{int(df['target'].sum())}</span>
            <span class="metric-label">With Heart Disease</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-box">
            <span class="metric-value" style="color:#00d4aa;">{int((df['target']==0).sum())}</span>
            <span class="metric-label">No Heart Disease</span>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-box">
            <span class="metric-value" style="color:#ffd166;">13</span>
            <span class="metric-label">Input Features</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature dictionary (narrow) + data table (wide)
    col_guide, col_table = st.columns([1, 2], gap="large")

    with col_guide:
        st.markdown('<span class="section-label">Feature Dictionary</span>',
                    unsafe_allow_html=True)

        # (column_name, label, description, accent_color)
        # coral = clinically strongest predictors, gold = target label
        features_explained = [
            ("age",      "Age",                "Patient age in years",                                       "teal"),
            ("sex",      "Sex",                "1 = Male · 0 = Female",                                      "teal"),
            ("cp",       "Chest Pain Type",    "0=typical angina · 1=atypical · 2=non-anginal · 3=asymp",   "coral"),
            ("trestbps", "Resting BP",         "Resting blood pressure in mm Hg",                           "teal"),
            ("chol",     "Cholesterol",        "Serum cholesterol in mg/dl",                                 "teal"),
            ("fbs",      "Fasting Blood Sugar","1 if fasting blood sugar > 120 mg/dl · 0 otherwise",        "teal"),
            ("restecg",  "Resting ECG",        "0=normal · 1=ST-T wave abnormality · 2=LV hypertrophy",     "teal"),
            ("thalach",  "Max Heart Rate",     "Maximum heart rate achieved during exercise test",          "coral"),
            ("exang",    "Exercise Angina",    "Exercise-induced angina: 1=Yes · 0=No",                     "teal"),
            ("oldpeak",  "ST Depression",      "ST depression induced by exercise relative to rest (mm)",   "coral"),
            ("slope",    "ST Slope",           "Slope of peak exercise ST segment (1–3)",                   "teal"),
            ("ca",       "Major Vessels",      "Number of major vessels colored by fluoroscopy (0–3)",      "coral"),
            ("thal",     "Thalassemia",        "3=normal · 6=fixed defect · 7=reversible defect",           "coral"),
            ("target",   "TARGET (Label)",     "0 = No disease · 1 = Disease present (what we predict)",    "gold"),
        ]

        for fname, label, desc, color in features_explained:
            st.markdown(f"""
            <div style="padding:8px 0; border-bottom:1px solid #1e2a3d;">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:2px;">
                    <span class="tag tag-{color}">{fname}</span>
                    <span style="font-family:'DM Serif Display',serif;
                                 font-size:13px; color:#e8edf5;">{label}</span>
                </div>
                <div style="font-size:11px; color:#4a5a75; padding-left:4px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_table:
        st.markdown('<span class="section-label">Patient Data (scrollable)</span>',
                    unsafe_allow_html=True)

        # Build display dataframe:
        # - Add row number column
        # - Replace numeric target with emoji label
        # - Drop the original numeric target column
        display_df = df.copy()
        display_df.insert(0, '#', range(len(display_df)))
        display_df['diagnosis'] = display_df['target'].map(
            {0: '✅ No Disease', 1: '🔴 Disease'}
        )

        # st.dataframe renders an interactive scrollable table with
        # column_config for custom formatting per column
        st.dataframe(
            display_df.drop(columns=['target']),
            use_container_width=True,
            height=480,
            column_config={
                "#":         st.column_config.NumberColumn("#",       width="small"),
                "diagnosis": st.column_config.TextColumn("Diagnosis", width="medium"),
                "age":       st.column_config.NumberColumn("Age",     format="%d"),
                "chol":      st.column_config.NumberColumn("Chol",    format="%d"),
                "trestbps":  st.column_config.NumberColumn("BP",      format="%d"),
                "thalach":   st.column_config.NumberColumn("MaxHR",   format="%d"),
            },
            hide_index=True
        )
        st.markdown("""
        <div style="font-size:11px; color:#4a5a75; margin-top:8px;
                    font-family:'JetBrains Mono',monospace;">
            Scroll horizontally to see all 13 features.
        </div>
        """, unsafe_allow_html=True)

    # Feature distribution histograms — four key features
    # Each chart: two overlapping histograms (no-disease vs disease)
    # Clean separation = strong predictor
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Feature Distributions by Diagnosis for the 4 most important features</span>',
                unsafe_allow_html=True)

    key_features = ['thal', 'cp', 'ca', 'thalach']
    feat_labels  = {
        'thal': 'Thalassemia Type', 'cp': 'Chest Pain Type',
        'ca': 'Major Vessels (ca)', 'thalach': 'Max Heart Rate'
    }

    cols = st.columns(4)
    for i, feat in enumerate(key_features):
        with cols[i]:
            fig = go.Figure()
            # No-disease plotted first (behind), disease on top
            fig.add_trace(go.Histogram(
                x=df[df['target']==0][feat], name="No Disease",
                marker_color="#00d4aa", opacity=0.7, nbinsx=15
            ))
            fig.add_trace(go.Histogram(
                x=df[df['target']==1][feat], name="Disease",
                marker_color="#ff6b6b", opacity=0.7, nbinsx=15
            ))
            fig.update_layout(
                title=dict(text=feat_labels[feat],
                           font={"family": "DM Serif Display",
                                 "color": "#e8edf5", "size": 13}),
                barmode="overlay",
                showlegend=(i == 0),   # only first chart shows legend
                xaxis_title=feat, yaxis_title="Count"
            )
            apply_theme(fig, height=220)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="callout">
        <strong>Reading these charts:</strong> Where teal (no disease) and
        coral (disease) bars separate cleanly, that feature is a strong
        predictor. Notice how <strong>thal</strong> and <strong>cp</strong>
        show very clean separation — this is why they dominate our
        decision trees and appear in nearly every Rashomon Set model.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SCREEN 3 — BUILD THE SET
# =============================================================================

def render_build(results, analyses):
    """
    Interactive scatter plot of all 3,360 trained models.

    X-axis: max_depth (complexity: 2=simple, 8=complex)
    Y-axis: 5-Fold CV accuracy
    Teal dots: models inside the Rashomon Set at the chosen ε
    Dark dots: models outside the set
    Red dashed: ε threshold line
    Gold dotted: L* (best accuracy) line

    The epsilon slider recomputes threshold and qualifying count live.
    Horizontal jitter on X prevents dots at the same depth from stacking.

    Below: depth breakdown cards (7 columns) showing how many models
    at each depth level qualify for the Rashomon Set.

    Args:
        results  (dict): contains all_models list and epsilon
        analyses (dict): contains complexity depth_summary
    """

    st.markdown("""
    <div class="page-title">Build the<br><em>Rashomon Set</em></div>
    <div class="page-subtitle">
        Every dot is one trained model. Drag the epsilon slider to move
        the threshold line and watch which models qualify.
        Teal dots are in the Rashomon Set. Dark dots are excluded.
    </div>
    """, unsafe_allow_html=True)

    # Epsilon slider — st.slider returns the current value on every rerun
    col_slider, col_info = st.columns([2, 1])
    with col_slider:
        epsilon_display = st.slider(
            "Epsilon (ε) — tolerance below best accuracy L*",
            min_value=0.005, max_value=0.10,
            value=float(results['epsilon']),
            step=0.005, format="%.3f",
            help="Models with accuracy ≥ L* − ε qualify for the Rashomon Set"
        )

    L_star     = results['reference_accuracy']
    threshold  = L_star - epsilon_display
    # Live count of models above the slider-controlled threshold
    qualifying = sum(1 for m in results['all_models']
                     if m['accuracy'] >= threshold)

    with col_info:
        st.markdown(f"""
        <div class="card" style="margin-top:8px; padding:16px;">
            <div style="display:flex; gap:16px; text-align:center; flex-wrap:wrap;">
                <div>
                    <span class="metric-value" style="font-size:18px;">{L_star:.4f}</span>
                    <span class="metric-label">L* (best)</span>
                </div>
                <div>
                    <span class="metric-value" style="font-size:18px; color:#ff6b6b;">{threshold:.4f}</span>
                    <span class="metric-label">Threshold</span>
                </div>
                <div>
                    <span class="metric-value" style="font-size:18px; color:#ffd166;">{qualifying}</span>
                    <span class="metric-label">Qualify</span>
                </div>
                <div>
                    <span class="metric-value" style="font-size:18px; color:#a29bfe;">
                        {qualifying/results['total_trained']*100:.1f}%
                    </span>
                    <span class="metric-label">Fraction</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Build scatter plot
    all_models = results['all_models']
    in_set  = [m for m in all_models if m['accuracy'] >= threshold]
    out_set = [m for m in all_models if m['accuracy'] <  threshold]

    # Random horizontal jitter so dots at the same depth separate visually.
    # Fixed seed (42) ensures the jitter is the same on every re-render.
    rng = np.random.default_rng(42)
    def jitter(arr, scale=0.15):
        return [v + rng.uniform(-scale, scale) for v in arr]

    fig = go.Figure()

    # Out-of-set: very dark, barely visible dots
    if out_set:
        fig.add_trace(go.Scatter(
            x=jitter([m['max_depth'] for m in out_set]),
            y=[m['accuracy']  for m in out_set],
            mode="markers", name="Outside Rashomon Set",
            marker=dict(color="#1e2a3d", size=5,
                        line=dict(color="#2a3a55", width=0.5)),
            hovertemplate="Depth: %{customdata[0]}<br>Acc: %{y:.4f}<extra>Outside</extra>",
            customdata=[[m['max_depth']] for m in out_set]
        ))

    # In-set: teal gradient colored by accuracy value, larger markers
    if in_set:
        fig.add_trace(go.Scatter(
            x=jitter([m['max_depth'] for m in in_set]),
            y=[m['accuracy']  for m in in_set],
            mode="markers", name="Rashomon Set",
            marker=dict(
                color=[m['accuracy'] for m in in_set],  # gradient by accuracy
                colorscale=[[0, "#00a882"], [1, "#00d4aa"]],
                size=7, line=dict(color="#006655", width=0.5),
                showscale=True,
                colorbar=dict(
                    title=dict(text="Accuracy",
                               font=dict(color="#8899bb", size=10)),
                    thickness=12, len=0.6,
                    tickfont=dict(color="#8899bb", size=9)
                )
            ),
            hovertemplate="Depth: %{customdata[0]}<br>Acc: %{y:.4f}<br>"
                          "Min split: %{customdata[1]}<extra>In Set</extra>",
            customdata=[[m['max_depth'], m['min_samples_split']] for m in in_set]
        ))

    # Horizontal reference lines
    fig.add_hline(y=threshold, line_dash="dash", line_color="#ff6b6b",
                  line_width=1.5,
                  annotation_text=f"ε threshold = {threshold:.4f}",
                  annotation_font_color="#ff6b6b", annotation_font_size=10,
                  annotation_position="right")
    fig.add_hline(y=L_star, line_dash="dot", line_color="#ffd166",
                  line_width=1,
                  annotation_text=f"L* = {L_star:.4f}",
                  annotation_font_color="#ffd166", annotation_font_size=10,
                  annotation_position="right")

    fig.update_layout(
        title=dict(
            text="Model Landscape: Complexity vs Accuracy (3,360 models)",
            font={"family": "DM Serif Display", "color": "#e8edf5", "size": 18}
        ),
        xaxis=dict(title="Max Depth (complexity)",
                   tickvals=[2,3,4,5,6,7,8],
                   ticktext=["2\n(simple)","3","4","5","6","7","8\n(complex)"]),
        yaxis_title="5-Fold CV Accuracy", showlegend=True
    )
    apply_theme(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Depth breakdown cards — one per depth level (2 through 8)
    st.markdown('<span class="section-label">Rashomon Set Breakdown by Depth</span>',
                unsafe_allow_html=True)

    depth_summary = analyses['complexity_accuracy']['depth_summary']
    cols = st.columns(7)
    for i, depth in enumerate([2,3,4,5,6,7,8]):
        stats = depth_summary[depth]
        in_s  = stats['in_rashomon_set']
        # Teal number if any models at this depth qualify, dark grey if none
        color = "#00d4aa" if in_s > 0 else "#2a3a55"
        with cols[i]:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:16px 8px;">
                <div style="font-family:'DM Serif Display',serif;
                            font-size:26px; color:{color};">{in_s}</div>
                <div style="font-family:'JetBrains Mono',monospace;
                            font-size:10px; color:#8899bb; margin:4px 0;">
                    Depth {depth}</div>
                <div style="font-family:'JetBrains Mono',monospace;
                            font-size:10px; color:#4a5a75;">
                    {stats['pct_in_set']*100:.0f}% of {stats['total_models']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout" style="margin-top:16px;">
        <strong>What you should notice:</strong> Depth-3 trees qualify at 100%
        while depth-4 through depth-8 trees qualify at only 3–9%. A 3-question
        flowchart — simple enough to memorize — performs just as well as a
        256-leaf black box. This is the interpretable ML argument made visible.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SCREEN 4 — PATIENT EXPLORER
# =============================================================================

def render_patient(results, analyses, df):
    """
    Individual patient prediction explorer.

    Dropdown: all 297 patients sorted with most uncertain (closest to 50%
    model split) first. "Uncertain" means disease_vote_pct is near 0.50.

    Left column:
      - Donut chart: disease vs no-disease vote percentage across 633 models.
        Center annotation shows dominant prediction + percentage.
      - Two metric boxes: raw vote counts.

    Right column:
      - 13 feature bars: name | normalized bar | value + unit.
        Bar width = (value - min) / (max - min) normalized to 0–100%.

    Below: Horizontal bar chart of ALL 297 patients sorted by disease vote
    fraction. Color gradient from teal (stable no-disease) through gold
    (uncertain) to coral (stable disease). Vertical dashed lines at 30%
    and 70% mark the uncertain zone.

    Args:
        results  (dict): Rashomon Set output
        analyses (dict): contains 'patient_disagreement' analysis
        df       (DataFrame): full heart.csv
    """

    st.markdown("""
    <div class="page-title">Patient Prediction<br><em>Explorer</em></div>
    <div class="page-subtitle">
        Select any patient and see what fraction of the 633 Rashomon Set
        models predict disease. Near 50% = genuine uncertainty —
        the diagnosis flips depending on which model was deployed.
    </div>
    """, unsafe_allow_html=True)

    patient_data = analyses['patient_disagreement']

    # Sort so most uncertain (smallest |vote - 0.5|) appears first in dropdown
    unstable_first = sorted(patient_data,
                            key=lambda p: abs(p['disease_vote_pct'] - 0.5))

    def patient_label(p):
        """Build the dropdown label for one patient."""
        marker   = "⚠️ UNCERTAIN" if 0.30 <= p['disease_vote_pct'] <= 0.70 else "✅ STABLE"
        true_str = "DISEASE" if p['true_label'] == 1 else "NO DISEASE"
        return (f"Patient #{p['patient_idx']:03d}  [{true_str}]  "
                f"{p['disease_vote_pct']*100:.0f}% disease votes  {marker}")

    options        = [patient_label(p) for p in unstable_first]
    selected_label = st.selectbox(
        "Select a patient:",
        options, index=0,
        help="Sorted with the most uncertain patient first (⚠️)"
    )
    patient = unstable_first[options.index(selected_label)]

    # Extract values for this patient
    disease_pct    = patient['disease_vote_pct']
    no_disease_pct = patient['no_disease_vote_pct']
    true_label     = patient['true_label']
    is_uncertain   = 0.30 <= disease_pct <= 0.70

    # Choose badge CSS classes based on diagnosis and uncertainty
    badge_class = "badge-disease"     if true_label == 1  else "badge-nodisease"
    badge_text  = "TRUE: DISEASE"     if true_label == 1  else "TRUE: NO DISEASE"
    zone_class  = "badge-uncertain"   if is_uncertain      else badge_class
    zone_text   = "⚠️ UNCERTAIN ZONE" if is_uncertain      else "✅ STABLE PREDICTION"

    # Patient header card
    st.markdown(f"""
    <div class="card" style="margin-bottom:24px;">
        <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
            <div style="font-family:'DM Serif Display',serif; font-size:32px;
                        color:#e8edf5;">Patient #{patient['patient_idx']:03d}</div>
            <span class="patient-badge {badge_class}">{badge_text}</span>
            <span class="patient-badge {zone_class}">{zone_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_donut, col_feats = st.columns([1, 1], gap="large")

    with col_donut:
        # Donut chart: disease (coral) vs no-disease (teal) vote split
        # Hole center shows dominant prediction + percentage
        fig_donut = go.Figure(go.Pie(
            values=[disease_pct, no_disease_pct],
            labels=["Disease", "No Disease"],
            hole=0.65,
            marker=dict(
                colors=["#ff6b6b", "#00d4aa"],
                line=dict(color="#0a0e1a", width=3)   # dark gap between slices
            ),
            textinfo="percent",
            textfont=dict(family="JetBrains Mono", size=12, color="#e8edf5"),
            hovertemplate="%{label}: %{value:.1%}<extra></extra>"
        ))

        # Center annotation overlaid on the donut hole
        dominant_pct   = max(disease_pct, no_disease_pct)
        dominant_text  = "DISEASE"    if disease_pct >= 0.5 else "NO DISEASE"
        dominant_color = "#ff6b6b"    if disease_pct >= 0.5 else "#00d4aa"

        fig_donut.add_annotation(
            text=(f"<b>{dominant_pct:.0%}</b><br>"
                  f"<span style='font-size:10px;color:#8899bb'>{dominant_text}</span>"),
            x=0.5, y=0.5,
            font=dict(family="JetBrains Mono", size=20, color=dominant_color),
            showarrow=False, align="center"
        )
        fig_donut.update_layout(
            title=dict(
                text=f"Rashomon Set Votes ({patient['total_models']} models)",
                font={"family": "DM Serif Display", "color": "#e8edf5", "size": 15}
            ),
            showlegend=True,
            legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1,
                        font=dict(color="#8899bb", size=10))
        )
        apply_theme(fig_donut, height=360)
        st.plotly_chart(fig_donut, use_container_width=True)

        # Raw vote count boxes below the donut
        st.markdown(f"""
        <div style="display:flex; gap:12px; margin-top:-8px;">
            <div class="metric-box" style="flex:1;">
                <span class="metric-value" style="color:#ff6b6b; font-size:22px;">
                    {patient['disease_votes']}</span>
                <span class="metric-label">Disease Votes</span>
            </div>
            <div class="metric-box" style="flex:1;">
                <span class="metric-value" style="font-size:22px;">
                    {patient['no_disease_votes']}</span>
                <span class="metric-label">No Disease Votes</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_feats:
        # Feature bars: name | normalized bar | value + unit
        # Bar width = (value - min) / (max - min) clamped to [0, 100]%
        st.markdown('<span class="section-label">Patient Measurements</span>',
                    unsafe_allow_html=True)

        # (label, unit, min_value, max_value) for normalization
        feature_info = {
            'age':      ('Age',              'years', 0,   80),
            'sex':      ('Sex',              '1=M',   0,   1),
            'cp':       ('Chest Pain',       '0–3',   0,   4),
            'trestbps': ('Blood Pressure',   'mmHg',  80,  200),
            'chol':     ('Cholesterol',      'mg/dl', 100, 600),
            'fbs':      ('Fasting BG',       'bool',  0,   1),
            'restecg':  ('ECG Results',      '0–2',   0,   2),
            'thalach':  ('Max Heart Rate',   'bpm',   60,  220),
            'exang':    ('Ex. Angina',       '1=yes', 0,   1),
            'oldpeak':  ('ST Depression',    'mm',    0,   7),
            'slope':    ('ST Slope',         '1–3',   0,   3),
            'ca':       ('Major Vessels',    '0–3',   0,   4),
            'thal':     ('Thalassemia',      '3/6/7', 0,   7),
        }

        for fname, value in patient['features'].items():
            label, unit, min_v, max_v = feature_info.get(fname, (fname, '', 0, 1))
            normalized = min(1.0, max(0.0,
                (float(value) - min_v) / (max_v - min_v)
                if (max_v - min_v) > 0 else 0
            ))
            bar_pct = int(normalized * 100)
            st.markdown(f"""
            <div class="feature-row">
                <span class="feature-name">{fname}</span>
                <div class="feature-bar-bg">
                    <div class="feature-bar-fill" style="width:{bar_pct}%"></div>
                </div>
                <span class="feature-val">{value:.1f}
                    <span style="color:#4a5a75; font-size:10px;">{unit}</span>
                </span>
            </div>
            """, unsafe_allow_html=True)

    # All-patients overview: horizontal bar chart, one bar per patient
    # Sorted by disease vote fraction, color-coded by zone
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">All 297 Patients — Prediction Stability</span>',
                unsafe_allow_html=True)

    sorted_patients = sorted(patient_data, key=lambda p: p['disease_vote_pct'])

    fig_bar = go.Figure(go.Bar(
        x=[p['disease_vote_pct'] for p in sorted_patients],
        y=[f"P{p['patient_idx']:03d}" for p in sorted_patients],
        orientation='h',
        marker=dict(
            color=[p['disease_vote_pct'] for p in sorted_patients],
            # Three-zone gradient: teal (no disease) → gold (uncertain) → coral (disease)
            colorscale=[
                [0.0, "#00d4aa"], [0.3, "#00d4aa"],
                [0.45, "#ffd166"], [0.55, "#ffd166"],
                [0.7,  "#ff6b6b"], [1.0, "#ff6b6b"]
            ],
            showscale=True,
            colorbar=dict(
                title=dict(text="Disease Vote %",
                           font=dict(color="#8899bb", size=10)),
                tickformat=".0%", thickness=12,
                tickfont=dict(color="#8899bb", size=9)
            )
        ),
        hovertemplate="Patient %{y}<br>Disease votes: %{x:.1%}<extra></extra>"
    ))

    # Vertical lines marking the 30–70% uncertain zone boundaries
    fig_bar.add_vline(x=0.3, line_dash="dash", line_color="#ffd166",
                      line_width=1, opacity=0.5)
    fig_bar.add_vline(x=0.7, line_dash="dash", line_color="#ffd166",
                      line_width=1, opacity=0.5)
    fig_bar.add_vline(x=0.5, line_dash="dot",  line_color="#ff6b6b",
                      line_width=1, opacity=0.5)

    fig_bar.update_layout(
        title=dict(
            text="Disease Vote Fraction per Patient — Yellow zone = uncertain",
            font={"family": "DM Serif Display", "color": "#e8edf5", "size": 15}
        ),
        xaxis=dict(title="Fraction of Models Predicting Disease", tickformat=".0%"),
        yaxis=dict(showticklabels=False, title="Patients (sorted)"),
        showlegend=False
    )
    apply_theme(fig_bar, height=360)
    st.plotly_chart(fig_bar, use_container_width=True)


# =============================================================================
# SCREEN 5 — FEATURE IMPORTANCE
# =============================================================================

def render_features(results, analyses):
    """
    Feature importance instability visualization.

    Main chart — two overlaid horizontal bar traces:
      1. Range band (min to max importance across all 633 models)
         Shows the full spread of opinions about each feature.
         Built as a stacked bar: base=min, width=max-min.
      2. Mean bar with gold error bars (±std)
         Shows average reliance on each feature and its variability.

    Table — all statistics per feature including CoV (Coefficient of
    Variation = std/mean). Color-coded: red>0.5, yellow>0.2, green≤0.2.

    Key insight: feature importance from one model is not the truth —
    it is one sample from a distribution across the Rashomon Set.

    Args:
        results  (dict): Rashomon Set output
        analyses (dict): contains 'feature_importance' analysis
    """

    st.markdown("""
    <div class="page-title">Feature Importance<br><em>Wars</em></div>
    <div class="page-subtitle">
        Equally-accurate models have completely different internal logic.
        Bars show mean importance across all 633 Rashomon Set models.
        Error bars show how much models disagree about each feature.
    </div>
    """, unsafe_allow_html=True)

    fi     = analyses['feature_importance']
    agg    = fi['aggregated']      # dict: feature → {mean, std, min, max, coeff_variation}
    ranked = fi['ranked_features'] # features sorted by mean importance (descending)

    means  = [agg[f]['mean'] for f in ranked]
    stds   = [agg[f]['std']  for f in ranked]
    maxes  = [agg[f]['max']  for f in ranked]
    mins   = [agg[f]['min']  for f in ranked]

    fig_fi = go.Figure()

    # Trace 1: Range band — base=min, width=max-min
    # Creates a background band showing the full range of importance per feature
    fig_fi.add_trace(go.Bar(
        y=ranked,
        x=[agg[f]['max'] - agg[f]['min'] for f in ranked],  # width = range
        base=[agg[f]['min'] for f in ranked],                # left edge = min
        orientation='h', name="Min–Max Range",
        marker=dict(color="rgba(0,212,170,0.10)",
                    line=dict(color="rgba(0,212,170,0.18)", width=1)),
        hovertemplate="%{y}: range %{base:.3f}–%{x:.3f}<extra></extra>"
    ))

    # Trace 2: Mean bar with gold ±std error bars
    # Color gradient by importance magnitude (dark → bright teal)
    fig_fi.add_trace(go.Bar(
        y=ranked, x=means, orientation='h', name="Mean Importance",
        marker=dict(
            color=means,
            colorscale=[[0,"#006655"],[0.5,"#00a882"],[1,"#00d4aa"]],
            line=dict(color="#004433", width=0.5)
        ),
        error_x=dict(type='data', array=stds, color="#ffd166",
                     thickness=1.5, width=4),
        hovertemplate=(
            "<b>%{y}</b><br>Mean: %{x:.4f}<br>"
            "Std: %{customdata[0]:.4f}<br>"
            "Range: %{customdata[1]:.3f}–%{customdata[2]:.3f}<extra></extra>"
        ),
        customdata=[[stds[i], mins[i], maxes[i]] for i in range(len(ranked))]
    ))

    fig_fi.update_layout(
        title=dict(text="Feature Importance: Mean ± Std Across 633 Models",
                   font={"family": "DM Serif Display", "color": "#e8edf5", "size": 16}),
        xaxis_title="Gini-based Importance Score",
        barmode="overlay",    # range band behind mean bar
        showlegend=True
    )
    apply_theme(fig_fi, height=460)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Statistics table + explanatory callouts
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<span class="section-label">Full Statistics</span>',
                    unsafe_allow_html=True)

        # Build HTML table row by row
        # CoV color-coding: red > 0.5, yellow > 0.2, teal ≤ 0.2
        rows = ""
        for f in ranked:
            stats    = agg[f]
            coeff    = stats['coeff_variation']
            cv_color = ("#ff6b6b" if coeff > 0.5
                        else "#ffd166" if coeff > 0.2 else "#00d4aa")
            rows += f"""
            <tr style="border-bottom:1px solid #1e2a3d;">
                <td style="padding:8px 12px; font-family:'JetBrains Mono',monospace;
                           font-size:12px; color:#8899bb;">{f}</td>
                <td style="padding:8px 12px; font-family:'JetBrains Mono',monospace;
                           font-size:12px; color:#00d4aa; text-align:right;">{stats['mean']:.4f}</td>
                <td style="padding:8px 12px; font-family:'JetBrains Mono',monospace;
                           font-size:12px; color:#ffd166; text-align:right;">{stats['std']:.4f}</td>
                <td style="padding:8px 12px; font-family:'JetBrains Mono',monospace;
                           font-size:12px; color:#4a5a75; text-align:right;">{stats['min']:.3f}</td>
                <td style="padding:8px 12px; font-family:'JetBrains Mono',monospace;
                           font-size:12px; color:#4a5a75; text-align:right;">{stats['max']:.3f}</td>
                <td style="padding:8px 12px; font-family:'JetBrains Mono',monospace;
                           font-size:12px; color:{cv_color}; text-align:right;">{coeff:.2f}</td>
            </tr>"""

        st.markdown(f"""
        <div class="card" style="padding:0; overflow:hidden;">
          <table style="width:100%; border-collapse:collapse;">
            <thead>
              <tr style="background:#111827; border-bottom:1px solid #2a3a55;">
                <th style="padding:10px 12px; text-align:left; font-family:'JetBrains Mono';
                           font-size:10px; color:#4a5a75; text-transform:uppercase;
                           letter-spacing:.08em;">Feature</th>
                <th style="padding:10px 12px; text-align:right; font-family:'JetBrains Mono';
                           font-size:10px; color:#00d4aa; text-transform:uppercase;">Mean</th>
                <th style="padding:10px 12px; text-align:right; font-family:'JetBrains Mono';
                           font-size:10px; color:#ffd166; text-transform:uppercase;">Std</th>
                <th style="padding:10px 12px; text-align:right; font-family:'JetBrains Mono';
                           font-size:10px; color:#4a5a75; text-transform:uppercase;">Min</th>
                <th style="padding:10px 12px; text-align:right; font-family:'JetBrains Mono';
                           font-size:10px; color:#4a5a75; text-transform:uppercase;">Max</th>
                <th style="padding:10px 12px; text-align:right; font-family:'JetBrains Mono';
                           font-size:10px; color:#8899bb; text-transform:uppercase;">CoV</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:10px;
                    color:#4a5a75; margin-top:8px;">
            CoV = std/mean.
            <span style="color:#ff6b6b;">Red &gt;0.5</span> |
            <span style="color:#ffd166;">Yellow &gt;0.2</span> |
            <span style="color:#00d4aa;">Green ≤0.2</span>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="callout" style="margin-top:0;">
            <strong>What CoV means:</strong><br><br>
            Coefficient of Variation = std ÷ mean. A high CoV means a
            feature's importance swings wildly across equally-accurate
            models — sometimes dominant, sometimes ignored entirely.<br><br>
            <strong>thal</strong>: low CoV — stable dominance.<br>
            <strong>age, thalach</strong>: high CoV — contested.
        </div>
        <div class="callout coral">
            <strong>Why this matters:</strong><br><br>
            Ask "what is the most important warning sign?" and different
            equally-valid models give different answers. There is no single
            true feature importance — only a distribution across the Rashomon Set.
        </div>
        <div class="callout gold">
            <strong>The error bars:</strong><br><br>
            Gold error bars show ± one standard deviation. Wide bars = high
            disagreement. Narrow bars = stable agreement. Feature importance
            instability made visible.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# SCREEN 6 — EPSILON EXPLORER
# =============================================================================

def render_epsilon(results, analyses):
    """
    Shows how the Rashomon Set size changes across epsilon values.

    Two charts:
      1. Area line chart: X=epsilon, Y=qualifying model count.
         Red dashed line marks our chosen epsilon with an annotation pin.
      2. Percentage chart: same data as % of all 3360 models.

    Three summary cards at the bottom (tight / our choice / loose).

    No retraining needed: all 3360 models' accuracies are already in
    results['all_models']. The epsilon_sweep analysis in analyze.py
    simply re-filtered at 20 different epsilon values.

    Key insight: epsilon is an arbitrary human choice with enormous
    consequences. The rapid growth curve proves multiplicity is
    the default, not an edge case.

    Args:
        results  (dict): Rashomon Set output
        analyses (dict): contains 'epsilon_sweep' analysis
    """

    st.markdown("""
    <div class="page-title">The Epsilon<br><em>Explorer</em></div>
    <div class="page-subtitle">
        Epsilon is the tolerance knob. Tighten it and only the very best
        models qualify. Relax it slightly and the set explodes. This proves
        model multiplicity is the default condition, not a rare edge case.
    </div>
    """, unsafe_allow_html=True)

    sweep     = analyses['epsilon_sweep']
    eps_vals  = [s['epsilon']       for s in sweep]
    set_sizes = [s['rashomon_size'] for s in sweep]
    pct_vals  = [s['pct_of_total']  for s in sweep]

    # Area line chart (absolute set size)
    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=eps_vals, y=set_sizes,
        mode="lines+markers", name="Rashomon Set Size",
        line=dict(color="#00d4aa", width=2.5),
        marker=dict(color="#00d4aa", size=7,
                    line=dict(color="#004433", width=1)),
        fill='tozeroy',                          # shade area under curve
        fillcolor="rgba(0,212,170,0.07)",
        hovertemplate="ε=%{x:.3f}<br>%{y} models (%{customdata:.1%})<extra></extra>",
        customdata=pct_vals
    ))

    # Mark chosen epsilon with vertical line + annotation callout
    chosen_eps  = results['epsilon']
    chosen_size = next(s['rashomon_size'] for s in sweep
                       if abs(s['epsilon'] - chosen_eps) < 0.001)

    fig_sweep.add_vline(x=chosen_eps, line_dash="dash",
                        line_color="#ff6b6b", line_width=1.5,
                        annotation_text=f"Our ε={chosen_eps}",
                        annotation_font_color="#ff6b6b", annotation_font_size=11,
                        annotation_position="top right")
    fig_sweep.add_annotation(
        x=chosen_eps, y=chosen_size,
        text=f"{chosen_size} models",
        font=dict(family="JetBrains Mono", color="#ff6b6b", size=11),
        showarrow=True, arrowcolor="#ff6b6b", arrowsize=0.8,
        ax=40, ay=-40,
        bgcolor="#1a2235", bordercolor="#ff6b6b", borderwidth=1, borderpad=6
    )
    fig_sweep.update_layout(
        title=dict(text="Rashomon Set Size vs Epsilon",
                   font={"family": "DM Serif Display",
                         "color": "#e8edf5", "size": 18}),
        xaxis_title="Epsilon (ε)", yaxis_title="Qualifying Models",
        showlegend=False
    )
    apply_theme(fig_sweep, height=400)
    st.plotly_chart(fig_sweep, use_container_width=True)

    # Percentage line chart
    fig_pct = go.Figure(go.Scatter(
        x=eps_vals, y=[p*100 for p in pct_vals],
        mode="lines+markers",
        line=dict(color="#ffd166", width=2),
        marker=dict(color="#ffd166", size=6),
        fill='tozeroy', fillcolor="rgba(255,209,102,0.05)",
        hovertemplate="ε=%{x:.3f}<br>%{y:.1f}% qualify<extra></extra>"
    ))
    fig_pct.add_hline(y=100, line_dash="dot", line_color="#2a3a55", line_width=1)
    fig_pct.add_vline(x=chosen_eps, line_dash="dash",
                      line_color="#ff6b6b", line_width=1.5)
    fig_pct.update_layout(
        title=dict(text="% of All Models Qualifying vs Epsilon",
                   font={"family": "DM Serif Display",
                         "color": "#e8edf5", "size": 15}),
        xaxis_title="Epsilon (ε)", yaxis_title="% Qualifying",
        showlegend=False
    )
    apply_theme(fig_pct, height=260)
    st.plotly_chart(fig_pct, use_container_width=True)

    # Three summary cards: tight / our choice / loose
    s_tight = sweep[0]   # epsilon = 0.005 (strictest)
    s_mid   = next(s for s in sweep if abs(s['epsilon'] - chosen_eps) < 0.001)
    s_loose = sweep[-1]  # epsilon = 0.100 (loosest)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="card">
            <span class="section-label">Tight (ε = {s_tight['epsilon']})</span>
            <div style="font-family:'DM Serif Display',serif; font-size:32px;
                        color:#00d4aa;">{s_tight['rashomon_size']}</div>
            <div style="font-size:12px; color:#8899bb; margin-top:4px;">
                models ({s_tight['pct_of_total']:.1%} of all)
            </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card-highlight">
            <span class="section-label">Our Choice (ε = {s_mid['epsilon']})</span>
            <div style="font-family:'DM Serif Display',serif; font-size:32px;
                        color:#ff6b6b;">{s_mid['rashomon_size']}</div>
            <div style="font-size:12px; color:#8899bb; margin-top:4px;">
                models ({s_mid['pct_of_total']:.1%} of all)
            </div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="card">
            <span class="section-label">Loose (ε = {s_loose['epsilon']})</span>
            <div style="font-family:'DM Serif Display',serif; font-size:32px;
                        color:#ffd166;">{s_loose['rashomon_size']}</div>
            <div style="font-size:12px; color:#8899bb; margin-top:4px;">
                models ({s_loose['pct_of_total']:.1%} of all)
            </div>
        </div>""", unsafe_allow_html=True)


# =============================================================================
# SCREEN 7 — ABOUT
# =============================================================================

def render_about(results):
    """
    Methodology, research context, limitations, tech stack.

    Left column (wider):
      - What the project is and why it was built
      - Six-step technical pipeline (download → train → CV → L* → predict → importance)
      - Three honest limitation callouts (model class, dataset size, few unstable patients)

    Right column (narrower):
      - Four connected research papers by Prof. Semenova with explicit link
        to which visualization in the app illustrates each paper's concept
      - Tech stack table (library + role)
      - Six reproducibility numbers so anyone can verify or reproduce

    Args:
        results (dict): Rashomon Set output — used for live numbers
    """

    st.markdown("""
    <div class="page-title">About This<br><em>Project</em></div>
    <div class="page-subtitle">
        How this was built, what decisions were made, what the limitations
        are, and how it connects to published research on the Rashomon Effect.
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown('<span class="section-label">What This Project Is</span>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p style="font-size:14px; color:#8899bb; line-height:1.9; margin:0;">
                An interactive research demonstration built to visualize
                <strong style="color:#e8edf5;">model multiplicity</strong> —
                the phenomenon where many equally-accurate machine learning models
                exist simultaneously. Built to explore and explain core concepts
                from Prof. Lesia Semenova's research group at Rutgers University.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Six pipeline steps — each explained in plain language
        st.markdown('<span class="section-label">Technical Pipeline</span>',
                    unsafe_allow_html=True)

        pipeline_steps = [
            ("1. Download Data",
             "UCI Cleveland Heart Disease dataset fetched from the official UCI "
             "ML Repository. 303 rows → 297 after dropping 6 rows with '?' missing "
             "values. Target converted from 0–4 scale to binary (0/1)."),
            ("2. Train 3,360 Models",
             "Decision trees trained across a full hyperparameter grid: "
             "7 max_depths × 4 min_samples_split × 4 min_samples_leaf × 30 seeds = "
             "3,360 unique tree configurations."),
            ("3. 5-Fold Cross Validation",
             "Each model evaluated with 5-Fold CV instead of a single train/test "
             "split. A single split on 297 patients gives coarse accuracy steps "
             "(1/60 = 0.0167), making nearly every model look identical. CV gives "
             "fine-grained differences."),
            ("4. Find True L*",
             f"L* = maximum CV accuracy across all 3,360 models "
             f"({results['reference_accuracy']:.4f}). Threshold = L* − ε. "
             f"Models above {results['threshold']:.4f} form the Rashomon Set."),
            ("5. Honest Predictions (cross_val_predict)",
             "Per-patient predictions use sklearn's cross_val_predict — each "
             "patient predicted by a model trained WITHOUT that patient. "
             "Avoids data leakage and gives honest disagreement measurements."),
            ("6. Feature Importance",
             "Rashomon Set models fit on the full dataset to populate "
             "feature_importances_. Mean, std, and CoV computed per feature "
             "across all 633 models to reveal importance instability."),
        ]

        for title, desc in pipeline_steps:
            st.markdown(f"""
            <div style="display:flex; gap:12px; margin-bottom:12px;
                        background:#111827; border:1px solid #1e2a3d;
                        border-radius:8px; padding:14px;">
                <div style="width:8px; min-width:8px; background:#00d4aa;
                            border-radius:4px; margin-top:2px;"></div>
                <div>
                    <div style="font-family:'DM Serif Display',serif; font-size:15px;
                                color:#e8edf5; margin-bottom:4px;">{title}</div>
                    <div style="font-size:13px; color:#8899bb; line-height:1.7;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Honest limitations — critical for research credibility
        st.markdown('<span class="section-label">Honest Limitations</span>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="callout coral">
            <strong>Only decision trees:</strong> We explore only one model class.
            The real Rashomon Set also includes logistic regression, neural networks,
            SVMs, and ensembles. True multiplicity is far larger.
        </div>
        <div class="callout coral">
            <strong>Small dataset:</strong> 297 patients is a small sample.
            CV accuracy varies significantly between folds (std ≈ 0.08), meaning
            Rashomon Set membership has some noise.
        </div>
        <div class="callout coral">
            <strong>Few unstable patients:</strong> Only 3 fall in the 30–70%
            uncertainty zone. This reflects both the dataset (thal strongly
            predicts outcome) and the fact that 480 of 633 models are depth-3
            trees with similar structure.
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        # Four connected research papers
        st.markdown('<span class="section-label">Research Connection</span>',
                    unsafe_allow_html=True)

        papers = [
            ("The Rashomon Set Has It All", "2025", "NeurIPS",
             "Analyzes trustworthiness of trees under multiplicity. "
             "Our depth breakdown chart directly illustrates this.", "teal"),
            ("The Double-Edged Nature of the Rashomon Set", "2025", "arXiv",
             "Risks AND opportunities of multiplicity. Our 'problem vs "
             "opportunity' framing on the Home page reflects this.", "coral"),
            ("On the Existence of Simpler ML Models", "2022", "FAccT",
             "Simple models can match complex ones. Depth-3 trees at 100% "
             "qualification rate is a live demonstration.", "gold"),
            ("Amazing Things From Many Good Models", "2024", "ICML",
             "Foundational argument for using the Rashomon Set. "
             "Our epsilon sweep shows the set growing directly.", "purple"),
        ]

        for title, year, venue, desc, color in papers:
            border_c = {"teal":"#00a882","coral":"#cc5555",
                        "gold":"#aa8833","purple":"#7c78cc"}[color]
            tag_c    = f"tag-{color}"
            st.markdown(f"""
            <div style="background:#111827; border:1px solid {border_c};
                        border-radius:8px; padding:14px; margin-bottom:10px;">
                <span class="tag {tag_c}">{venue} {year}</span>
                <div style="font-family:'DM Serif Display',serif; font-size:14px;
                            color:#e8edf5; margin:6px 0 4px 0; line-height:1.3;">
                    {title}</div>
                <div style="font-size:12px; color:#4a5a75; line-height:1.6;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        # Tech stack table
        st.markdown('<span class="section-label" style="margin-top:20px; display:block;">Tech Stack</span>',
                    unsafe_allow_html=True)

        stack = [
            ("Python 3.11",       "Language"),
            ("scikit-learn 1.8",  "Decision trees, CV"),
            ("Streamlit 1.57",    "Web app framework"),
            ("Plotly 6.7",        "Interactive charts"),
            ("Pandas 3.0",        "Data handling"),
            ("NumPy 2.4",         "Numerical operations"),
        ]
        st.markdown('<div class="card" style="padding:16px;">',
                    unsafe_allow_html=True)
        for lib, role in stack:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between;
                        padding:6px 0; border-bottom:1px solid #1e2a3d;">
                <span style="font-family:'JetBrains Mono',monospace;
                             font-size:12px; color:#00d4aa;">{lib}</span>
                <span style="font-size:12px; color:#4a5a75;">{role}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Reproducibility numbers — six exact values for verification
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Reproducibility — Exact Numbers</span>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    nums = [
        ("297",    "Patients in dataset",  "#00d4aa"),
        ("13",     "Input features",       "#00d4aa"),
        ("3,360",  "Models trained",       "#ffd166"),
        (f"{results['rashomon_size']}", "In Rashomon Set", "#ff6b6b"),
        (f"{results['reference_accuracy']:.4f}", "True L*", "#ff6b6b"),
        ("0.02",   "Epsilon used",         "#a29bfe"),
    ]
    for col, (val, label, color) in zip([c1, c2, c3, c4, c5, c6], nums):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <span class="metric-value" style="color:{color}; font-size:20px;">{val}</span>
                <span class="metric-label">{label}</span>
            </div>""", unsafe_allow_html=True)


# =============================================================================
# MAIN — entry point called by Streamlit on every user interaction
# =============================================================================

def main():
    """
    Called on every script re-run (every button click, slider drag, etc.).

    1. Initialize session_state["page"] = "Home" on first load.
    2. Call load_everything() — runs once, cached on all subsequent calls.
    3. Store results in session_state so sidebar can display live stats.
    4. Render the sidebar (navigation + stats).
    5. Route to the correct render_*() function based on current page.
    """

    # Default page on first load
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    # Load data and build analyses.
    # @st.cache_data makes this instant on all calls after the first.
    with st.spinner("Building Rashomon Set — ~2 minutes on first load..."):
        results, analyses, df = load_everything(epsilon=0.02)

    # Store in session_state for sidebar live stats
    st.session_state["results"] = results

    # Render sidebar (must come before main content)
    render_sidebar()

    # Route to correct page
    page = st.session_state.get("page", "Home")

    if   page == "Home":          render_home(results, analyses)
    elif page == "Dataset":       render_dataset(df)
    elif page == "Build the Set": render_build(results, analyses)
    elif page == "Patient View":  render_patient(results, analyses, df)
    elif page == "Features":      render_features(results, analyses)
    elif page == "Epsilon":       render_epsilon(results, analyses)
    elif page == "About":         render_about(results)


# Streamlit runs the whole file top-to-bottom on every interaction.
# This guard is convention — main() is always called here.
if __name__ == "__main__":
    main()
