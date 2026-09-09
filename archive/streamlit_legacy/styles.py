"""Global CSS design system and theme styles for the PDF RAG application."""

from __future__ import annotations

CUSTOM_CSS = """
<style>
/* ── Google Fonts Import ────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global CSS Variables & Theme ─────────────────────────────────────────── */
:root {
    --font-primary: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    
    --bg-dark: #090d16;
    --surface-card: rgba(17, 24, 39, 0.7);
    --surface-hover: rgba(30, 41, 59, 0.8);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(99, 102, 241, 0.4);
    
    --primary-indigo: #6366f1;
    --primary-violet: #8b5cf6;
    --primary-cyan: #06b6d4;
    --primary-emerald: #10b981;
    --primary-amber: #f59e0b;
    --primary-rose: #f43f5e;
    
    --gradient-primary: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    --gradient-emerald: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
    --gradient-card: linear-gradient(145deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
}

/* ── Typography Overrides ─────────────────────────────────────────────────── */
html, body, [class*="css"], .stMarkdown, .stText, p, span, label {
    font-family: var(--font-primary) !important;
}

code, pre {
    font-family: var(--font-mono) !important;
}

/* ── Main Container Spacing ─────────────────────────────────────────────────── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1200px !important;
}

/* ── Hero Banner ────────────────────────────────────────────────────────────── */
.hero-wrapper {
    background: radial-gradient(ellipse at 50% -20%, rgba(99, 102, 241, 0.25) 0%, rgba(15, 23, 42, 0) 70%);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
}

.hero-wrapper::before {
    content: '';
    position: absolute;
    top: 0;
    left: 15%;
    right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.6), rgba(236, 72, 153, 0.6), transparent);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.9rem;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #a5b4fc;
    margin-bottom: 0.9rem;
}

.hero-badge .badge-dot {
    width: 7px;
    height: 7px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 10px #10b981;
    animation: pulse-dot 2s infinite ease-in-out;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

.hero-title {
    font-size: 2.35rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
    margin: 0 0 0.6rem 0 !important;
    background: var(--gradient-primary);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
    line-height: 1.5;
    font-weight: 400;
}

/* ── Modern Glassmorphic Metric Cards ─────────────────────────────────────── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

@media (max-width: 900px) {
    .metrics-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

.metric-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.15rem 1.2rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-3px);
    border-color: var(--border-hover);
    box-shadow: 0 12px 24px -10px rgba(99, 102, 241, 0.2);
    background: var(--surface-hover);
}

.metric-card .icon-wrapper {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    margin-bottom: 0.6rem;
    background: rgba(255, 255, 255, 0.05);
}

.metric-card .metric-label {
    font-size: 0.76rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 0.25rem;
}

.metric-card .metric-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -0.02em;
}

/* ── Welcome Empty State ──────────────────────────────────────────────────── */
.empty-state-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 3rem 2rem;
    text-align: center;
    margin: 2rem 0;
    backdrop-filter: blur(12px);
    position: relative;
}

.empty-state-icon {
    font-size: 3.5rem;
    margin-bottom: 1.2rem;
    display: inline-block;
    filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.4));
}

.empty-state-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.6rem;
}

.empty-state-desc {
    font-size: 0.98rem;
    color: #94a3b8;
    max-width: 580px;
    margin: 0 auto 2rem auto;
    line-height: 1.6;
}

.features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.2rem;
    max-width: 850px;
    margin: 0 auto;
    text-align: left;
}

@media (max-width: 768px) {
    .features-grid {
        grid-template-columns: 1fr;
    }
}

.feature-pill {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.2rem;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.feature-pill:hover {
    border-color: rgba(99, 102, 241, 0.3);
    transform: translateY(-2px);
}

.feature-pill-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.feature-pill-desc {
    font-size: 0.8rem;
    color: #94a3b8;
    line-height: 1.45;
}

/* ── Source Attribution Banners ───────────────────────────────────────────── */
.attribution-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    font-weight: 500;
}

.attribution-document {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #6ee7b7;
}

.attribution-general {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fcd34d;
}

.attribution-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.badge-doc {
    background: rgba(16, 185, 129, 0.25);
    color: #a7f3d0;
}

.badge-gen {
    background: rgba(245, 158, 11, 0.25);
    color: #fde68a;
}

/* ── Retrieval Confidence Meters ──────────────────────────────────────────── */
.retrieval-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.1rem;
    margin-bottom: 0.85rem;
}

.retrieval-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.6rem;
}

.confidence-bar-bg {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 9999px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.confidence-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Sidebar Styling ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0c121e !important;
    border-right: 1px solid var(--border-subtle) !important;
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1.2rem;
}

.sidebar-card-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #818cf8;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ── Buttons & Interactive Controls ───────────────────────────────────────── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    border: 1px solid var(--border-subtle) !important;
}

.stButton > button:hover {
    border-color: var(--primary-indigo) !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25) !important;
    transform: translateY(-1px) !important;
}

.stDownloadButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

/* ── Streamlit Expanders ──────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    border-radius: 10px !important;
    background: rgba(255, 255, 255, 0.02) !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

/* ── File Uploader Dropzone Styling ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(99, 102, 241, 0.35) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    background: rgba(15, 23, 42, 0.4) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--primary-indigo) !important;
    background: rgba(99, 102, 241, 0.05) !important;
}
</style>
"""


def get_custom_css() -> str:
    """Return the global CSS style block."""
    return CUSTOM_CSS
