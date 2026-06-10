import streamlit as st


def inject_styles() -> None:
    """Inject custom CSS ke halaman Streamlit."""
    st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 1.9rem; margin: 0; }
    .main-header p  { font-size: 0.9rem; margin: 0.4rem 0 0; opacity: 0.8; }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .metric-card .label { font-size: 0.78rem; color: #6b7280; font-weight: 500; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1e3a8a; }

    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #1e3a8a;
        border-left: 4px solid #3b82f6;
        padding-left: 0.6rem; margin: 1.5rem 0 0.8rem;
    }
    .badge-default {
        background: #eff6ff; color: #1d4ed8;
        padding: 0.2rem 0.7rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600;
    }
    div[data-testid="stTabs"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)
