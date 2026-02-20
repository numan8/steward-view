import os
import io
import time
import pickle
from pathlib import Path
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st


# ----------------------------
# Page config + styling
# ----------------------------
st.set_page_config(
    page_title="Steward View",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }

[data-testid="stMetric"] {
  background: rgba(255,255,255,0.60);
  border: 1px solid rgba(0,0,0,0.06);
  padding: 12px 14px;
  border-radius: 14px;
}

.sidebar-title {
  font-weight: 700;
  font-size: 0.95rem;
  margin: 0.25rem 0 0.5rem 0;
  opacity: 0.85;
}

hr { margin: 1rem 0; opacity: 0.25; }
</style>
""",
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "output"
CSV_PATH = OUTPUT_DIR / "labeled_data.csv"
MATCHER_PATH = OUTPUT_DIR / "matcher.pkl"


# ----------------------------
# Helpers
# ----------------------------
def _ensure_api_key() -> str:
    """
    Enforce OPENAI_API_KEY exists.
    Prefer Streamlit Secrets, then environment.
    """
    key = ""
    if "OPENAI_API_KEY" in st.secrets:
        key = str(st.secrets["OPENAI_API_KEY"]).strip()
        os.environ["OPENAI_API_KEY"] = key
    else:
        key = os.getenv("OPENAI_API_KEY", "").strip()

    if not key:
        st.error(
            "OPENAI_API_KEY is required.\n\n"
            "Add it in Streamlit → App Settings → **Secrets** as:\n"
            'OPENAI_API_KEY = "sk-..."\n\n'
            "Then rerun the app."
        )
        st.stop()

    return key


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_matcher_counter(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            matcher = pickle.load(f)
        return getattr(matcher, "counter", None)
    except Exception:
        return None


def run_pipeline():
    """
    Runs analysis.run() and captures stdout for logs.
    """
    import analysis  # package defined via pyproject.toml

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    start = time.time()
    with redirect_stdout(buf):
        analysis.run()
    elapsed = time.time() - start
    return elapsed, buf.getvalue()


def fmt_money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def safe_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns


def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


# ----------------------------
# Enforce key
# ----------------------------
_ensure_api_key()


# ----------------------------
# Header
# ----------------------------
left, right = st.columns([0.72, 0.28], vertical_alignment="center")
with left:
    st.markdown("## 📊 Steward View")
    st.caption("Client-ready dashboard for the Steward View pipeline output (cleaned + tagged + labeled transactions).")
with right:
    st.markdown("**Environment:** Streamlit Cloud / Hosted")
    st.caption("OpenAI key loaded ✅")

st.divider()


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">🚀 Pipeline</div>', unsafe_allow_html=True)

    run_btn = st.button("▶ Run pipeline (analysis.run)", use_container_width=True)
    show_logs = st.checkbox("Show logs after run", value=True)

    st.caption("Expected output:")
    st.code("data/output/labeled_data.csv", language="text")

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">🧭 Explorer</div>', unsafe_allow_html=True)
    preview_rows = st.slider("Rows to preview", 50, 2000, 250, step=50)


# ----------------------------
# Run + Logs row
# ----------------------------
run_col, log_col = st.columns([0.55, 0.45])

with run_col:
    st.subheader("Run")
    st.write("This runs the full pipeline including OpenAI labeling and writes the labeled CSV output.")
    if run_btn:
        with st.spinner("Running pipeline…"):
            try:
                elapsed, logs = run_pipeline()
                st.success(f"Pipeline completed in {elapsed:.1f}s")
                st.session_state["last_logs"] = logs
            except Exception as e:
                st.error("Pipeline failed. Copy the error below and I’ll help you fix it.")
                st.exception(e)

with log_col:
    st.subheader("Logs")
    logs = st.session_state.get("last_logs", "")
    if logs and show_logs:
        st.text_area("Console output", logs, height=220)
    else:
        st.caption("Run the pipeline to see logs here.")

st.divider()


# ----------------------------
# Load dataset
# ----------------------------
if not CSV_PATH.exists():
    st.warning("No output CSV found yet. Run the pipeline from the sidebar.")
    st.stop()

df = load_csv(str(CSV_PATH))

if safe_col(df, "date"):
    df["date"] = to_datetime_safe(df["date"])


# ----------------------------
# KPI row
# ----------------------------
k1, k2, k3, k4 = st.columns(4)

total_rows = len(df)
accounts = df["account"].nunique() if safe_col(df, "account") else None
total_spend = df["amount"].sum() if safe_col(df, "amount") else None

date_min = df["date"].min() if safe_col(df, "date") else None
date_max = df["date"].max() if safe_col(df, "date") else None

with k1:
    st.metric("Rows", f"{total_rows:,}")
with k2:
    st.metric("Accounts", f"{accounts:,}" if accounts is not None else "—")
with k3:
    st.metric("Total Amount", fmt_money(total_spend) if total_spend is not None else "—")
with k4:
    if date_min is not None and date_max is not None:
        st.metric("Date Range", f"{date_min.date()} → {date_max.date()}")
    else:
        st.metric("Date Range", "—")


# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_explorer, tab_quality, tab_matcher = st.tabs(
    ["📌 Overview", "🧭 Data Explorer", "✅ Data Quality", "🧩 Matcher"]
)

# ---- Overview ----
with tab_overview:
    left, right = st.columns([0.6, 0.4])

    with left:
        st.subheader("Recent Transactions")
        show_cols = [c for c in ["date", "amount", "description", "account", "llm_category", "llm_vendor"] if c in df.columns]
        if not show_cols:
            show_cols = df.columns.tolist()[:8]

        view_df = df.sort_values("date", ascending=False) if safe_col(df, "date") else df
        st.dataframe(view_df[show_cols].head(25), use_container_width=True, height=360)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV",
            data=csv_bytes,
            file_name="labeled_data.csv",
            mime="text/csv",
        )

    with right:
        st.subheader("Top Breakdowns")

        if safe_col(df, "account"):
            st.write("**Transactions by account**")
            st.bar_chart(df["account"].value_counts().head(12))

        if safe_col(df, "llm_category"):
            st.write("**Top categories**")
            st.bar_chart(df["llm_category"].fillna("(missing)").value_counts().head(12))

        if safe_col(df, "llm_vendor"):
            st.write("**Top vendors**")
            st.bar_chart(df["llm_vendor"].fillna("(missing)").value_counts().head(12))


# ---- Explorer ----
with tab_explorer:
    st.subheader("Explore & Filter")

    f1, f2, f3, f4 = st.columns([0.28, 0.24, 0.24, 0.24])

    # Date filter
    if safe_col(df, "date") and df["date"].notna().any():
        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
        with f1:
            date_range = st.date_input("Date range", (min_d, max_d))
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            d0, d1 = date_range
            mask_date = (df["date"].dt.date >= d0) & (df["date"].dt.date <= d1)
        else:
            mask_date = pd.Series([True] * len(df))
    else:
        with f1:
            st.caption("No date column.")
        mask_date = pd.Series([True] * len(df))

    # Account filter
    if safe_col(df, "account"):
        with f2:
            accounts_list = ["(all)"] + sorted(df["account"].dropna().astype(str).unique().tolist())
            account_choice = st.selectbox("Account", accounts_list, index=0)
        mask_acc = True if account_choice == "(all)" else (df["account"].astype(str) == account_choice)
    else:
        mask_acc = True

    # Category filter
    if safe_col(df, "llm_category"):
        with f3:
            cats = ["(all)"] + sorted(df["llm_category"].fillna("(missing)").astype(str).unique().tolist())
            cat_choice = st.selectbox("Category", cats, index=0)
        if cat_choice == "(all)":
            mask_cat = True
        elif cat_choice == "(missing)":
            mask_cat = df["llm_category"].isna()
        else:
            mask_cat = df["llm_category"].astype(str) == cat_choice
    else:
        mask_cat = True

    # Search
    with f4:
        query = st.text_input("Search description/vendor", value="").strip().lower()

    filtered = df[mask_date & mask_acc & mask_cat].copy()

    if query:
        cols_to_search = [c for c in ["description", "llm_vendor"] if c in filtered.columns]
        if cols_to_search:
            mask_q = filtered[cols_to_search].astype(str).apply(
                lambda r: r.str.lower().str.contains(query, na=False)
            ).any(axis=1)
            filtered = filtered[mask_q]

    st.caption(f"Showing **{len(filtered):,}** rows after filters.")

    with st.expander("Columns", expanded=False):
        col_pick = st.multiselect(
            "Choose columns to display",
            options=df.columns.tolist(),
            default=[c for c in ["date", "amount", "description", "account", "llm_category", "llm_vendor"] if c in df.columns]
            or df.columns.tolist()[:10],
        )

    st.dataframe(
        filtered[col_pick].head(preview_rows) if col_pick else filtered.head(preview_rows),
        use_container_width=True,
        height=520,
    )


# ---- Quality ----
with tab_quality:
    st.subheader("Quality checks")

    q1, q2 = st.columns([0.55, 0.45])

    with q1:
        st.write("**Missing values (top 20)**")
        na = df.isna().sum().sort_values(ascending=False).head(20)
        na_df = na.rename("missing").reset_index().rename(columns={"index": "column"})
        st.dataframe(na_df, use_container_width=True, height=420)

    with q2:
        st.write("**Schema (dtypes)**")
        dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
        dtype_counts.columns = ["dtype", "count"]
        st.dataframe(dtype_counts, use_container_width=True)

        notes = []
        if "llm_vendor" in df.columns and df["llm_vendor"].isna().mean() > 0.15:
            notes.append("Some `llm_vendor` values are missing — may indicate rate limits or partial run.")
        if "llm_category" in df.columns and df["llm_category"].isna().mean() > 0.15:
            notes.append("Some `llm_category` values are missing — may indicate rate limits or partial run.")
        if not notes:
            notes.append("No major issues detected from basic checks.")

        st.write("**Notes**")
        for n in notes:
            st.info(n)


# ---- Matcher ----
with tab_matcher:
    st.subheader("Amazon matcher")

    counter = load_matcher_counter(MATCHER_PATH)
    if counter is None:
        st.info("No matcher stats found (matcher.pkl missing or unreadable).")
    else:
        try:
            cdf = pd.DataFrame({"rule": list(counter.keys()), "count": list(counter.values())}).sort_values(
                "count", ascending=False
            )
            st.caption("Matches made by each reconciliation rule.")
            st.dataframe(cdf, use_container_width=True, height=460)
            st.bar_chart(cdf.set_index("rule")["count"])
        except Exception:
            st.write(counter)


st.caption("© Steward View — Streamlit dashboard")
