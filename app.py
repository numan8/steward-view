import os
import io
import time
import pickle
from pathlib import Path
from contextlib import redirect_stdout

import pandas as pd
import streamlit as st


# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="Steward View (PyFi) — Streamlit",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "data" / "output"
CSV_PATH = OUTPUT_DIR / "labeled_data.csv"
MATCHER_PATH = OUTPUT_DIR / "matcher.pkl"


# ----------------------------
# Helpers
# ----------------------------
def get_api_key_from_secrets() -> str:
    if "OPENAI_API_KEY" in st.secrets:
        return str(st.secrets["OPENAI_API_KEY"]).strip()
    return ""


def set_api_key(key: str) -> None:
    key = (key or "").strip()
    if key:
        os.environ["OPENAI_API_KEY"] = key


@st.cache_data(show_spinner=False)
def load_output_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def safe_load_pickle(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def load_matcher_counter(path: Path):
    matcher = safe_load_pickle(path)
    if matcher is None:
        return None
    return getattr(matcher, "counter", None)


def run_pipeline():
    """
    Runs analysis.run() and captures stdout so we can display it in Streamlit.
    """
    import analysis  # package name defined in pyproject.toml

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    start = time.time()
    with redirect_stdout(buf):
        analysis.run()

    elapsed = time.time() - start
    logs = buf.getvalue()
    return elapsed, logs


def file_status_chip(label: str, ok: bool, detail: str = ""):
    emoji = "✅" if ok else "⚠️"
    txt = f"{emoji} {label}"
    if detail:
        txt += f" · {detail}"
    st.caption(txt)


# ----------------------------
# Header / Top Bar
# ----------------------------
st.title("📊 Steward View — Streamlit Runner")

with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.6, 1.2, 1.2, 1.0], vertical_alignment="center")

    # API key box (top, not only sidebar)
    with c1:
        secret_key = get_api_key_from_secrets()
        api_key_input = st.text_input(
            "OpenAI API Key (optional)",
            value=secret_key,
            type="password",
            help="Best practice: set OPENAI_API_KEY in Streamlit Secrets. This field is optional unless you use labeling/chat.",
        )
        set_api_key(api_key_input)

    # Run button
    with c2:
        st.write("")
        st.write("")
        run_btn = st.button("▶️ Run analysis.run()", use_container_width=True)

    # File status
    with c3:
        st.write("**Outputs**")
        file_status_chip("labeled_data.csv", CSV_PATH.exists(), str(CSV_PATH.relative_to(ROOT)))
        file_status_chip("matcher.pkl", MATCHER_PATH.exists(), str(MATCHER_PATH.relative_to(ROOT)))

    # Quick actions
    with c4:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.toast("Cache cleared. Data will reload.", icon="✅")

st.write("")


# ----------------------------
# Pipeline run area (runs once per click)
# ----------------------------
if "last_run_logs" not in st.session_state:
    st.session_state.last_run_logs = ""
if "last_run_time" not in st.session_state:
    st.session_state.last_run_time = None

if run_btn:
    with st.spinner("Running pipeline…"):
        try:
            elapsed, logs = run_pipeline()
            st.session_state.last_run_logs = logs
            st.session_state.last_run_time = elapsed

            # Ensure fresh read of the new CSV
            st.cache_data.clear()

            st.success(f"Pipeline completed in {elapsed:.1f}s")
            if logs.strip():
                with st.expander("📜 Console logs (latest run)", expanded=False):
                    st.text_area("Output", logs, height=260)
                    st.download_button(
                        "⬇️ Download logs",
                        data=logs.encode("utf-8"),
                        file_name="pipeline_logs.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
            else:
                st.info("No console output captured.")
        except Exception as e:
            st.error("Pipeline failed. Copy the error below and I’ll help you fix it.")
            st.exception(e)

elif st.session_state.last_run_time is not None:
    st.caption(f"Last run: {st.session_state.last_run_time:.1f}s")


# ----------------------------
# Tabs
# ----------------------------
tab_run, tab_data, tab_stats, tab_qa = st.tabs(["🧪 Run", "📄 Data Explorer", "📈 Stats", "💬 Q&A"])


# ----------------------------
# Tab: Run
# ----------------------------
with tab_run:
    st.subheader("Run Instructions")
    st.markdown(
        """
- Click **Run analysis.run()** at the top.
- Expected outputs:
  - `data/output/labeled_data.csv`
  - `data/output/matcher.pkl` (optional)
"""
    )
    if st.session_state.last_run_logs.strip():
        with st.expander("📜 Latest logs", expanded=True):
            st.text_area("Logs", st.session_state.last_run_logs, height=280)


# ----------------------------
# Tab: Data Explorer
# ----------------------------
with tab_data:
    st.subheader("Output dataset")

    if not CSV_PATH.exists():
        st.warning("No output CSV yet. Click **Run analysis.run()** above.")
    else:
        df = load_output_csv(str(CSV_PATH))

        # Controls
        with st.container(border=True):
            f1, f2, f3, f4 = st.columns([1.4, 1.4, 1.0, 1.0], vertical_alignment="center")
            with f1:
                cols_default = df.columns.tolist()[:12]
                cols = st.multiselect(
                    "Columns",
                    df.columns.tolist(),
                    default=cols_default,
                    help="Select which columns to display.",
                )
            with f2:
                search = st.text_input(
                    "Search (contains, any column)",
                    value="",
                    placeholder="e.g., Walmart, Visa, refund, etc.",
                )
            with f3:
                only_missing = st.checkbox("Only rows with any missing", value=False)
            with f4:
                row_limit = st.number_input("Row limit", min_value=50, max_value=20000, value=1000, step=50)

        view = df.copy()

        if only_missing:
            view = view[view.isna().any(axis=1)]

        if search.strip():
            s = search.strip().lower()
            mask = view.astype(str).apply(lambda r: r.str.lower().str.contains(s, na=False)).any(axis=1)
            view = view[mask]

        if cols:
            view = view[cols]

        st.caption(f"Loaded: `{CSV_PATH}` • Rows: {len(df):,} • Showing: {min(len(view), int(row_limit)):,}")
        st.dataframe(view.head(int(row_limit)), use_container_width=True, height=520)

        # Downloads
        d1, d2 = st.columns([1, 1])
        with d1:
            st.download_button(
                "⬇️ Download full labeled_data.csv",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="labeled_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "⬇️ Download filtered view",
                data=view.to_csv(index=False).encode("utf-8"),
                file_name="labeled_data_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ----------------------------
# Tab: Stats
# ----------------------------
with tab_stats:
    st.subheader("Quick stats")

    if not CSV_PATH.exists():
        st.info("Run the pipeline first so the CSV exists.")
    else:
        df = load_output_csv(str(CSV_PATH))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", f"{len(df):,}")
        k2.metric("Columns", f"{df.shape[1]:,}")
        k3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        k4.metric("Columns w/ missing", f"{int((df.isna().sum() > 0).sum()):,}")

        st.divider()

        c1, c2 = st.columns([1.1, 1.0])

        with c1:
            st.write("**Column types**")
            dtype_counts = df.dtypes.astype(str).value_counts()
            st.dataframe(dtype_counts.rename("count").reset_index().rename(columns={"index": "dtype"}), use_container_width=True)

            st.write("**Missing values (top 20)**")
            na = df.isna().sum().sort_values(ascending=False)
            na = na[na > 0].head(20)
            if len(na) == 0:
                st.success("No missing values detected.")
            else:
                na_df = na.rename("missing").reset_index().rename(columns={"index": "column"})
                st.dataframe(na_df, use_container_width=True, height=360)

        with c2:
            st.write("**Preview: numeric summary**")
            num = df.select_dtypes(include="number")
            if num.shape[1] == 0:
                st.info("No numeric columns found.")
            else:
                st.dataframe(num.describe().T, use_container_width=True, height=420)

        st.divider()

        st.subheader("Amazon matcher (optional)")
        counter = load_matcher_counter(MATCHER_PATH)
        if counter is None:
            st.info("No matcher counter found (matcher.pkl missing or unreadable).")
        else:
            try:
                cdf = pd.DataFrame({"rule": list(counter.keys()), "count": list(counter.values())})
                cdf = cdf.sort_values("count", ascending=False)
                st.dataframe(cdf, use_container_width=True, height=360)
            except Exception:
                st.write(counter)


# ----------------------------
# Tab: Q&A
# ----------------------------
with tab_qa:
    st.subheader("Ask a question (uses OpenAI)")
    st.caption("This calls the repo’s `analysis.inspect.Chat` which needs `OPENAI_API_KEY`.")

    q1, q2 = st.columns([1.4, 1.0], vertical_alignment="center")
    with q1:
        question = st.text_input("Question about the output CSV", value="", placeholder="e.g., What are the most common merchant categories?")
    with q2:
        st.write("")
        ask = st.button("💬 Ask", use_container_width=True)

    if ask:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set. Add it in Streamlit Secrets or type it at the top.")
        elif not CSV_PATH.exists():
            st.error("Run the pipeline first so the CSV exists.")
        elif not question.strip():
            st.error("Type a question first.")
        else:
            try:
                from analysis.inspect import Chat

                with st.spinner("Thinking…"):
                    chat = Chat()  # uses the pipeline’s CSV output path internally
                    answer = chat.msg(question.strip())

                if isinstance(answer, str) and answer.strip():
                    st.success(answer)
                else:
                    st.success("Answered. If you don’t see text, the Chat method may print output instead.")
            except Exception as e:
                st.error("Chat failed. Paste the error and I’ll fix it.")
                st.exception(e)
