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
def get_api_key() -> str:
    # Prefer Streamlit Secrets
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


def load_matcher_counter(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            matcher = pickle.load(f)
        # The README says matcher.counter exists
        return getattr(matcher, "counter", None)
    except Exception:
        return None


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


# ----------------------------
# UI
# ----------------------------
st.title("📊 Steward View — Streamlit Runner")

with st.sidebar:
    st.header("⚙️ Settings")

    secret_key = get_api_key()
    api_key_input = st.text_input(
        "OpenAI API Key (optional but needed for labeling/chat)",
        value=secret_key,
        type="password",
        help="Recommended: set OPENAI_API_KEY in Streamlit Secrets instead of typing it here.",
    )

    set_api_key(api_key_input)

    st.divider()

    st.subheader("Pipeline")
    run_btn = st.button("▶️ Run analysis.run()", use_container_width=True)

    st.caption("Outputs expected:")
    st.code("data/output/labeled_data.csv\n(data/output/matcher.pkl optional)", language="text")


colA, colB = st.columns([1.2, 1])

with colA:
    st.subheader("Run + Logs")

    if run_btn:
        with st.spinner("Running pipeline…"):
            try:
                elapsed, logs = run_pipeline()
                st.success(f"Done in {elapsed:.1f}s")
                if logs.strip():
                    st.text_area("Console output", logs, height=260)
                else:
                    st.info("No console output captured.")
            except Exception as e:
                st.error("Pipeline failed. Copy the error below and I’ll help you fix it.")
                st.exception(e)

    st.subheader("Output dataset")

    if CSV_PATH.exists():
        df = load_output_csv(str(CSV_PATH))
        st.caption(f"Loaded: `{CSV_PATH}`  •  Rows: {len(df):,}  •  Columns: {df.shape[1]:,}")

        # Simple filters
        with st.expander("🔎 Filters", expanded=False):
            cols = st.multiselect("Show columns", df.columns.tolist(), default=df.columns.tolist()[:12])
            search = st.text_input("Search (contains, any column)", value="")

        view = df.copy()
        if search.strip():
            s = search.strip().lower()
            mask = view.astype(str).apply(lambda r: r.str.lower().str.contains(s, na=False)).any(axis=1)
            view = view[mask]

        if cols:
            view = view[cols]

        st.dataframe(view, use_container_width=True, height=420)

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download labeled_data.csv",
            data=csv_bytes,
            file_name="labeled_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.warning("No output CSV yet. Click **Run analysis.run()** in the sidebar.")

with colB:
    st.subheader("Quick Stats")

    if CSV_PATH.exists():
        df = load_output_csv(str(CSV_PATH))

        # Basic numeric overview
        st.write("**Column overview**")
        st.write(df.dtypes.astype(str).value_counts())

        st.divider()

        st.write("**Missing values (top 15)**")
        na = df.isna().sum().sort_values(ascending=False).head(15)
        st.dataframe(na.rename("missing").reset_index().rename(columns={"index": "column"}), use_container_width=True)

        st.divider()

        st.subheader("Amazon matcher (optional)")
        counter = load_matcher_counter(MATCHER_PATH)
        if counter is None:
            st.info("No matcher counter found (matcher.pkl missing or unreadable).")
        else:
            # counter might be dict-like or Counter
            try:
                cdf = pd.DataFrame({"rule": list(counter.keys()), "count": list(counter.values())})
                cdf = cdf.sort_values("count", ascending=False)
                st.dataframe(cdf, use_container_width=True, height=260)
            except Exception:
                st.write(counter)

    st.divider()

    st.subheader("Optional: Ask a question (uses OpenAI)")
    st.caption("This calls the repo’s `analysis.inspect.Chat` which needs `OPENAI_API_KEY`.")

    question = st.text_input("Question about the output CSV", value="")
    ask = st.button("💬 Ask", use_container_width=True)

    if ask:
        if not os.environ.get("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set. Add it in Streamlit Secrets or sidebar.")
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
                # Some implementations return a string, some print—handle both
                if isinstance(answer, str) and answer.strip():
                    st.success(answer)
                else:
                    st.success("Answered. If you don’t see text, the Chat method may print output instead.")
            except Exception as e:
                st.error("Chat failed. Paste the error and I’ll fix it.")
                st.exception(e)
