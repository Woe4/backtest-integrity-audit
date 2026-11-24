"""Streamlit UI for running backtest integrity checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from backtest_integrity_audit import cscv_pbo, deflated_sharpe_ratio

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = ROOT / "data" / "cache"
SAMPLES = {
    "Gaussian noise": SAMPLE_DIR / "noise_returns.parquet",
    "SPY": SAMPLE_DIR / "spy_returns.parquet",
    "QQQ": SAMPLE_DIR / "qqq_returns.parquet",
}


def _load_matrix(source: str, sample_name: str, uploaded_file) -> tuple[pd.DataFrame, str]:
    if source == "Cached sample":
        path = SAMPLES[sample_name]
        if not path.is_file():
            raise FileNotFoundError(
                f"Sample file not found: {path.relative_to(ROOT)}. Generate samples with `backtest-samples`."
            )
        return pd.read_parquet(path), str(path.relative_to(ROOT))
    if uploaded_file is None:
        raise ValueError("Choose a Parquet file before running the audit.")
    return pd.read_parquet(uploaded_file), uploaded_file.name


def main() -> None:
    st.set_page_config(page_title="Backtest Integrity Auditor", page_icon="📊", layout="wide")
    st.title("Backtest Integrity Auditor")
    st.write(
        "Upload a T × N strategy return matrix or select a cached sample, then run "
        "the Deflated Sharpe Ratio and CSCV Probability of Backtest Overfitting."
    )

    with st.form("audit_form"):
        source = st.radio("Return matrix", ("Cached sample", "Upload Parquet"), horizontal=True)
        sample_name = ""
        uploaded_file = None
        if source == "Cached sample":
            sample_name = st.selectbox("Choose a sample", tuple(SAMPLES))
        else:
            uploaded_file = st.file_uploader(
                "Choose a Parquet file containing daily/simple returns (one strategy per column)",
                type=("parquet",),
            )

        left, right = st.columns(2)
        with left:
            n_blocks = st.selectbox("CSCV blocks", (8, 12, 16), index=0)
            pbo_metric = st.selectbox(
                "CSCV performance metric",
                ("sharpe", "total_return"),
                format_func=lambda value: "Sharpe ratio" if value == "sharpe" else "Compounded total return",
            )
        with right:
            risk_free_rate = st.number_input(
                "Risk-free return per observation",
                value=0.0,
                step=0.0001,
                format="%.6f",
                help="Used by both Sharpe calculations. Enter a per-period return, not an annual rate.",
            )
            effective_trials = st.number_input(
                "Effective independent trials for DSR (0 = number of columns)",
                min_value=0,
                value=0,
                step=1,
                help="Use a smaller count when strategy variants are strongly correlated.",
            )

        run_requested = st.form_submit_button("Run audit", type="primary", use_container_width=True)

    if not run_requested:
        st.info("Choose a return matrix and settings, then select **Run audit**.")
        return

    try:
        returns, source_label = _load_matrix(source, sample_name, uploaded_file)
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            raise ValueError("The Parquet file must contain a non-empty T × N table.")
        if returns.shape[1] < 2:
            raise ValueError("At least two strategy columns are required for the audit.")

        with st.spinner("Calculating DSR and enumerating CSCV splits…"):
            dsr = deflated_sharpe_ratio(
                returns,
                n_trials=int(effective_trials) if effective_trials else None,
                risk_free_rate=float(risk_free_rate),
            )
            pbo = cscv_pbo(
                returns,
                n_blocks=int(n_blocks),
                metric=pbo_metric,
                risk_free_rate=float(risk_free_rate),
            )
    except Exception as exc:
        st.error(f"Audit failed: {type(exc).__name__}: {exc}")
        st.caption("Check that the file is a Parquet T × N matrix of finite simple returns.")
        return

    st.success(f"Audit completed for `{source_label}` — {returns.shape[0]:,} observations × {returns.shape[1]:,} strategies.")

    first_row = st.columns(4)
    first_row[0].metric("Deflated Sharpe Ratio", f"{dsr.dsr:.1%}")
    first_row[1].metric("Selected strategy", dsr.selected_strategy)
    first_row[2].metric("Selected Sharpe", f"{dsr.selected_sharpe:.4f}")
    first_row[3].metric("Expected max Sharpe", f"{dsr.expected_max_sharpe:.4f}")

    second_row = st.columns(4)
    second_row[0].metric("PBO", f"{pbo.pbo:.1%}")
    second_row[1].metric("CSCV splits", f"{pbo.n_splits:,}")
    second_row[2].metric("Blocks", pbo.n_blocks)
    second_row[3].metric("Strategies", pbo.n_strategies)

    with st.expander("What these results mean"):
        st.markdown(
            "- **DSR** is the estimated probability that the selected strategy's true per-period Sharpe "
            "exceeds the expected maximum Sharpe from the trial search. It accounts for sample length, "
            "skewness, and kurtosis.\n"
            "- **PBO** is the share of CSCV splits where the in-sample winner ranks below the median "
            "out of sample. Lower is better.\n"
            "- Both statistics are diagnostics; interpret them in light of how many independent trials "
            "were actually run and how the return matrix was constructed."
        )
        st.write(
            f"DSR moments: skewness {dsr.skewness:.4f}, excess kurtosis {dsr.excess_kurtosis:.4f}; "
            f"effective trials {dsr.effective_trials}. PBO ranks strategies by **{pbo.metric}**."
        )

    st.subheader("CSCV split results")
    display = pbo.splits.copy()
    display["oos_relative_rank"] = display["oos_relative_rank"].round(4)
    display["in_sample_score"] = display["in_sample_score"].round(6)
    display["oos_score"] = display["oos_score"].round(6)
    st.dataframe(display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
