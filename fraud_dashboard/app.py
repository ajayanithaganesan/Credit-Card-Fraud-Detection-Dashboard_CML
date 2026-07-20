"""
fraud_dashboard.app
-------------------
Streamlit UI for the Credit Card Fraud Detection Dashboard.
"""

from __future__ import annotations

import io

import plotly.graph_objects as go
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from . import utils

BLUE = "#1e40af"
GREEN = "#16a34a"
RED = "#dc2626"
SLATE = "#334155"


@st.cache_resource(show_spinner=False)
def _get_model():
    return utils.load_model()


@st.cache_resource(show_spinner=False)
def _get_scaler():
    return utils.load_scaler()


@st.cache_data(show_spinner=False)
def _get_sample_csv_bytes_from_s3(bucket: str, key: str, region: str | None = None):
    return utils.download_sample_csv_bytes_from_s3(bucket, key, region=region)


def main() -> None:
    st.set_page_config(
        page_title="Credit Card Fraud Detection Dashboard",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: #ffffff; }}
            .block-container {{ padding-top: 2rem; max-width: 1400px; }}

            .bank-header {{
                background: {BLUE};
                padding: 1.5rem 2rem;
                border-radius: 12px;
                color: #ffffff;
                margin-bottom: 1.5rem;
            }}
            .bank-header h1 {{ margin: 0; font-size: 1.8rem; font-weight: 700; }}
            .bank-header p  {{ margin: 0.25rem 0 0; opacity: 0.9; font-size: 1rem; }}
            .bank-sub {{ color: {SLATE}; font-size: 0.95rem; margin-bottom: 1rem; }}

            div[data-testid="stMetric"] {{
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 1rem 1.25rem;
            }}

            .risk-row {{
                display: flex; justify-content: space-between;
                padding: 0.5rem 0.9rem; margin-bottom: 0.4rem;
                border-radius: 8px; border: 1px solid #fecaca;
                background: #fef2f2; color: {SLATE}; font-size: 0.95rem;
            }}
            .risk-prob {{ font-weight: 700; color: {RED}; }}

            .section-title {{
                font-size: 1.25rem; font-weight: 700; color: {SLATE};
                margin: 1.5rem 0 0.75rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="bank-header">
            <h1>💳 Credit Card Transaction Fraud Detection Dashboard</h1>
            <p>Transaction Risk Monitoring System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="bank-sub">Upload customer transaction history in CSV format to '
        "identify suspicious transactions.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("### 📁 Upload Transactions CSV")
    uploaded_file = st.file_uploader(
        "Upload Transactions CSV", type=["csv"], label_visibility="collapsed"
    )

    sample_path = utils.SAMPLE_DATA_PATH
    sample_bucket, sample_key, sample_region = utils.get_sample_data_s3_config()
    sample_available = sample_path.exists() or (sample_bucket and sample_key)

    if uploaded_file is None and sample_available:
        if st.button("Use sample dataset"):
            st.session_state["use_sample"] = True

    if uploaded_file is not None:
        st.session_state["use_sample"] = False

    data_source = uploaded_file
    if data_source is None and st.session_state.get("use_sample"):
        if sample_bucket and sample_key:
            try:
                sample_bytes = _get_sample_csv_bytes_from_s3(
                    sample_bucket, sample_key, sample_region
                )
                data_source = io.BytesIO(sample_bytes)
            except Exception as exc:  # noqa: BLE001
                st.warning(
                    "Could not load the sample dataset from S3, so the app will "
                    f"fall back to the local sample file if available. Details: {exc}"
                )
                if sample_path.exists():
                    data_source = sample_path
        elif sample_path.exists():
            data_source = sample_path

    if data_source is None:
        if sample_available:
            st.info("Awaiting a transactions CSV upload or sample dataset selection.")
        else:
            st.info(
                "Awaiting a transactions CSV upload. Configure SAMPLE_DATA_BUCKET "
                "and SAMPLE_DATA_KEY to enable the sample dataset button from S3."
            )
        st.stop()

    try:
        model = _get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    scaler = _get_scaler()
    if scaler is None:
        st.warning(
            "No scaler.pkl found. Time and Amount were scaled during training, so "
            "predictions may be unreliable until you add the fitted StandardScaler "
            "as scaler.pkl in the project root."
        )

    try:
        raw_df = utils.read_csv(data_source)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()

    missing = utils.missing_feature_columns(raw_df)
    if missing:
        st.warning(
            "The uploaded file is missing expected feature columns: "
            f"{', '.join(missing)}. They will be treated as 0 for prediction."
        )

    with st.spinner("Analysing transactions..."):
        result = utils.run_predictions(model, raw_df, scaler=scaler)

    st.success(f"✅ Analysis complete — {len(result):,} transactions processed.")

    summary = utils.build_summary(result)

    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{summary['total']:,}")
    c2.metric("Fraudulent Transactions", f"{summary['fraud']:,}")
    c3.metric("Genuine Transactions", f"{summary['genuine']:,}")
    c4.metric("Fraud Percentage", f"{summary['fraud_pct']:.2f}%")

    st.markdown(
        '<div class="section-title">Fraud Distribution</div>', unsafe_allow_html=True
    )
    labels = ["Genuine", "Fraud"]
    values = [summary["genuine"], summary["fraud"]]
    colors = [GREEN, RED]

    chart_left, chart_right = st.columns(2)

    with chart_left:
        pie = go.Figure(
            data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors))]
        )
        pie.update_traces(textinfo="percent+label", sort=False)
        pie.update_layout(
            title="Pie Chart",
            height=360,
            margin=dict(t=50, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(pie, use_container_width=True)

    with chart_right:
        donut = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.55,
                    marker=dict(colors=colors),
                )
            ]
        )
        donut.update_traces(textinfo="percent+label", sort=False)
        donut.update_layout(
            title="Donut Chart",
            height=360,
            margin=dict(t=50, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(donut, use_container_width=True)

    st.markdown(
        '<div class="section-title">🚨 High Risk Transactions</div>',
        unsafe_allow_html=True,
    )
    high_risk = utils.top_high_risk(result, n=10)
    if high_risk.empty or high_risk[utils.CONFIDENCE_COL].max() == 0:
        st.info("No elevated-risk transactions detected.")
    else:
        for _, row in high_risk.iterrows():
            st.markdown(
                f'<div class="risk-row"><span>Transaction '
                f'{int(row["Transaction"])}</span>'
                f'<span class="risk-prob">Fraud Probability: '
                f'{row[utils.CONFIDENCE_COL]:.2f}%</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div class="section-title">Transaction Records</div>',
        unsafe_allow_html=True,
    )

    controls_left, controls_right = st.columns([1, 2])
    with controls_left:
        view = st.selectbox(
            "View Transactions",
            ["All Transactions", "Fraud Only", "Genuine Only", "High Risk (>90%)"],
        )
    with controls_right:
        search = st.text_input("Search transaction index...", value="")

    view_df = utils.filter_view(result, view).reset_index().rename(
        columns={"index": "Transaction"}
    )

    if search.strip():
        view_df = view_df[
            view_df["Transaction"].astype(str).str.contains(search.strip(), na=False)
        ]

    row_style = JsCode(
        f"""
        function(params) {{
            if (params.data['{utils.PREDICTION_COL}'] === '{utils.FRAUD_LABEL}') {{
                if (params.data['{utils.CONFIDENCE_COL}'] > 95) {{
                    return {{ 'backgroundColor': '#fca5a5', 'color': '#7f1d1d' }};
                }}
                return {{ 'backgroundColor': '#fee2e2', 'color': '#7f1d1d' }};
            }}
            return {{ 'backgroundColor': '#dcfce7', 'color': '#14532d' }};
        }}
        """
    )

    gb = GridOptionsBuilder.from_dataframe(view_df)
    gb.configure_default_column(
        resizable=True, sortable=True, filter=True, editable=False, minWidth=110
    )
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=20)
    gb.configure_column("Transaction", pinned="left", width=130)
    gb.configure_column(utils.PREDICTION_COL, pinned="right", width=130)
    gb.configure_column(utils.CONFIDENCE_COL, pinned="right", width=150)
    grid_options = gb.build()
    grid_options["getRowStyle"] = row_style

    AgGrid(
        view_df,
        gridOptions=grid_options,
        allow_unsafe_jscode=True,
        theme="balham",
        height=520,
        fit_columns_on_grid_load=False,
        update_mode=GridUpdateMode.NO_UPDATE,
        enable_enterprise_modules=False,
    )

    st.caption(f"Showing {len(view_df):,} of {len(result):,} transactions.")

    st.download_button(
        label="⬇ Download Analysis Report",
        data=utils.to_csv_bytes(result),
        file_name="fraud_analysis_report.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
