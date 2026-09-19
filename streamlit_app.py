from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st


SUBSYSTEMS = ["Door", "ACV", "Rail Corrugation", "SHM"]

APP_DIR = Path(__file__).resolve().parent
RAIL_CODE_DIR = APP_DIR / "subsystems" / "rail_corrugation" / "code"
RAIL_MODEL_PATH = (
    APP_DIR / "subsystems" / "rail_corrugation" / "model" / "rail_model.joblib"
)
ACV_CODE_DIR = APP_DIR / "subsystems" / "acv" / "code"


st.set_page_config(
    page_title="Train Condition Monitoring System",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #123b63 0%, #1d6fa5 100%);
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 24px rgba(18, 59, 99, 0.18);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
    }

    .hero p {
        margin: 0.55rem 0 0;
        color: #e6f3ff;
        font-size: 1rem;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 650;
        color: #17324d;
        margin: 0.4rem 0 0.7rem;
    }

    .placeholder {
        padding: 1.2rem;
        border: 1px solid #dce7f0;
        border-radius: 14px;
        background: #f8fbfd;
        color: #526575;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_rail_prediction(uploaded_files):
    """调用原版 Rail Corrugation predict.py，不改动其预测逻辑。"""
    code_dir = str(RAIL_CODE_DIR)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    import predict as rail_predict

    with tempfile.TemporaryDirectory(prefix="rail_prediction_") as temp_dir:
        temp_dir = Path(temp_dir)
        input_paths = []
        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            input_path = temp_dir / safe_name
            input_path.write_bytes(uploaded_file.getbuffer())
            input_paths.append(input_path)

        bundle = rail_predict.load_bundle(RAIL_MODEL_PATH)
        details = rail_predict.predict_files(input_paths, bundle)

    submission = details[["file_id", "prediction"]].copy()
    return details, submission


def run_acv_prediction(uploaded_files):
    """调用原版 acv_rank.py（rank_workbook），不改动其评分与排序逻辑。"""
    import pandas as pd

    code_dir = str(ACV_CODE_DIR)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    import acv_rank

    submission_rows = []
    detail_rows = []
    with tempfile.TemporaryDirectory(prefix="acv_prediction_") as temp_dir:
        temp_dir = Path(temp_dir)
        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            if Path(safe_name).suffix.lower() != ".xlsx":
                raise ValueError(f"ACV input must be an .xlsx file: {safe_name}")
            input_path = temp_dir / safe_name
            input_path.write_bytes(uploaded_file.getbuffer())

            try:
                result = acv_rank.rank_workbook(input_path)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{safe_name}: {exc}") from exc

            # ranked_car_ids 保留表头中的原始编号格式（如 "03"），符合提交规范。
            ranked_ids = result.ranked_car_ids
            submission_rows.append(
                {"file_id": safe_name, "ranked_cars": "|".join(ranked_ids)}
            )
            rank_of = {car_id: pos for pos, car_id in enumerate(ranked_ids, start=1)}
            for i, car_id in enumerate(result.car_ids):
                detail_rows.append(
                    {
                        "file_id": safe_name,
                        "car": car_id,
                        "rank": rank_of[car_id],
                        # 原程序得分为 2 倍温度刻度，此处除以 2 还原为 °C。
                        "mean_excess_temp_C": result.scores[i] / 2,
                        "eligible_samples": result.valid_counts[i],
                    }
                )

    submission = pd.DataFrame(submission_rows, columns=["file_id", "ranked_cars"])
    details = pd.DataFrame(detail_rows).sort_values(["file_id", "rank"])
    return details, submission


st.markdown(
    """
    <div class="hero">
        <h1>Train Condition Monitoring System</h1>
        <p>Select a subsystem, upload sensor files, and view the prediction results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="section-title">Select subsystem</div>', unsafe_allow_html=True)
selected_subsystem = st.radio(
    "Subsystem",
    SUBSYSTEMS,
    horizontal=True,
    label_visibility="collapsed",
)


left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.markdown('<div class="section-title">Input files</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        accept_multiple_files=True,
        label_visibility="collapsed",
        help=(
            "Rail Corrugation: upload CSV files. ACV: upload .xlsx workbooks "
            "(first worksheet, headers in the first row)."
        ),
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")
        for uploaded_file in uploaded_files:
            st.write(f"• {uploaded_file.name}")

    run_clicked = st.button(
        "Run Prediction",
        type="primary",
        use_container_width=True,
    )

with right:
    st.markdown('<div class="section-title">Current selection</div>', unsafe_allow_html=True)
    st.info(f"Selected subsystem: **{selected_subsystem}**")
    if selected_subsystem == "Rail Corrugation":
        st.markdown(
            '<div class="placeholder">Rail Corrugation is connected. '
            "Upload CSV files and run the original prediction program.</div>",
            unsafe_allow_html=True,
        )
    elif selected_subsystem == "ACV":
        st.markdown(
            '<div class="placeholder">ACV is connected. '
            "Upload .xlsx workbooks to rank cars by refrigerant-leak likelihood.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="placeholder">Subsystem instructions will appear here after the corresponding module is connected.</div>',
            unsafe_allow_html=True,
        )


st.markdown('<div class="section-title">Prediction output</div>', unsafe_allow_html=True)

if run_clicked:
    if not uploaded_files:
        st.error("Please upload at least one input file.")
    elif selected_subsystem == "ACV":
        try:
            with st.spinner("Running ACV ranking..."):
                acv_detail, acv_submission = run_acv_prediction(uploaded_files)
            st.session_state["acv_detail_df"] = acv_detail
            st.session_state["acv_submission_df"] = acv_submission
        except Exception as exc:  # noqa: BLE001
            st.error(f"ACV prediction failed: {exc}")
    elif selected_subsystem != "Rail Corrugation":
        st.warning(
            "The application framework is ready, but the selected subsystem has not been connected yet."
        )
    else:
        try:
            with st.spinner("Running Rail Corrugation prediction..."):
                detail_df, submission_df = run_rail_prediction(uploaded_files)
            st.session_state["rail_detail_df"] = detail_df
            st.session_state["rail_submission_df"] = submission_df
        except Exception as exc:  # noqa: BLE001
            st.error(f"Rail Corrugation prediction failed: {exc}")


detail_df = st.session_state.get("rail_detail_df")
submission_df = st.session_state.get("rail_submission_df")

acv_detail_df = st.session_state.get("acv_detail_df")
acv_submission_df = st.session_state.get("acv_submission_df")

if selected_subsystem == "ACV" and acv_submission_df is not None:
    st.dataframe(acv_submission_df, use_container_width=True, hide_index=True)
    st.dataframe(acv_detail_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download acv_predictions.csv",
        data=acv_submission_df.to_csv(index=False).encode("utf-8"),
        file_name="acv_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
elif selected_subsystem == "Rail Corrugation" and detail_df is not None:
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download rail_predictions.csv",
        data=submission_df.to_csv(index=False).encode("utf-8"),
        file_name="rail_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
    st.markdown(
        '<div class="placeholder">Prediction results, downloadable output files, and subsystem-specific visualizations will appear here.</div>',
        unsafe_allow_html=True,
    )


st.markdown('<div class="section-title">Visualization</div>', unsafe_allow_html=True)

if selected_subsystem == "ACV" and acv_detail_df is not None:
    st.bar_chart(
        acv_detail_df.pivot(index="car", columns="file_id", values="mean_excess_temp_C")
    )
elif selected_subsystem == "Rail Corrugation" and detail_df is not None:
    chart_columns = [
        column
        for column in ["p_side1_fault", "p_side2_fault"]
        if column in detail_df.columns
    ]
    if chart_columns:
        st.bar_chart(detail_df.set_index("file_id")[chart_columns])
    else:
        st.caption("The Rail predictor did not return fault-score columns.")
else:
    st.markdown(
        '<div class="placeholder">Visualization area reserved for the selected subsystem.</div>',
        unsafe_allow_html=True,
    )
