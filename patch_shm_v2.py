from pathlib import Path

p = Path("streamlit_app.py")
s = p.read_text(encoding="utf-8")

patches = [
    (
        "import streamlit as st\n",
        "import pandas as pd\nimport streamlit as st\n",
    ),
    (
        'DOOR_ADAPTER_PATH = APP_DIR / "subsystems" / "door" / "adapter.py"\n',
        'DOOR_ADAPTER_PATH = APP_DIR / "subsystems" / "door" / "adapter.py"\n'
        'SHM_CODE_DIR = APP_DIR / "subsystems" / "shm" / "code"\n',
    ),
    (
        "    return details, submission, segments\n",
        '''    return details, submission, segments


def run_shm_prediction(uploaded_files):
    """调用 part4_shm.py，不改动其预测逻辑。"""
    code_dir = str(SHM_CODE_DIR)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    import part4_shm

    items = [(Path(f.name).name, f.getvalue()) for f in uploaded_files]
    results = part4_shm.predict_shm_batch(items, include_details=True)
    csv_bytes = part4_shm.export_shm_csv(results)
    return results, csv_bytes
''',
    ),
    (
        '''            "Rail Corrugation: upload one or more CSV files."
''',
        '''            "Rail Corrugation: upload one or more CSV files. "
            "SHM: upload one or more headerless single-column stress CSV files."
''',
    ),
    (
        '''    else:
        st.markdown(
            '<div class="placeholder">Subsystem instructions will appear here after the corresponding module is connected.</div>',
''',
        '''    elif selected_subsystem == "SHM":
        st.markdown(
            '<div class="placeholder">SHM is connected. Upload headerless single-column '
            "stress CSV files to estimate the cumulative fatigue damage of each record.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="placeholder">Subsystem instructions will appear here after the corresponding module is connected.</div>',
''',
    ),
    (
        '''    elif selected_subsystem != "Rail Corrugation":
        st.warning(
''',
        '''    elif selected_subsystem == "SHM":
        try:
            with st.spinner("Running SHM prediction..."):
                shm_results, shm_csv = run_shm_prediction(uploaded_files)
            st.session_state["shm_results"] = shm_results
            st.session_state["shm_csv"] = shm_csv
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop("shm_results", None)
            st.session_state.pop("shm_csv", None)
            st.error(f"SHM prediction failed: {exc}")
    elif selected_subsystem != "Rail Corrugation":
        st.warning(
''',
    ),
    (
        'door_segments = st.session_state.get("door_segments")\n',
        'door_segments = st.session_state.get("door_segments")\n'
        'shm_results = st.session_state.get("shm_results")\n',
    ),
    (
        '''        file_name="rail_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
''',
        '''        file_name="rail_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
elif selected_subsystem == "SHM" and shm_results is not None:
    shm_table = pd.DataFrame(
        [
            {
                "file_id": r["file_id"],
                "prediction": r["prediction"],
                "samples": r["samples"],
                "cycle_count": r["cycle_count"],
                "max_amplitude": r["max_amplitude"],
                "warnings": "; ".join(r["warnings"]),
            }
            for r in shm_results
        ]
    )
    st.caption(
        "prediction: estimated cumulative fatigue damage of each record "
        "(not a failure probability, health percentage, or remaining life)."
    )
    st.dataframe(shm_table, use_container_width=True, hide_index=True)
    for r in shm_results:
        for message in r["warnings"]:
            st.warning(f"{r['file_id']}: {message}")
    st.download_button(
        "Download shm_predictions.csv",
        data=st.session_state["shm_csv"],
        file_name="shm_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
''',
    ),
    (
        '''        st.caption("The Rail predictor did not return fault-score columns.")
else:
''',
        '''        st.caption("The Rail predictor did not return fault-score columns.")
elif selected_subsystem == "SHM" and shm_results is not None:
    chosen = st.selectbox("Select a file", [r["file_id"] for r in shm_results])
    record = next(r for r in shm_results if r["file_id"] == chosen)
    st.caption(
        "Stress signal (original units; x-axis: sample index; "
        "extrema-preserving display envelope)"
    )
    waveform_df = pd.DataFrame(record["waveform"], columns=["sample_index", "stress"])
    st.line_chart(waveform_df.set_index("sample_index"))
    bins_df = pd.DataFrame(record["bins"])
    bins_df.index = [
        f"{i + 1:02d}: {b['low']:.2f}-{b['high']:.2f}"
        for i, b in enumerate(record["bins"])
    ]
    col_count, col_damage = st.columns(2)
    with col_count:
        st.caption("Cycle count by stress-amplitude interval")
        st.bar_chart(bins_df[["count"]])
    with col_damage:
        st.caption("Damage contribution by stress-amplitude interval")
        st.bar_chart(bins_df[["damage"]])
else:
''',
    ),
]

if "SHM_CODE_DIR" in s:
    raise SystemExit("[终止] 文件中已存在 SHM 代码，未写入任何修改")

for i, (old, new) in enumerate(patches, 1):
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[终止] 第 {i} 处锚点匹配次数为 {n}，未写入任何修改")
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print(f"已完成 {len(patches)} 处修改")
EOFcat > patch_shm_v2.py << 'EOF'
from pathlib import Path

p = Path("streamlit_app.py")
s = p.read_text(encoding="utf-8")

patches = [
    (
        "import streamlit as st\n",
        "import pandas as pd\nimport streamlit as st\n",
    ),
    (
        'DOOR_ADAPTER_PATH = APP_DIR / "subsystems" / "door" / "adapter.py"\n',
        'DOOR_ADAPTER_PATH = APP_DIR / "subsystems" / "door" / "adapter.py"\n'
        'SHM_CODE_DIR = APP_DIR / "subsystems" / "shm" / "code"\n',
    ),
    (
        "    return details, submission, segments\n",
        '''    return details, submission, segments


def run_shm_prediction(uploaded_files):
    """调用 part4_shm.py，不改动其预测逻辑。"""
    code_dir = str(SHM_CODE_DIR)
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    import part4_shm

    items = [(Path(f.name).name, f.getvalue()) for f in uploaded_files]
    results = part4_shm.predict_shm_batch(items, include_details=True)
    csv_bytes = part4_shm.export_shm_csv(results)
    return results, csv_bytes
''',
    ),
    (
        '''            "Rail Corrugation: upload one or more CSV files."
''',
        '''            "Rail Corrugation: upload one or more CSV files. "
            "SHM: upload one or more headerless single-column stress CSV files."
''',
    ),
    (
        '''    else:
        st.markdown(
            '<div class="placeholder">Subsystem instructions will appear here after the corresponding module is connected.</div>',
''',
        '''    elif selected_subsystem == "SHM":
        st.markdown(
            '<div class="placeholder">SHM is connected. Upload headerless single-column '
            "stress CSV files to estimate the cumulative fatigue damage of each record.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="placeholder">Subsystem instructions will appear here after the corresponding module is connected.</div>',
''',
    ),
    (
        '''    elif selected_subsystem != "Rail Corrugation":
        st.warning(
''',
        '''    elif selected_subsystem == "SHM":
        try:
            with st.spinner("Running SHM prediction..."):
                shm_results, shm_csv = run_shm_prediction(uploaded_files)
            st.session_state["shm_results"] = shm_results
            st.session_state["shm_csv"] = shm_csv
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop("shm_results", None)
            st.session_state.pop("shm_csv", None)
            st.error(f"SHM prediction failed: {exc}")
    elif selected_subsystem != "Rail Corrugation":
        st.warning(
''',
    ),
    (
        'door_segments = st.session_state.get("door_segments")\n',
        'door_segments = st.session_state.get("door_segments")\n'
        'shm_results = st.session_state.get("shm_results")\n',
    ),
    (
        '''        file_name="rail_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
''',
        '''        file_name="rail_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
elif selected_subsystem == "SHM" and shm_results is not None:
    shm_table = pd.DataFrame(
        [
            {
                "file_id": r["file_id"],
                "prediction": r["prediction"],
                "samples": r["samples"],
                "cycle_count": r["cycle_count"],
                "max_amplitude": r["max_amplitude"],
                "warnings": "; ".join(r["warnings"]),
            }
            for r in shm_results
        ]
    )
    st.caption(
        "prediction: estimated cumulative fatigue damage of each record "
        "(not a failure probability, health percentage, or remaining life)."
    )
    st.dataframe(shm_table, use_container_width=True, hide_index=True)
    for r in shm_results:
        for message in r["warnings"]:
            st.warning(f"{r['file_id']}: {message}")
    st.download_button(
        "Download shm_predictions.csv",
        data=st.session_state["shm_csv"],
        file_name="shm_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
else:
''',
    ),
    (
        '''        st.caption("The Rail predictor did not return fault-score columns.")
else:
''',
        '''        st.caption("The Rail predictor did not return fault-score columns.")
elif selected_subsystem == "SHM" and shm_results is not None:
    chosen = st.selectbox("Select a file", [r["file_id"] for r in shm_results])
    record = next(r for r in shm_results if r["file_id"] == chosen)
    st.caption(
        "Stress signal (original units; x-axis: sample index; "
        "extrema-preserving display envelope)"
    )
    waveform_df = pd.DataFrame(record["waveform"], columns=["sample_index", "stress"])
    st.line_chart(waveform_df.set_index("sample_index"))
    bins_df = pd.DataFrame(record["bins"])
    bins_df.index = [
        f"{i + 1:02d}: {b['low']:.2f}-{b['high']:.2f}"
        for i, b in enumerate(record["bins"])
    ]
    col_count, col_damage = st.columns(2)
    with col_count:
        st.caption("Cycle count by stress-amplitude interval")
        st.bar_chart(bins_df[["count"]])
    with col_damage:
        st.caption("Damage contribution by stress-amplitude interval")
        st.bar_chart(bins_df[["damage"]])
else:
''',
    ),
]

if "SHM_CODE_DIR" in s:
    raise SystemExit("[终止] 文件中已存在 SHM 代码，未写入任何修改")

for i, (old, new) in enumerate(patches, 1):
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"[终止] 第 {i} 处锚点匹配次数为 {n}，未写入任何修改")
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print(f"已完成 {len(patches)} 处修改")
