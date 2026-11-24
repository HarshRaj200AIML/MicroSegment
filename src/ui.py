"""Streamlit pages and shared application state."""

from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

from .clustering import (
    ClusteringResult,
    clustering_memory_limit_bytes,
    elbow_scores,
    fit_dbscan,
    fit_kmeans,
    fit_kmedoids,
    pca_projection,
)
from .data_cleaning import preprocess_data, quality_report
from .data_loader import dataset_summary, discover_datasets, infer_column_roles, load_dataset
from .eda import correlation_matrix, numeric_summary
from .evaluation import evaluate_clustering
from .feature_engineering import default_clustering_features, feature_audit
from .profiling import cluster_profile, kmeans_stability
from .reporting import (
    build_assignment_csv_payload,
    build_assignment_metadata_payload,
    build_assignment_zip_payload,
    save_outputs,
)
from .provenance import (
    build_assignment_manifest,
    build_model_manifest,
    create_model_run_id,
    dataset_fingerprint,
    preprocessing_metadata,
    schema_fingerprint,
    validate_assignment_provenance,
    validate_model_manifest,
)
from .utils import (
    PROJECT_ROOT,
    ExportValidationError,
    MicroSegmentError,
    ModelProvenanceError,
    RANDOM_SEED,
    ensure_project_dirs,
    get_logger,
)
from .visualization import (
    cluster_scatter,
    distribution_chart,
    heatmap,
    missing_chart,
    registration_3d_chart,
    registration_mix_chart,
    registration_state_summary,
    radar_chart,
    state_registration_bubble_chart,
    state_registration_choropleth,
    state_registration_map,
    top_states_chart,
)


LOGGER = get_logger("microsegment.ui")


PAGES = [
    ("app.py", "Dashboard", ":material/space_dashboard:"),
    ("pages/1_Dataset_Explorer.py", "Dataset Explorer", ":material/database:"),
    ("pages/2_Data_Preprocessing.py", "Preprocessing", ":material/tune:"),
    ("pages/3_Exploratory_Analysis.py", "EDA Analytics", ":material/query_stats:"),
    ("pages/4_Clustering_Studio.py", "Clustering Studio", ":material/hub:"),
    ("pages/6_Visual_Analytics.py", "3D Visualization", ":material/view_in_ar:"),
    ("pages/5_Entrepreneur_Segments.py", "Cluster Profiles", ":material/groups:"),
    ("pages/8_Model_Evaluation.py", "Model Evaluation", ":material/monitoring:"),
    ("pages/9_New_Assignments.py", "New Assignments", ":material/person_add:"),
    ("pages/10_Provenance.py", "Provenance", ":material/verified:"),
    ("pages/7_Reports.py", "Export & Reports", ":material/download:"),
    ("pages/11_Settings.py", "Settings", ":material/settings:"),
]


@st.cache_data(show_spinner=False)
def _read_path(path: str, modified: float):
    del modified
    return load_dataset(path)


@st.cache_resource(show_spinner=False)
def _read_saved_model(path: str, modified: float):
    """Read a saved estimator once; callers still validate it against active rows."""
    del modified
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def _read_saved_model_config(path: str, modified: float) -> dict:
    del modified
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def _read_saved_segmented_rows(path: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path, low_memory=False)


def _load_active_dataset():
    uploaded = st.session_state.get("microsegment_upload")
    if uploaded is not None:
        uploaded.seek(0)
        return load_dataset(uploaded)
    datasets = discover_datasets(PROJECT_ROOT)
    if not datasets:
        return None
    selected = st.session_state.get("microsegment_source")
    paths = {str(path): path for path in datasets}
    if selected not in paths:
        preferred = next((path for path in datasets if path.suffix.lower() == ".dta"), None)
        if preferred is None:
            preferred = next((path for path in datasets if path.suffix.lower() == ".csv"), datasets[0])
        selected = str(preferred)
    path = paths[selected]
    return _read_path(str(path), path.stat().st_mtime)


def _sidebar(model_ready: bool = False, model_run_id: str | None = None) -> None:
    st.sidebar.markdown(
        "<div class='brand-shell'><div class='brand-mark'>MS</div>"
        "<div><div class='brand-name'>MicroSegment</div>"
        "<div class='brand-sub'>Research workbench</div></div></div>",
        unsafe_allow_html=True,
    )
    is_restored = bool((st.session_state.get("cluster_manifest") or {}).get("legacy_artifact"))
    status_copy = "Saved model restored" if is_restored else "Validated model ready" if model_ready else "Dataset workspace ready"
    status_class = "status-pulse" if model_ready else "status-pulse status-pulse-muted"
    st.sidebar.markdown(
        f"<div class='sidebar-status'><span class='{status_class}'></span>"
        f"{escape(status_copy)}</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    st.sidebar.markdown("<div class='nav-label'>Navigation</div>", unsafe_allow_html=True)
    for page_path, label, icon in PAGES:
        st.sidebar.page_link(page_path, label=label, icon=icon)
    st.sidebar.markdown(
        "<div class='model-status-panel'>"
        "<div class='model-status-title'>Model status</div>"
        f"<div class='model-status-value'>{'RESTORED' if is_restored else 'READY' if model_ready else 'AWAITING MODEL'}</div>"
        f"<div class='model-status-detail'>{escape(model_run_id or 'No verified model run in this session')}</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Results describe the loaded records. They do not establish individual success, creditworthiness, or causal impact.")


def _get_data():
    try:
        return _load_active_dataset()
    except Exception as exc:
        LOGGER.exception("Dataset load failed")
        if isinstance(exc, MicroSegmentError):
            st.error(f"Could not load this dataset: {exc}")
        else:
            st.error("Could not load this dataset. Check the file format and try again.")
        return None


def _page_header(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        "<section class='page-hero'>"
        f"<div class='eyebrow'>{escape(eyebrow)}</div>"
        f"<h1>{escape(title)}</h1>"
        f"<p class='page-intro'>{escape(subtitle)}</p>"
        "</section>",
        unsafe_allow_html=True,
    )


_CLUSTER_COLORS = ["#21B7FF", "#A855F7", "#00D98A", "#FF9B21", "#FB7185", "#7DD3FC", "#E879F9"]


def _kpi(label: str, value: str, detail: str = "", *, tone: str = "cyan", icon: str = "") -> None:
    st.markdown(
        f"<div class='metric-card metric-card-{escape(tone)}'>"
        f"<div class='metric-icon'>{icon}</div>"
        f"<div class='metric-label'>{escape(label)}</div>"
        f"<div class='metric-value'>{escape(value)}</div>"
        f"<div class='metric-detail'>{escape(detail)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _format_metric(value: object, digits: int = 3) -> str:
    """Format an available model metric without inventing an empty-state value."""
    if value is None or pd.isna(value):
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _cluster_label(value: object) -> str:
    return "Noise" if int(value) == -1 else f"Cluster {int(value) + 1}"


def _cluster_color(value: object) -> str:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return "#8FA2C8"
    return "#64748B" if numeric == -1 else _CLUSTER_COLORS[numeric % len(_CLUSTER_COLORS)]


def _cluster_distribution(labels: object | None) -> pd.DataFrame:
    """Return real assignment counts and percentages for dashboard surfaces."""
    if labels is None:
        return pd.DataFrame(columns=["cluster", "label", "records", "percent", "color"])
    values = pd.Series(np.asarray(labels), name="cluster").value_counts().sort_index()
    total = int(values.sum())
    result = values.rename("records").reset_index()
    result["label"] = result["cluster"].map(_cluster_label)
    result["percent"] = result["records"].div(total).mul(100) if total else 0.0
    result["color"] = result["cluster"].map(_cluster_color)
    return result


def _model_timestamp(manifest: dict | None) -> str:
    raw = (manifest or {}).get("created_at")
    if not raw:
        raw = (manifest or {}).get("artifact_saved_at")
    if not raw:
        return "No model run yet"
    try:
        timestamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return timestamp.astimezone(timezone.utc).strftime("%d %b %Y · %H:%M UTC")
    except ValueError:
        return str(raw)


def _top_header(info: object, result: object | None, manifest: dict | None) -> None:
    """Render model metadata from live Streamlit state, never fabricated placeholders."""
    algorithm = getattr(result, "algorithm", None) or "No active algorithm"
    choices = ["K-Means", "DBSCAN", "K-Medoids (optional)"]
    if "dashboard_algorithm_selection" not in st.session_state:
        default_algorithm = algorithm if algorithm in choices else "K-Medoids (optional)" if algorithm == "K-Medoids" else "K-Means"
        st.session_state.dashboard_algorithm_selection = default_algorithm
    run_id = (manifest or {}).get("model_run_id") or ("Saved artifact · run ID unavailable" if (manifest or {}).get("legacy_artifact") else "No model run")
    is_restored = bool((manifest or {}).get("legacy_artifact"))
    state = "RESTORED" if result is not None and is_restored else "READY" if result is not None else "MODEL NEEDED"
    source_name = Path(getattr(info, "source", "")).name or "No active source"
    source_context = "District-level registration aggregates" if "district" in source_name.casefold() else "Active registration dataset"
    left, right = st.columns([1.12, 1], vertical_alignment="center")
    with left:
        st.markdown(
            "<div class='command-header'>"
            "<div class='command-product'><span class='command-orb'></span>MicroSegment</div>"
            f"<div class='command-context'>Rural registration analytics · {escape(source_context)}</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with right:
        with st.container(horizontal=True, horizontal_alignment="right", gap="small"):
            st.selectbox(
                "Algorithm",
                choices,
                key="dashboard_algorithm_selection",
                label_visibility="collapsed",
                help=f"Active fitted model: {algorithm}" if result is not None else "Choose an algorithm, then fit it from the dashboard or Clustering Studio.",
            )
            st.badge(state, icon=":material/check_circle:" if result is not None else ":material/schedule:", color="violet" if is_restored else "green" if result is not None else "orange")
            if st.button("Refresh", icon=":material/refresh:", key="microsegment_refresh", help="Rerun the dashboard with the current dataset and session state"):
                st.rerun()
            with st.popover("Workspace", icon=":material/account_circle:", help="View the active local-workspace context"):
                st.caption("Local analysis workspace")
                st.write(f"Active source: `{source_name}`")
                st.write(f"Model run: `{run_id}`")
    run_label = "RUN ID" if (manifest or {}).get("model_run_id") else "MODEL SOURCE"
    updated_label = "LAST UPDATED" if (manifest or {}).get("created_at") else "ARTIFACT FILE" if is_restored else "LAST UPDATED"
    st.markdown(
        "<div class='model-command-strip'>"
        f"<span>{run_label}&nbsp;&nbsp;<strong>{escape(run_id)}</strong></span>"
        f"<span>{updated_label}&nbsp;&nbsp;<strong>{escape(_model_timestamp(manifest))}</strong></span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _coverage(labels: object | None, row_count: int) -> tuple[float | None, int]:
    if labels is None or not row_count:
        return None, 0
    values = np.asarray(labels)
    assigned = int(np.sum(values != -1))
    return assigned / row_count * 100, assigned


def _coverage_card(coverage: float | None, assigned: int, row_count: int) -> None:
    value = "N/A" if coverage is None else f"{coverage:.1f}%"
    progress = 0 if coverage is None else min(100, max(0, coverage))
    detail = "No active model" if coverage is None else f"{assigned:,} of {row_count:,} records assigned"
    st.markdown(
        "<div class='metric-card metric-card-lime metric-card-coverage'>"
        f"<div class='coverage-ring' style='--coverage:{progress:.2f}%;'><span>{escape(value)}</span></div>"
        "<div class='metric-label'>Coverage</div>"
        f"<div class='metric-detail'>{escape(detail)}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _neon_plot_layout(figure: go.Figure, *, height: int) -> go.Figure:
    """Use a consistent presentation layer while leaving the underlying data untouched."""
    figure.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,14,31,0.5)",
        font={"family": "Manrope, Segoe UI, sans-serif", "color": "#DCE8FF"},
        margin={"l": 22, "r": 18, "t": 28, "b": 24},
        legend={
            "bgcolor": "rgba(9, 16, 35, 0.72)",
            "bordercolor": "rgba(103, 235, 255, 0.18)",
            "borderwidth": 1,
            "font": {"size": 11},
        },
    )
    return figure


def _pca_3d_chart(matrix: object, labels: object) -> tuple[go.Figure | None, str]:
    """Project a fitted model's feature matrix for display only.

    The model is always fit on every selected record. A deterministic subset is
    used only for browser rendering when there are many records; no points or
    clusters are generated by this view.
    """
    if matrix is None or labels is None:
        return None, "Fit a model to view its PCA projection."
    row_count, feature_count = matrix.shape
    components = min(3, row_count, feature_count)
    if components < 2:
        return None, "At least two transformed features are needed for a PCA visualization."
    dense_bytes = int(row_count * feature_count * 8)
    if dense_bytes > clustering_memory_limit_bytes():
        return None, "The full 3D PCA projection exceeds this workspace's safe memory limit. Use the 2D projection in 3D Visualization instead."
    try:
        values = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        projection = PCA(n_components=components, random_state=RANDOM_SEED).fit_transform(values)
    except Exception as exc:
        return None, f"PCA projection is unavailable: {exc}"

    display_limit = 3000
    if len(projection) > display_limit:
        sample_index = np.linspace(0, len(projection) - 1, display_limit, dtype=int)
        projection = projection[sample_index]
        rendered_labels = np.asarray(labels)[sample_index]
        note = f"Displaying a deterministic sample of {display_limit:,} of {row_count:,} modeled records."
    else:
        rendered_labels = np.asarray(labels)
        note = f"Displaying all {row_count:,} modeled records."

    data = pd.DataFrame({
        "PCA 1": projection[:, 0],
        "PCA 2": projection[:, 1],
        "Cluster": [_cluster_label(value) for value in rendered_labels],
        "cluster_color": [_cluster_color(value) for value in rendered_labels],
    })
    if components >= 3:
        data["PCA 3"] = projection[:, 2]
        figure = px.scatter_3d(
            data,
            x="PCA 1",
            y="PCA 2",
            z="PCA 3",
            color="Cluster",
            color_discrete_map={label: color for label, color in zip(data["Cluster"], data["cluster_color"])},
            hover_data={"cluster_color": False},
            opacity=0.78,
        )
        figure.update_traces(marker={"size": 3.4, "line": {"width": 0.25, "color": "rgba(225,251,255,0.85)"}})
        figure.update_scenes(
            bgcolor="rgba(0,0,0,0)",
            xaxis={"backgroundcolor": "#0B142B", "gridcolor": "rgba(126,176,255,0.16)", "showbackground": True},
            yaxis={"backgroundcolor": "#0B142B", "gridcolor": "rgba(126,176,255,0.16)", "showbackground": True},
            zaxis={"backgroundcolor": "#0B142B", "gridcolor": "rgba(126,176,255,0.16)", "showbackground": True},
            camera={"eye": {"x": 1.42, "y": 1.35, "z": 0.88}},
        )
        return _neon_plot_layout(figure, height=370), note

    figure = px.scatter(
        data,
        x="PCA 1",
        y="PCA 2",
        color="Cluster",
        color_discrete_map={label: color for label, color in zip(data["Cluster"], data["cluster_color"])},
        hover_data={"cluster_color": False},
        opacity=0.8,
    )
    figure.update_traces(marker={"size": 7, "line": {"width": 0.35, "color": "rgba(225,251,255,0.85)"}})
    figure.update_xaxes(gridcolor="rgba(126,176,255,0.12)", zerolinecolor="rgba(126,176,255,0.18)")
    figure.update_yaxes(gridcolor="rgba(126,176,255,0.12)", zerolinecolor="rgba(126,176,255,0.18)")
    return _neon_plot_layout(figure, height=370), f"2D PCA fallback. {note}"


def _distribution_donut(distribution: pd.DataFrame) -> go.Figure | None:
    if distribution.empty:
        return None
    figure = go.Figure(
        go.Pie(
            labels=distribution["label"],
            values=distribution["records"],
            hole=0.68,
            domain={"x": [0.0, 0.57], "y": [0.0, 1.0]},
            marker={"colors": distribution["color"], "line": {"color": "#091126", "width": 2}},
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,} records<br>%{percent}<extra></extra>",
        )
    )
    figure.add_annotation(
        text=f"<b>{int(distribution['records'].sum()):,}</b><br><span style='font-size:10px;color:#9DB0D0'>MODELED</span>",
        x=0.285,
        y=0.5,
        showarrow=False,
        font={"color": "#F3F7FF", "size": 18},
    )
    _neon_plot_layout(figure, height=190)
    figure.update_layout(
        showlegend=True,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        legend={"orientation": "v", "y": 0.5, "x": 0.61, "xanchor": "left", "yanchor": "middle", "font": {"size": 10}},
    )
    return figure


def _profile_bar_chart(profile: pd.DataFrame, feature: str) -> go.Figure | None:
    column = f"{feature}__mean"
    if profile.empty or column not in profile.columns:
        return None
    values = profile.reset_index()[["cluster", column]].copy()
    values["Cluster"] = values["cluster"].map(_cluster_label)
    values["color"] = values["cluster"].map(_cluster_color)
    figure = go.Figure()
    for _, row in values.iterrows():
        figure.add_trace(
            go.Bar(
                name=row["Cluster"],
                x=[row["Cluster"]],
                y=[row[column]],
                marker={"color": row["color"], "line": {"color": "rgba(226,250,255,0.65)", "width": 0.35}},
                hovertemplate=f"<b>{row['Cluster']}</b><br>{escape(feature)} mean: %{{y:,.3f}}<extra></extra>",
            )
        )
    figure.update_layout(showlegend=False, bargap=0.38, yaxis_title=f"Observed mean · {feature.replace('_', ' ').title()}")
    figure.update_yaxes(gridcolor="rgba(126,176,255,0.12)", zerolinecolor="rgba(126,176,255,0.18)")
    return _neon_plot_layout(figure, height=250)


def _insight_items(result: object | None, distribution: pd.DataFrame, manifest: dict | None) -> list[tuple[str, str]]:
    if result is None:
        return [("Model not fitted", "Fit a model in Clustering Studio to generate results-backed insights.")]
    metrics = getattr(result, "metrics", {})
    items = [("Detected clusters", f"{int(metrics.get('n_clusters') or 0)} non-noise cluster(s) are present in the active fitted model.")]
    silhouette = metrics.get("silhouette")
    if silhouette is not None:
        items.append(("Separation diagnostic", f"Silhouette score: {float(silhouette):.3f}. Higher values indicate more separation in the selected feature space."))
    if not distribution.empty:
        largest = distribution.loc[distribution["percent"].idxmax()]
        items.append(("Distribution", f"{largest['label']} contains {int(largest['records']):,} records ({float(largest['percent']):.1f}% of modeled rows)."))
    noise = int(metrics.get("noise_points") or 0)
    if noise:
        total = int(distribution["records"].sum()) if not distribution.empty else 0
        percentage = noise / total * 100 if total else 0
        items.append(("Noise handling", f"DBSCAN marked {noise:,} record(s) as noise ({percentage:.1f}%); they are shown separately from clusters."))
    preprocessing = (manifest or {}).get("preprocessing", {})
    if preprocessing:
        items.append(("Preprocessing", "Numeric values use median imputation and standard scaling; categorical values use most-frequent imputation and one-hot encoding."))
    return items[:4]


def _render_insights(result: object | None, distribution: pd.DataFrame, manifest: dict | None) -> None:
    st.markdown("<div class='panel-heading'><span>Key insights</span><small>RESULTS-BACKED</small></div>", unsafe_allow_html=True)
    for title, copy in _insight_items(result, distribution, manifest):
        st.markdown(
            "<div class='insight-item'><span class='insight-mark'></span>"
            f"<div><strong>{escape(title)}</strong><p>{escape(copy)}</p></div></div>",
            unsafe_allow_html=True,
        )


def _render_cluster_flow(distribution: pd.DataFrame, result: object | None) -> None:
    """Show static, model-derived assignment routing; no claimed real-time activity."""
    st.markdown("<div class='panel-heading'><span>Live cluster movement</span><small>STATIC MODEL RESULT</small></div>", unsafe_allow_html=True)
    if result is None or distribution.empty:
        st.markdown("<div class='empty-flow'>Fit a model to display the verified record-to-cluster assignment summary.</div>", unsafe_allow_html=True)
        return
    total = int(distribution["records"].sum())
    process_label = "Centroid assignment" if getattr(result, "algorithm", "") == "K-Means" else "Density assignment"
    st.markdown(
        "<div class='flow-stages'>"
        f"<div><span class='flow-icon'>01</span><strong>{total:,}</strong><small>Input records</small></div>"
        f"<i></i><div><span class='flow-icon'>02</span><strong>{escape(process_label)}</strong><small>{escape(getattr(result, 'algorithm', 'Model'))}</small></div>"
        f"<i></i><div><span class='flow-icon'>03</span><strong>{len(distribution):,}</strong><small>Assigned labels</small></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    for _, row in distribution.iterrows():
        st.markdown(
            "<div class='flow-row'>"
            f"<span>{escape(str(row['label']))}</span>"
            f"<div class='flow-track'><div style='width:{float(row['percent']):.3f}%;background:{escape(str(row['color']))};'></div></div>"
            f"<strong>{int(row['records']):,}</strong><small>{float(row['percent']):.1f}%</small>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.caption("This is a static summary of the active fitted model. The dashboard does not claim real-time record movement when no assignment operation is running.")


def _render_performance(result: object | None) -> None:
    st.markdown("<div class='panel-heading'><span>Model performance</span><small>CURRENT RUN</small></div>", unsafe_allow_html=True)
    if result is None:
        st.markdown("<div class='empty-panel'>No performance diagnostics are available until a model is fitted.</div>", unsafe_allow_html=True)
        return
    metrics = getattr(result, "metrics", {})
    entries = [
        ("Silhouette", _format_metric(metrics.get("silhouette"))),
        ("Davies–Bouldin", _format_metric(metrics.get("davies_bouldin"))),
        ("Calinski–Harabasz", _format_metric(metrics.get("calinski_harabasz"), 1)),
    ]
    if getattr(result, "inertia", None) is not None:
        entries.insert(0, ("Inertia", _format_metric(result.inertia, 1)))
    for label, value in entries:
        st.markdown(
            "<div class='performance-row'>"
            f"<span>{escape(label)}</span><strong>{escape(value)}</strong>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.caption("Metrics are current-run diagnostics. No historical curve is shown because no parameter sweep or prior-run history is stored.")


def _render_profile_comparison(frame: pd.DataFrame, labels: object | None, *, key_suffix: str) -> None:
    st.markdown("<div class='panel-heading'><span>Cluster profiles</span><small>OBSERVED MEANS</small></div>", unsafe_allow_html=True)
    if labels is None:
        st.markdown("<div class='empty-panel'>Fit a model to compare observed feature statistics by cluster.</div>", unsafe_allow_html=True)
        return
    profile, _ = cluster_profile(frame, labels)
    features = [column.removesuffix("__mean") for column in profile.columns if column.endswith("__mean")]
    if not features:
        st.markdown("<div class='empty-panel'>No numeric profile statistics are available for the selected features.</div>", unsafe_allow_html=True)
        return
    selected = st.selectbox(
        "Profile feature",
        features,
        format_func=lambda item: item.replace("_", " ").title(),
        key=f"dashboard_profile_feature_{key_suffix}",
        label_visibility="collapsed",
    )
    chart = _profile_bar_chart(profile, selected)
    if chart is not None:
        st.plotly_chart(chart, width="stretch", config={"displaylogo": False, "responsive": True})


def _dashboard_fit_button(frame: pd.DataFrame, source_name: str, algorithm: str, *, key: str) -> None:
    candidates = default_clustering_features(frame)
    selected = [column for column in st.session_state.get("cluster_features", candidates) if column in candidates]
    if not selected:
        selected = candidates
    fit_label = f"Fit {algorithm}"
    if algorithm == "K-Means":
        fit_label += " · K=4"
    if st.button(
        fit_label,
        type="primary",
        icon=":material/play_arrow:",
        disabled=not selected or len(frame) < 3,
        key=key,
    ):
        try:
            with st.spinner("Fitting the selected model on the active dataset…"):
                _fit_model_into_session(
                    frame,
                    source_name,
                    selected,
                    algorithm,
                    n_clusters=min(4, max(2, len(frame) - 1)),
                )
            st.rerun()
        except Exception as exc:
            st.error(f"Model fitting failed: {exc}")
    if selected:
        st.caption("Uses the current feature selection. Set K, DBSCAN density parameters, or optional K-Medoids settings in Clustering Studio.")


def _render_overview(
    frame: pd.DataFrame,
    summary: dict,
    result: object | None,
    manifest: dict | None,
    source_name: str,
) -> None:
    """Compose the main dashboard from the active dataset and fitted-model state."""
    if result is None:
        st.info("No saved model passed the active dataset check. Choose an algorithm above and fit it to populate the dashboard with current results.", icon=":material/info:")
    selected_algorithm = st.session_state.get("dashboard_algorithm_selection", "K-Means")
    if result is not None:
        active_algorithm = "K-Medoids (optional)" if result.algorithm == "K-Medoids" else result.algorithm
        if selected_algorithm != active_algorithm:
            with st.container(border=True):
                st.markdown(
                    f"<div class='fit-prompt'><span>Selected algorithm: <strong>{escape(selected_algorithm)}</strong></span>"
                    f"<small>Active model: {escape(result.algorithm)}. Fitting creates a new in-session run.</small></div>",
                    unsafe_allow_html=True,
                )
                _dashboard_fit_button(frame, st.session_state.get("cluster_source", ""), selected_algorithm, key="dashboard_refit_model")

    coverage, assigned = _coverage(getattr(result, "labels", None), len(frame))
    metrics = getattr(result, "metrics", {}) if result is not None else {}
    kpis = st.columns(4, gap="small")
    with kpis[0]:
        _kpi("Total records", f"{len(frame):,}", "Active dataset rows", tone="cyan", icon="<span class='metric-glyph'>◫</span>")
    with kpis[1]:
        cluster_value = "N/A" if result is None else str(int(metrics.get("n_clusters") or 0))
        cluster_detail = "Fit a model to detect clusters" if result is None else f"{getattr(result, 'algorithm', 'Active model')} · non-noise labels"
        _kpi("Clusters", cluster_value, cluster_detail, tone="blue", icon="<span class='metric-glyph'>◇</span>")
    with kpis[2]:
        _kpi("Silhouette score", _format_metric(metrics.get("silhouette")), "Current-run diagnostic" if result is not None else "No active model", tone="violet", icon="<span class='metric-glyph'>◌</span>")
    with kpis[3]:
        _coverage_card(coverage, assigned, len(frame))

    distribution = _cluster_distribution(getattr(result, "labels", None))
    primary, side = st.columns([1.08, 0.92], gap="small", vertical_alignment="top")
    with primary:
        with st.container(border=True):
            st.markdown("<div class='panel-heading'><span>Cluster space</span><small>PCA PROJECTION</small></div>", unsafe_allow_html=True)
            chart, note = _pca_3d_chart(st.session_state.get("cluster_matrix"), getattr(result, "labels", None))
            if chart is None:
                st.markdown(f"<div class='empty-chart empty-chart-idle'><span>◇</span>{escape(note)}</div>", unsafe_allow_html=True)
                if result is None:
                    _dashboard_fit_button(frame, source_name, selected_algorithm, key="dashboard_fit_model")
            else:
                event = st.plotly_chart(
                    chart,
                    width="stretch",
                    key="dashboard_cluster_space",
                    on_select="rerun",
                    selection_mode=("points", "box", "lasso"),
                    config={"displaylogo": False, "responsive": True, "scrollZoom": True},
                )
                selected_points = getattr(getattr(event, "selection", None), "points", [])
                if selected_points:
                    st.caption(f"{len(selected_points):,} projected point(s) selected. Selection is visual only and does not alter the model.")
                st.caption(note)
    with side:
        with st.container(border=True):
            _render_cluster_flow(distribution, result)
        with st.container(border=True):
            st.markdown("<div class='panel-heading'><span>Cluster distribution</span><small>ACTIVE MODEL</small></div>", unsafe_allow_html=True)
            donut = _distribution_donut(distribution)
            if donut is None:
                st.markdown("<div class='empty-panel'>Fit a model to calculate record shares by cluster.</div>", unsafe_allow_html=True)
            else:
                st.plotly_chart(donut, width="stretch", config={"displaylogo": False, "responsive": True})
    insights_column, profile_column, performance_column = st.columns([0.92, 1.35, 1.03], gap="small", vertical_alignment="top")
    with insights_column:
        with st.container(border=True):
            _render_insights(result, distribution, manifest)
    with profile_column:
        with st.container(border=True):
            _render_profile_comparison(frame, getattr(result, "labels", None), key_suffix="overview")
    with performance_column:
        with st.container(border=True):
            _render_performance(result)

    with st.expander("Dataset boundary and source context", icon=":material/shield:"):
        active_name = Path(source_name).name
        if "district" in active_name.casefold():
            scope_text = "This source contains district-level aggregate registration records and does not describe individual entrepreneur behavior or outcomes."
        else:
            scope_text = "Metrics describe records in this active dataset and do not support causal or individual-level conclusions."
        st.write(
            f"Active source: `{active_name}` · {summary['rows']:,} records · {summary['columns']:,} observed columns. {scope_text}"
        )


def _render_assignment_workspace(frame: pd.DataFrame, result: object, manifest: dict | None) -> None:
    """Render the existing provenance-validated new-record assignment workflow unchanged."""
    if result.algorithm != "K-Means":
        st.info("New-record assignment is available for K-Means models with verified provenance. The active model does not support this workflow.")
        return
    _, names = cluster_profile(frame, result.labels)
    valid_assignment_provenance, assignment_provenance_reason = validate_assignment_provenance(manifest)
    if not valid_assignment_provenance:
        st.error(f"Assignment blocked: {assignment_provenance_reason}")
        st.caption("Retrain the model to create verified assignment provenance.")
    new_upload = st.file_uploader("Choose a file with the same source features", type=["dta", "csv", "xlsx", "xls"], key="new_record_upload")
    if valid_assignment_provenance and new_upload is not None:
        try:
            new_frame = load_dataset(new_upload).frame
            selected_features = st.session_state.cluster_selected
            missing_features = [column for column in selected_features if column not in new_frame.columns]
            if missing_features:
                st.error(f"New file is missing selected features: {', '.join(map(str, missing_features))}")
            else:
                original_filename = getattr(new_upload, "name", "unknown")
                assignment_data_fingerprint = dataset_fingerprint(new_frame)
                event_key = (
                    str(original_filename),
                    assignment_data_fingerprint,
                    manifest["model_run_id"],
                )
                assignment_event = st.session_state.get("assignment_event")
                cached_key = st.session_state.get("assignment_event_key")

                if assignment_event is None or cached_key != event_key:
                    new_matrix = st.session_state.cluster_preprocessor.transform(new_frame[selected_features])
                    assignment_labels = result.model.predict(new_matrix)
                    assigned = new_frame.copy()
                    assigned["cluster"] = assignment_labels
                    assigned["segment_name"] = assigned["cluster"].map(names)
                    assignment_timestamp = datetime.now(timezone.utc)
                    assignment_metadata = build_assignment_manifest(
                        manifest=manifest,
                        assigned_frame=assigned,
                        original_input_filename=str(original_filename),
                        cluster_labels=assignment_labels,
                        timestamp=assignment_timestamp,
                    )
                    csv_payload = build_assignment_csv_payload(assigned)
                    metadata_payload = build_assignment_metadata_payload(assignment_metadata)
                    assignment_event = {
                        "assigned": assigned,
                        "metadata": assignment_metadata,
                        "csv_payload": csv_payload,
                        "metadata_payload": metadata_payload,
                        "zip_payload": build_assignment_zip_payload(csv_payload, metadata_payload),
                    }
                    st.session_state.assignment_event = assignment_event
                    st.session_state.assignment_event_key = event_key
                else:
                    assigned = assignment_event["assigned"]

                st.dataframe(assigned.head(20), width="stretch", hide_index=True)
                metadata = assignment_event["metadata"]
                st.download_button(
                    "Download assigned records",
                    assignment_event["csv_payload"],
                    file_name="microsegment_new_assignments.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download assignment metadata (JSON)",
                    assignment_event["metadata_payload"],
                    file_name="microsegment_assignment_metadata.json",
                    mime="application/json",
                )
                st.download_button(
                    "Download assignment package (CSV + JSON)",
                    assignment_event["zip_payload"],
                    file_name=f"microsegment_assignment_{metadata['model_run_id']}.zip",
                    mime="application/zip",
                )
        except Exception as exc:
            st.error(f"Could not assign new records: {exc}")


def _registration_controls(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Return a filtered display view without touching analysis or export state."""
    required = {"state_name", "micro", "small", "medium", "total"}
    if not required.issubset(frame.columns):
        return frame, "total"

    states = sorted(frame["state_name"].dropna().astype(str).str.strip().unique())
    available_states = set(states)
    saved_states = [
        state
        for state in st.session_state.get("microsegment_visual_states", [])
        if state in available_states
    ]
    metric_labels = {
        "total": "Total registrations",
        "micro": "Micro registrations",
        "small": "Small registrations",
        "medium": "Medium registrations",
    }
    with st.container(border=True):
        st.markdown(
            "<div class='visual-control-heading'><span class='status-pulse'></span>"
            "<div><strong>Visual command centre</strong>"
            "<span>Explore a focused view — these controls never change the data, model, or downloads.</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        state_column, metric_column, reset_column = st.columns([1.5, 1, 0.55], vertical_alignment="bottom")
        with state_column:
            selected_states = st.multiselect(
                "Focus states / UTs",
                options=states,
                default=saved_states,
                placeholder="All states and UTs",
                key="microsegment_visual_states",
            )
        with metric_column:
            metric = st.selectbox(
                "Map intensity",
                options=list(metric_labels),
                format_func=lambda value: metric_labels[value],
                key="microsegment_visual_metric",
            )
        with reset_column:
            if st.button("Reset view", icon=":material/restart_alt:", key="microsegment_visual_reset"):
                st.session_state.pop("microsegment_visual_states", None)
                st.session_state.pop("microsegment_visual_metric", None)
                st.rerun()

    if not selected_states:
        return frame, metric
    return frame[frame["state_name"].astype(str).str.strip().isin(selected_states)].copy(), metric


def _registration_insights(state_summary: pd.DataFrame) -> None:
    """Present compact, source-derived highlights for the active visual view."""
    if state_summary.empty:
        return
    total = float(state_summary["total"].sum())
    leading = state_summary.iloc[0]
    top_five_share = (state_summary.head(5)["total"].sum() / total * 100) if total else 0.0
    micro_share = (state_summary["micro"].sum() / total * 100) if total else 0.0
    st.markdown("### Data story")
    first, second, third = st.columns(3)
    with first:
        _kpi("LEADING STATE / UT", str(leading["state_name"]).title(), f"{int(leading['total']):,} total registrations")
    with second:
        _kpi("TOP-5 CONCENTRATION", f"{top_five_share:.1f}%", "Share of the selected visual view")
    with third:
        _kpi("MICRO ENTERPRISE MIX", f"{micro_share:.1f}%", "Of all registrations in this view")


def _clear_dependent_model_state(reason: str) -> None:
    for key in (
        "cluster_result",
        "cluster_preprocessor",
        "cluster_matrix",
        "cluster_selected",
        "cluster_source",
        "cluster_manifest",
        "assignment_event",
        "assignment_event_key",
    ):
        st.session_state.pop(key, None)
    st.session_state["cluster_invalid_reason"] = reason


def _current_model(info: object | None = None, frame: pd.DataFrame | None = None):
    result = st.session_state.get("cluster_result")
    manifest = st.session_state.get("cluster_manifest")
    if result is None:
        return None
    if info is None or frame is None:
        return result
    selected = list(
        st.session_state.get(
            "cluster_features",
            st.session_state.get("cluster_selected", manifest.get("selected_features", []) if manifest else []),
        )
    )
    valid, reason = validate_model_manifest(
        manifest,
        frame=frame,
        labels=result.labels,
        selected_features=selected,
        preprocessor=st.session_state.get("cluster_preprocessor"),
        model=result.model,
    )
    if not valid:
        _clear_dependent_model_state(reason)
        LOGGER.warning("Invalidated stale clustering state: %s", reason)
        return None
    return result


def _restore_saved_model(info: object, frame: pd.DataFrame) -> bool:
    """Restore a legacy saved model only when its saved rows and labels prove a match.

    Older project artifacts do not include a model run ID or modern manifest.
    We can safely reuse their fitted state for this session only when the active
    dataset equals the saved source rows exactly and the estimator's own labels
    equal the saved output labels. Missing provenance remains missing.
    """
    if st.session_state.get("cluster_result") is not None:
        return False

    model_path = PROJECT_ROOT / "models" / "clustering_pipeline.joblib"
    config_path = PROJECT_ROOT / "models" / "preprocessing_config.json"
    rows_path = PROJECT_ROOT / "outputs" / "segmented_data" / "segmented_data.csv"
    if not all(path.is_file() for path in (model_path, config_path, rows_path)):
        return False

    try:
        bundle = _read_saved_model(str(model_path), model_path.stat().st_mtime)
        config = _read_saved_model_config(str(config_path), config_path.stat().st_mtime)
        saved_rows = _read_saved_segmented_rows(str(rows_path), rows_path.stat().st_mtime)
        if not isinstance(bundle, dict):
            return False
        model = bundle.get("model")
        preprocessor = bundle.get("preprocessor")
        required_output_columns = {"cluster", *map(str, frame.columns)}
        if model is None or preprocessor is None or not required_output_columns.issubset(saved_rows.columns):
            return False

        # The saved export must contain this exact source schema and row order.
        saved_source = saved_rows.loc[:, list(frame.columns)].reset_index(drop=True)
        if not frame.reset_index(drop=True).equals(saved_source):
            return False
        labels = saved_rows["cluster"].to_numpy()
        estimator_labels = getattr(model, "labels_", None)
        if estimator_labels is None or not np.array_equal(np.asarray(estimator_labels), labels):
            return False

        algorithm = str(config.get("algorithm", ""))
        model_class = type(model).__name__.casefold()
        expected_classes = {"K-Means": "kmeans", "DBSCAN": "dbscan", "K-Medoids": "kmedoids"}
        if algorithm not in expected_classes or expected_classes[algorithm] not in model_class:
            return False
        features = config.get("features")
        if not isinstance(features, list) or not features or any(column not in frame.columns for column in features):
            return False

        matrix = preprocessor.transform(frame[features])
        if matrix.shape[0] != len(frame):
            return False
        metrics = evaluate_clustering(matrix, labels)
        inertia_value = getattr(model, "inertia_", None)
        inertia = float(inertia_value) if inertia_value is not None else None
        if algorithm == "K-Means":
            parameters = {
                "n_clusters": int(getattr(model, "n_clusters")),
                "n_init": getattr(model, "n_init", None),
            }
        elif algorithm == "DBSCAN":
            parameters = {
                "eps": float(getattr(model, "eps")),
                "min_samples": int(getattr(model, "min_samples")),
            }
        else:
            parameters = {"n_clusters": int(getattr(model, "n_clusters"))}

        actual_seed = getattr(model, "random_state", None)
        if not isinstance(actual_seed, (int, np.integer)) or isinstance(actual_seed, bool):
            actual_seed = None
        manifest = {
            "dataset_fingerprint": dataset_fingerprint(frame),
            "schema_fingerprint": schema_fingerprint(frame),
            "dataset_row_count": int(len(frame)),
            "dataset_column_info": {str(column): str(frame[column].dtype) for column in frame.columns},
            "dataset_name": Path(info.source).name,
            "selected_features": list(features),
            "algorithm": algorithm,
            "algorithm_parameters": parameters,
            "random_seed": int(actual_seed) if actual_seed is not None else None,
            "seed_applicable": actual_seed is not None,
            "preprocessing": preprocessing_metadata(preprocessor, list(features)),
            "metrics": metrics,
            "legacy_artifact": True,
            "artifact_saved_at": datetime.fromtimestamp(model_path.stat().st_mtime, timezone.utc).isoformat(),
        }
        st.session_state.cluster_result = ClusteringResult(algorithm, labels, model, metrics, inertia)
        st.session_state.cluster_preprocessor = preprocessor
        st.session_state.cluster_matrix = matrix
        st.session_state.cluster_selected = list(features)
        st.session_state.cluster_features = list(features)
        st.session_state.cluster_source = info.source
        st.session_state.cluster_manifest = manifest
        st.session_state.pop("cluster_invalid_reason", None)
        return True
    except Exception as exc:
        LOGGER.warning("Could not restore saved model for the active dataset: %s", exc)
        return False


def _features(frame: pd.DataFrame) -> list[str]:
    candidates = default_clustering_features(frame)
    if "cluster_features" not in st.session_state:
        st.session_state.cluster_features = candidates
    st.session_state.cluster_features = [column for column in st.session_state.cluster_features if column in candidates]
    selected_from_widget = st.session_state.get("cluster_features_selection")
    if selected_from_widget is not None:
        st.session_state.cluster_features = [
            column for column in selected_from_widget if column in candidates
        ]
    manifest = st.session_state.get("cluster_manifest")
    if manifest and list(st.session_state.cluster_features) != list(manifest.get("selected_features", [])):
        _clear_dependent_model_state("The selected clustering features differ from the trained model.")
    return st.session_state.cluster_features


def _fit_model_into_session(
    frame: pd.DataFrame,
    source_name: str,
    selected: list[str],
    algorithm: str,
    *,
    n_clusters: int = 4,
    eps: float = 0.5,
    min_samples: int = 5,
):
    """Run the existing fit pipeline and attach its verified state to this session."""
    matrix, preprocessor, chosen = preprocess_data(frame, selected)
    if algorithm == "K-Means":
        fitted = fit_kmeans(matrix, int(n_clusters))
        parameters = {"n_clusters": int(n_clusters), "n_init": 10}
    elif algorithm == "DBSCAN":
        fitted = fit_dbscan(matrix, float(eps), int(min_samples))
        parameters = {"eps": float(eps), "min_samples": int(min_samples)}
    else:
        fitted = fit_kmedoids(matrix, int(n_clusters))
        parameters = {"n_clusters": int(n_clusters)}

    random_seed = RANDOM_SEED if algorithm in {"K-Means", "K-Medoids", "K-Medoids (optional)"} else None
    model_run_id = create_model_run_id()
    manifest = build_model_manifest(
        frame=frame,
        source_name=source_name,
        selected_features=chosen,
        algorithm=fitted.algorithm,
        algorithm_parameters=parameters,
        random_seed=random_seed,
        preprocessor=preprocessor,
        metrics=fitted.metrics,
        model_run_id=model_run_id,
        seed_applicable=random_seed is not None,
    )
    st.session_state.cluster_result = fitted
    st.session_state.cluster_preprocessor = preprocessor
    st.session_state.cluster_matrix = matrix
    st.session_state.cluster_selected = list(chosen)
    st.session_state.cluster_features = list(chosen)
    st.session_state.cluster_source = source_name
    st.session_state.cluster_manifest = manifest
    st.session_state.pop("cluster_invalid_reason", None)
    st.session_state.pop("assignment_event", None)
    st.session_state.pop("assignment_event_key", None)
    return fitted


def render_page(page: str) -> None:
    ensure_project_dirs()
    css_path = PROJECT_ROOT / "assets" / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
    info = _get_data()

    if info is None:
        _sidebar()
        _page_header("MICROSEGMENT / WORKSPACE", "No dataset loaded", "Upload a Stata, CSV, or Excel file to begin analysis.")
        st.info("No supported dataset was found in `data/raw`. The requested `IndiaMicro-2022-full-data.dta` is not currently present in the workspace.")
        return

    frame = info.frame
    summary = dataset_summary(info)
    _restore_saved_model(info, frame)
    result = _current_model(info, frame)
    manifest = st.session_state.get("cluster_manifest") if result is not None else None
    selected_features = set((manifest or {}).get("selected_features", st.session_state.get("cluster_features", [])))
    for warning in getattr(info, "warnings", []):
        if page == "Dashboard" and result is not None and warning.startswith("High-cardinality categorical feature '"):
            warning_column = warning.split("'", 2)[1]
            if warning_column not in selected_features:
                continue
        st.warning(warning)
    sidebar_model_detail = (manifest or {}).get("model_run_id") or ("Verified saved artifact · no run ID" if (manifest or {}).get("legacy_artifact") else None)
    _sidebar(bool(result), sidebar_model_detail)
    _top_header(info, result, manifest)
    labels = result.labels if result is not None and len(result.labels) == len(frame) else None

    if page == "Dashboard":
        _render_overview(frame, summary, result, manifest, info.source)

    elif page == "Dataset Explorer":
        _page_header("DATA / INSPECT", "Dataset Explorer", "Inspect the source schema, quality signals, labels, and observed values.")
        with st.container(border=True):
            st.markdown("<div class='panel-heading'><span>Active dataset</span><small>UPLOAD / SOURCE</small></div>", unsafe_allow_html=True)
            source_column, upload_column = st.columns([1, 1], gap="small", vertical_alignment="bottom")
            paths = discover_datasets(PROJECT_ROOT)
            with source_column:
                if paths:
                    st.selectbox(
                        "Choose a project dataset",
                        [str(path) for path in paths],
                        format_func=lambda value: Path(value).name,
                        key="microsegment_source",
                    )
                else:
                    st.caption("No project dataset is available in data/raw.")
            with upload_column:
                uploaded = st.file_uploader(
                    "Upload a dataset",
                    type=["dta", "csv", "xlsx", "xls"],
                    key="microsegment_upload",
                    help="CSV, XLS, XLSX, and Stata DTA files are supported.",
                )
                if uploaded is not None:
                    st.caption(f"Using uploaded file: {uploaded.name}")
        st.caption(f"Source: {info.source}")
        a, b, c, d = st.columns(4)
        with a: _kpi("ROWS", f"{frame.shape[0]:,}")
        with b: _kpi("COLUMNS", str(frame.shape[1]))
        with c: _kpi("MISSING", f"{summary['missing_cells']:,}")
        with d: _kpi("DUPLICATES", f"{summary['duplicate_rows']:,}")
        st.markdown("### Records")
        st.dataframe(frame.head(100), width="stretch", hide_index=True)
        st.markdown("### Column audit")
        st.dataframe(feature_audit(frame), width="stretch", hide_index=True)
        if info.variable_labels:
            st.markdown("### Stata variable labels")
            st.dataframe(pd.DataFrame(info.variable_labels.items(), columns=["Variable", "Label"]), width="stretch", hide_index=True)
        if info.value_labels:
            st.markdown("### Stata value labels")
            st.json(info.value_labels)
        missing = summary["missing_by_column"]
        if missing.sum() > 0:
            st.plotly_chart(missing_chart(frame), width="stretch")
        st.markdown("### Descriptive statistics")
        stats = numeric_summary(frame)
        if not stats.empty:
            st.dataframe(stats, width="stretch")
        else:
            st.info("No numeric columns were detected.")

    elif page == "Preprocessing":
        _page_header("DATA / PREPARE", "Preprocessing", "Choose usable features and inspect the transformations before fitting a model.")
        roles = summary["roles"]
        st.caption(f"Identifier-like columns are detected and excluded from the default feature set. Detected identifiers: {', '.join(map(str, roles['identifier'])) or 'none'}")
        default_features = _features(frame)
        selected = st.multiselect(
            "Features for analysis",
            options=default_clustering_features(frame),
            default=default_features,
            key="cluster_features_selection",
        )
        st.session_state.cluster_features = list(selected)
        manifest = st.session_state.get("cluster_manifest")
        if manifest and list(selected) != list(manifest.get("selected_features", [])):
            _clear_dependent_model_state("The selected clustering features differ from the trained model.")
        st.caption("Numeric values: median imputation + StandardScaler. Categorical values: most-frequent imputation + one-hot encoding.")
        report = quality_report(frame)
        st.write(f"Duplicate records detected: **{report['duplicate_rows']:,}** · Constant columns: {', '.join(report['constant_columns']) or 'none'}")
        with st.spinner("Fitting preprocessing preview…"):
            if selected:
                try:
                    matrix, _, chosen = preprocess_data(frame, selected)
                    st.success(f"Prepared {matrix.shape[0]:,} records across {matrix.shape[1]:,} encoded features.")
                    st.dataframe(pd.DataFrame(matrix[:10].toarray() if hasattr(matrix, "toarray") else matrix[:10]), width="stretch")
                    st.json({"selected_features": chosen, "transformed_columns": int(matrix.shape[1]), "matrix_format": type(matrix).__name__})
                except Exception as exc:
                    st.error(f"Preprocessing preview failed: {exc}")
            else:
                st.warning("Select at least one feature to preview preprocessing.")
        if roles["numeric"]:
            column = st.selectbox("Inspect numeric outliers", roles["numeric"])
            st.dataframe(frame[column].describe(percentiles=[.01, .25, .5, .75, .99]).to_frame(), width="stretch")

    elif page == "EDA Analytics":
        _page_header("DATA / EXPLORE", "Exploratory Analysis", "Explore distributions and associations in the columns actually present.")
        if frame.empty:
            st.warning("No records to display.")
            return
        column = st.selectbox("Variable", list(frame.columns))
        st.plotly_chart(distribution_chart(frame, column), width="stretch")
        correlations = correlation_matrix(frame)
        if not correlations.empty:
            st.markdown("### Numeric correlations")
            st.plotly_chart(heatmap(correlations), width="stretch")
        else:
            st.info("A correlation heatmap needs at least two numeric variables.")
        st.markdown("### Numeric summaries")
        stats = numeric_summary(frame)
        if not stats.empty:
            st.dataframe(stats, width="stretch")

    elif page == "Clustering Studio":
        _page_header("MODEL / TRAIN", "Clustering Studio", "Fit a reproducible model and compare internal diagnostics; no single metric determines the best segmentation.")
        candidates = default_clustering_features(frame)
        selected = st.multiselect(
            "Clustering features",
            candidates,
            default=_features(frame),
            key="cluster_features_selection",
        )
        st.session_state.cluster_features = list(selected)
        manifest = st.session_state.get("cluster_manifest")
        if manifest and list(selected) != list(manifest.get("selected_features", [])):
            _clear_dependent_model_state("The selected clustering features differ from the trained model.")
        algo = st.selectbox("Algorithm", ["K-Means", "DBSCAN", "K-Medoids (optional)"])
        p1, p2 = st.columns(2)
        with p1:
            k = st.number_input("Number of clusters (K)", min_value=2, max_value=max(2, len(frame) - 1), value=min(4, max(2, len(frame) - 1)), disabled=algo == "DBSCAN")
        with p2:
            eps = st.number_input("DBSCAN eps", min_value=0.01, value=0.5, step=0.05, disabled=algo != "DBSCAN")
        min_samples = st.number_input("DBSCAN min_samples", min_value=2, value=5, step=1, disabled=algo != "DBSCAN")
        if st.button("Fit clustering model", type="primary", disabled=not selected or len(frame) < 3):
            try:
                with st.spinner("Preparing features and fitting model…"):
                    fitted = _fit_model_into_session(
                        frame,
                        info.source,
                        list(selected),
                        algo,
                        n_clusters=int(k),
                        eps=float(eps),
                        min_samples=int(min_samples),
                    )
                st.success(f"{fitted.algorithm} fitted to {len(frame):,} rows.")
            except Exception as exc:
                st.error(f"Model fitting failed: {exc}")
        result = _current_model(info, frame)
        if result is not None and len(result.labels) == len(frame):
            st.markdown("### Evaluation")
            st.json(result.metrics | {"inertia": result.inertia})
            if result.algorithm == "K-Means":
                with st.spinner("Calculating elbow curve and restart stability…"):
                    elbow = elbow_scores(st.session_state.cluster_matrix, maximum_k=min(10, max(2, len(frame) - 1)))
                    stability = kmeans_stability(st.session_state.cluster_matrix, int(k), repeats=4)
                st.caption(f"Mean adjusted Rand agreement across four seeded restarts: {stability:.3f}" if stability is not None else "Stability not available")
                if elbow:
                    st.plotly_chart(px.line(pd.DataFrame(elbow), x="k", y="inertia", markers=True, template="plotly_dark", labels={"k": "K", "inertia": "Within-cluster sum of squares"}), width="stretch")
            st.dataframe(pd.DataFrame({"cluster": result.labels}).value_counts().rename("records").reset_index(), width="stretch", hide_index=True)

    elif page == "Model Evaluation":
        _page_header("MODEL / EVALUATE", "Model evaluation", "Review current-run internal diagnostics. Historic performance is only shown when the application has actually calculated a parameter sweep.")
        if result is None or labels is None:
            st.info("Fit a model in Clustering Studio to view model diagnostics.")
            return
        metric_columns = st.columns(4, gap="small")
        with metric_columns[0]:
            _kpi("Clusters", str(int(result.metrics.get("n_clusters") or 0)), "Non-noise clusters", tone="cyan")
        with metric_columns[1]:
            _kpi("Silhouette", _format_metric(result.metrics.get("silhouette")), "Higher can indicate separation", tone="violet")
        with metric_columns[2]:
            _kpi("Davies–Bouldin", _format_metric(result.metrics.get("davies_bouldin")), "Lower can indicate separation", tone="blue")
        with metric_columns[3]:
            _kpi("Noise points", f"{int(result.metrics.get('noise_points') or 0):,}", "DBSCAN-only when present", tone="lime")
        main_column, detail_column = st.columns([1.25, 0.75], gap="small", vertical_alignment="top")
        with main_column:
            with st.container(border=True):
                if result.algorithm == "K-Means":
                    with st.spinner("Calculating the current K sweep…"):
                        elbow = elbow_scores(st.session_state.cluster_matrix, maximum_k=min(10, max(2, len(frame) - 1)))
                    if elbow:
                        curve = px.line(
                            pd.DataFrame(elbow),
                            x="k",
                            y="inertia",
                            markers=True,
                            labels={"k": "Cluster count (K)", "inertia": "Within-cluster sum of squares"},
                        )
                        curve.update_traces(line={"color": "#67EBFF", "width": 3}, marker={"size": 8, "color": "#A78BFA"})
                        curve.update_xaxes(dtick=1, gridcolor="rgba(126,176,255,0.12)")
                        curve.update_yaxes(gridcolor="rgba(126,176,255,0.12)")
                        st.markdown("<div class='panel-heading'><span>Inertia by K</span><small>CALCULATED NOW</small></div>", unsafe_allow_html=True)
                        st.plotly_chart(_neon_plot_layout(curve, height=410), width="stretch", config={"displaylogo": False, "responsive": True})
                else:
                    st.markdown("<div class='empty-chart'><span>◌</span>No K sweep applies to the active DBSCAN or K-Medoids model.</div>", unsafe_allow_html=True)
        with detail_column:
            with st.container(border=True):
                _render_performance(result)
            with st.container(border=True):
                distribution = _cluster_distribution(labels)
                _render_insights(result, distribution, manifest)

    elif page == "New Assignments":
        _page_header("MODEL / ASSIGN", "New assignments", "Assign compatible new records only through the active model's verified provenance chain.")
        if result is None or labels is None:
            st.info("Fit a model before assigning new records.")
            return
        with st.container(border=True):
            _render_assignment_workspace(frame, result, manifest)

    elif page == "Provenance":
        _page_header("MODEL / TRACE", "Provenance", "Inspect the active model run's immutable dataset, schema, feature, and parameter metadata.")
        if result is None or manifest is None:
            st.info("Fit a model to create an in-session provenance manifest.")
            return
        valid, reason = validate_model_manifest(
            manifest,
            frame=frame,
            labels=labels,
            selected_features=st.session_state.get("cluster_selected", []),
            preprocessor=st.session_state.get("cluster_preprocessor"),
            model=result.model,
        )
        state_column, manifest_column = st.columns([0.72, 1.28], gap="small", vertical_alignment="top")
        with state_column:
            with st.container(border=True):
                st.markdown("<div class='panel-heading'><span>Validation status</span><small>LIVE CHECK</small></div>", unsafe_allow_html=True)
                if valid:
                    st.success("The active model matches its dataset and preprocessing provenance.", icon=":material/verified:")
                else:
                    st.error(reason, icon=":material/error:")
                st.caption(f"Run ID: {manifest.get('model_run_id', 'unavailable')}")
                st.caption(f"Created: {_model_timestamp(manifest)}")
        with manifest_column:
            with st.container(border=True):
                st.markdown("<div class='panel-heading'><span>Model manifest</span><small>READ ONLY</small></div>", unsafe_allow_html=True)
                st.json(manifest)
        assignment_event = st.session_state.get("assignment_event")
        if assignment_event:
            with st.container(border=True):
                st.markdown("<div class='panel-heading'><span>Latest assignment event</span><small>SESSION STATE</small></div>", unsafe_allow_html=True)
                st.json(assignment_event.get("metadata", {}))

    elif page == "Settings":
        _page_header("WORKSPACE / SETTINGS", "Settings", "This view exposes active workspace facts without changing protected model, dataset, or export behavior.")
        settings = pd.DataFrame(
            [
                ("Active source", Path(info.source).name),
                ("Dataset rows", f"{len(frame):,}"),
                ("Dataset columns", str(frame.shape[1])),
                ("Model run", (manifest or {}).get("model_run_id", "No active model")),
                ("Algorithm", getattr(result, "algorithm", "No active model")),
                ("Theme", "Dark neon workspace"),
            ],
            columns=["Setting", "Current value"],
        )
        with st.container(border=True):
            st.dataframe(settings, width="stretch", hide_index=True)
        st.caption("The interface respects reduced-motion browser preferences. Dataset selection, preprocessing, model fitting, and exports remain on their dedicated workflow pages.")

    elif page == "Cluster Profiles":
        _page_header("MODEL / INTERPRET", "Segment Profiles", "Compare observed feature summaries. Names are descriptive shorthand, not validated business outcomes.")
        if labels is None:
            st.info("Fit a model in Clustering Studio first.")
            return
        profile, names = cluster_profile(frame, labels)
        mean_columns = [
            column for column in profile.columns
            if column.endswith("__mean") and pd.api.types.is_numeric_dtype(profile[column])
        ]
        if not profile.empty:
            st.markdown("### Segment snapshots")
            segment_columns = st.columns(min(4, len(profile)))
            for index, (cluster, row) in enumerate(profile.iterrows()):
                with segment_columns[index % len(segment_columns)]:
                    segment_name = str(names.get(cluster, f"Segment {cluster}"))
                    _kpi(
                        segment_name.upper(),
                        f"{int(row.get('size', 0)):,}",
                        "Records in this descriptive segment",
                    )
                    if mean_columns:
                        strongest = max(mean_columns, key=lambda column: abs(float(row.get(column, 0) or 0)))
                        readable_name = strongest.removesuffix("__mean").replace("_", " ").title()
                        st.caption(f"Observed mean · {readable_name}: {float(row.get(strongest, 0) or 0):,.2f}")
            if mean_columns:
                st.markdown("### Feature-shape comparison")
                selected_profile_features = [column.removesuffix("__mean") for column in mean_columns[:6]]
                st.caption("Radar view compares observed means from the active trained model; use the profile table for exact values.")
                st.plotly_chart(
                    radar_chart(profile, selected_profile_features),
                    width="stretch",
                    config={"displaylogo": False, "responsive": True},
                )
        st.markdown("### Full profile table")
        st.dataframe(profile, width="stretch")
        roles = infer_column_roles(frame)
        categorical = roles["categorical"] + roles["binary"]
        if categorical:
            chosen = st.selectbox("Categorical profile variable", categorical)
            cross = pd.crosstab(pd.Series(labels, name="Cluster"), frame[chosen].astype("string"), normalize="index")
            st.plotly_chart(px.imshow(cross, aspect="auto", color_continuous_scale="Blues", template="plotly_dark"), width="stretch")

    elif page == "3D Visualization":
        _page_header(
            "DATA + MODEL / VISUALIZE",
            "Visual Analytics",
            "Explore the active registration source in 2D, 3D, and on the map; add segmentation overlays after fitting a model.",
        )
        visual_frame, selected_metric = _registration_controls(frame)
        visual_state_summary = registration_state_summary(visual_frame)
        _registration_insights(visual_state_summary)
        map_tab, exploration_tab, model_tab = st.tabs(
            ["Registration map", "2D + 3D explorer", "Segmentation"]
        )
        with map_tab:
            if visual_state_summary.empty:
                st.info("A map needs state_name plus micro, small, medium, and total registration columns.")
            else:
                map_column, ranking_column = st.columns([1.15, 0.85], vertical_alignment="top")
                with map_column:
                    st.markdown("#### India state and UT heatmap")
                    st.caption(f"Hover, pan, and zoom. Color encodes {selected_metric} registrations in the active visual view.")
                    st.plotly_chart(
                        state_registration_choropleth(visual_frame, selected_metric),
                        width="stretch",
                        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
                    )
                with ranking_column:
                    st.markdown("#### Leading states and UTs")
                    st.plotly_chart(
                        top_states_chart(visual_frame),
                        width="stretch",
                        config={"displaylogo": False, "responsive": True},
                    )
                    st.caption("Boundary geometry: India Maps Data collection. District boundaries are aggregated to the state/UT display view; registration values remain from the active source.")
                with st.expander("Explore the registration point layer", icon=":material/location_searching:"):
                    st.caption("Each glow point represents a state or UT. Point size and color both use total registrations.")
                    st.plotly_chart(
                        state_registration_map(visual_frame),
                        width="stretch",
                        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
                    )
        with exploration_tab:
            st.markdown("#### Interactive 3D district view")
            st.caption("Rotate and zoom to inspect the relationship between micro, small, and total registrations. Log axes keep both small and large districts visible.")
            st.plotly_chart(
                registration_3d_chart(visual_frame),
                width="stretch",
                config={"displaylogo": False, "responsive": True},
            )
            bubble_column, mix_column = st.columns([1.15, 0.85], vertical_alignment="top")
            with bubble_column:
                st.markdown("#### 2D state comparison")
                st.caption("Each bubble is a state or UT; bubble size represents small-enterprise registrations.")
                st.plotly_chart(
                    state_registration_bubble_chart(visual_frame),
                    width="stretch",
                    config={"displaylogo": False, "responsive": True},
                )
            with mix_column:
                st.markdown("#### Registration mix")
                st.plotly_chart(
                    registration_mix_chart(visual_frame),
                    width="stretch",
                    config={"displaylogo": False, "responsive": True},
                )
        with model_tab:
            if labels is None:
                st.info("Fit a model in Clustering Studio to add the PCA segmentation overlay and profile tables.")
            else:
                projection, variance = pca_projection(st.session_state.cluster_matrix)
                st.plotly_chart(cluster_scatter(projection, labels), width="stretch")
                st.caption(f"PCA explained variance: {', '.join(f'{value:.1%}' for value in variance)}. Two dimensions are a visual projection and may not preserve all structure.")
                sizes = pd.Series(labels, name="cluster").value_counts().sort_index().rename("records").reset_index()
                st.plotly_chart(px.bar(sizes, x="cluster", y="records", color="cluster", template="plotly_dark"), width="stretch")
                profile, _ = cluster_profile(frame, labels)
                if "size" in profile:
                    st.dataframe(profile, width="stretch")

    elif page == "Export & Reports":
        _page_header("OUTPUT / EXPORT", "Reports & Downloads", "Download analysis tables and interactive charts for reproducible review.")
        if labels is None:
            invalid_reason = st.session_state.get("cluster_invalid_reason")
            if invalid_reason:
                st.error(f"The trained model is invalid: {invalid_reason} Retrain the model for the active dataset.")
            else:
                st.info("Fit a clustering model before exporting segment-specific outputs.")
            st.download_button("Download active source as CSV", frame.to_csv(index=False).encode("utf-8"), file_name="microsegment_source.csv", mime="text/csv")
            return
        profile, names = cluster_profile(frame, labels)
        segmented = frame.copy()
        segmented["cluster"] = labels
        segmented["segment_name"] = segmented["cluster"].map(names)
        manifest = st.session_state.get("cluster_manifest")
        valid, validation_reason = validate_model_manifest(
            manifest,
            frame=frame,
            labels=labels,
            selected_features=st.session_state.get("cluster_selected", []),
            preprocessor=st.session_state.get("cluster_preprocessor"),
            model=result.model,
        )
        if not valid:
            st.error(f"Exports blocked: {validation_reason} Retrain the model.")
            st.download_button("Download active source as CSV", frame.to_csv(index=False).encode("utf-8"), file_name="microsegment_source.csv", mime="text/csv")
            return
        st.download_button("Download segmented data", segmented.to_csv(index=False).encode("utf-8"), file_name="microsegment_segmented_data.csv", mime="text/csv")
        st.download_button("Download cluster profiles", profile.to_csv().encode("utf-8"), file_name="microsegment_cluster_profiles.csv", mime="text/csv")
        projection, _ = pca_projection(st.session_state.cluster_matrix)
        chart = cluster_scatter(projection, labels)
        st.download_button("Download PCA chart (HTML)", chart.to_html(include_plotlyjs="cdn").encode("utf-8"), file_name="microsegment_pca.html", mime="text/html")
        st.download_button("Download profiles as JSON", pd.Series(names).to_json(indent=2).encode("utf-8"), file_name="microsegment_segment_names.json", mime="application/json")
        if st.button("Save outputs and trained model", type="primary"):
            try:
                paths = save_outputs(
                    segmented,
                    profile,
                    st.session_state.cluster_preprocessor,
                    result.model,
                    {
                        "source": info.source,
                        "algorithm": result.algorithm,
                        "features": st.session_state.cluster_selected,
                        "metrics": result.metrics,
                        "dataset_fingerprint": manifest["dataset_fingerprint"],
                        "schema_fingerprint": manifest["schema_fingerprint"],
                    },
                    project_root=PROJECT_ROOT,
                    manifest=manifest,
                )
                st.success("Saved segmented data, profiles, trained model, and preprocessing configuration.")
                st.json({key: str(path.relative_to(PROJECT_ROOT)) for key, path in paths.items()})
            except Exception as exc:
                st.error(f"Could not save model outputs: {exc}")
        if result.algorithm == "K-Means":
            st.markdown("### Assign new records")
            _render_assignment_workspace(frame, result, manifest)
        st.caption("PDF report generation is not enabled; export the CSV tables and interactive PCA chart above.")


def run_app(page: str) -> None:
    st.set_page_config(page_title=f"{page} | MicroSegment", page_icon="MS", layout="wide", initial_sidebar_state="expanded")
    render_page(page)
