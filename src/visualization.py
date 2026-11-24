"""Plotly charts used by both the app and exported reports."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


INDIA_STATE_CENTROIDS = {
    "ANDAMAN AND NICOBAR ISLANDS": (11.7401, 92.6586),
    "ANDHRA PRADESH": (15.9129, 79.7400),
    "ARUNACHAL PRADESH": (28.2180, 94.7278),
    "ASSAM": (26.2006, 92.9376),
    "BIHAR": (25.0961, 85.3131),
    "CHANDIGARH": (30.7333, 76.7794),
    "CHHATTISGARH": (21.2787, 81.8661),
    "DELHI": (28.7041, 77.1025),
    "GOA": (15.2993, 74.1240),
    "GUJARAT": (22.2587, 71.1924),
    "HARYANA": (29.0588, 76.0856),
    "HIMACHAL PRADESH": (31.1048, 77.1734),
    "JAMMU AND KASHMIR": (33.7782, 76.5762),
    "JHARKHAND": (23.6102, 85.2799),
    "KARNATAKA": (15.3173, 75.7139),
    "KERALA": (10.8505, 76.2711),
    "LADAKH": (34.1526, 77.5771),
    "LAKSHADWEEP": (10.5667, 72.6417),
    "MADHYA PRADESH": (22.9734, 78.6569),
    "MAHARASHTRA": (19.7515, 75.7139),
    "MANIPUR": (24.6637, 93.9063),
    "MEGHALAYA": (25.4670, 91.3662),
    "MIZORAM": (23.1645, 92.9376),
    "NAGALAND": (26.1584, 94.5624),
    "ODISHA": (20.9517, 85.0985),
    "PUDUCHERRY": (11.9416, 79.8083),
    "PUNJAB": (31.1471, 75.3412),
    "RAJASTHAN": (27.0238, 74.2179),
    "SIKKIM": (27.5330, 88.5122),
    "TAMIL NADU": (11.1271, 78.6569),
    "TELANGANA": (18.1124, 79.0193),
    "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": (20.3974, 72.8328),
    "TRIPURA": (23.9408, 91.9882),
    "UTTAR PRADESH": (26.8467, 80.9462),
    "UTTARAKHAND": (30.0668, 79.0193),
    "WEST BENGAL": (22.9868, 87.8550),
}

REGISTRATION_COLUMNS = ("micro", "small", "medium", "total")

# The registration source uses the current official long-form name while the
# open boundary layer uses the shorter label.  This is a display-only join;
# it never changes the source data or any clustering input.
STATE_BOUNDARY_ALIASES = {
    "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU": "DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
}


def _apply_dashboard_layout(figure, *, height: int | None = None):
    """Apply the app's presentation styling without changing chart data."""
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,14,31,0.5)",
        font={"family": "Manrope, Segoe UI, sans-serif", "color": "#DCE8FF"},
        margin={"l": 20, "r": 20, "t": 34, "b": 24},
        legend={"bgcolor": "rgba(9,16,35,0.65)", "bordercolor": "rgba(134,174,255,0.18)"},
        height=height,
    )
    return figure


@lru_cache(maxsize=1)
def _india_boundary_geojson() -> dict | None:
    """Load the local boundary layer shipped with the dashboard, if present."""
    boundary_path = Path(__file__).resolve().parents[1] / "assets" / "india.geojson"
    if not boundary_path.exists():
        return None
    try:
        boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    # Add a normalized property without altering the attributed source file.
    for feature in boundary.get("features", []):
        properties = feature.setdefault("properties", {})
        state_name = str(properties.get("st_nm", "")).strip().upper()
        properties["state_key"] = state_name
    return boundary


def _boundary_state_key(state_name: object) -> str:
    normalized = str(state_name).strip().upper()
    return STATE_BOUNDARY_ALIASES.get(normalized, normalized)


def registration_state_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the active source into a geographic state/UT view for display."""
    required = {"state_name", *REGISTRATION_COLUMNS}
    if not required.issubset(frame.columns):
        return pd.DataFrame(
            columns=["state_name", *REGISTRATION_COLUMNS, "latitude", "longitude", "share_percent"]
        )

    values = frame[["state_name", *REGISTRATION_COLUMNS]].copy()
    values["state_name"] = values["state_name"].astype("string").str.strip().str.upper()
    for column in REGISTRATION_COLUMNS:
        values[column] = pd.to_numeric(values[column], errors="coerce").fillna(0)

    summary = values.groupby("state_name", as_index=False)[list(REGISTRATION_COLUMNS)].sum()
    centroids = pd.DataFrame.from_dict(
        INDIA_STATE_CENTROIDS, orient="index", columns=["latitude", "longitude"]
    ).rename_axis("state_name").reset_index()
    summary = summary.merge(centroids, on="state_name", how="inner")
    total = summary["total"].sum()
    summary["share_percent"] = summary["total"].div(total).mul(100) if total else 0.0
    return summary.sort_values("total", ascending=False, ignore_index=True)


def state_registration_map(frame: pd.DataFrame):
    """Map actual registration totals to representative state/UT coordinates."""
    summary = registration_state_summary(frame)
    figure = px.scatter_geo(
        summary,
        lat="latitude",
        lon="longitude",
        size="total",
        color="total",
        hover_name="state_name",
        hover_data={
            "latitude": False,
            "longitude": False,
            "micro": ":,.0f",
            "small": ":,.0f",
            "medium": ":,.0f",
            "total": ":,.0f",
            "share_percent": ":.2f",
        },
        color_continuous_scale=["#4D3D9F", "#348FD9", "#46E2D3", "#E7D46A"],
        size_max=34,
        labels={
            "total": "Total registrations",
            "micro": "Micro registrations",
            "small": "Small registrations",
            "medium": "Medium registrations",
            "share_percent": "Share of source (%)",
        },
    )
    figure.update_geos(
        scope="asia",
        fitbounds="locations",
        showland=True,
        landcolor="#101B39",
        showocean=True,
        oceancolor="#081225",
        showcountries=True,
        countrycolor="rgba(131, 186, 255, 0.32)",
        coastlinecolor="rgba(131, 186, 255, 0.25)",
        bgcolor="rgba(0,0,0,0)",
    )
    figure.update_traces(marker={"line": {"color": "rgba(230,248,255,0.78)", "width": 0.8}})
    return _apply_dashboard_layout(figure, height=450)


def state_registration_choropleth(frame: pd.DataFrame, metric: str = "total"):
    """Color real registration totals over the locally bundled India boundaries."""
    metric = metric if metric in REGISTRATION_COLUMNS else "total"
    summary = registration_state_summary(frame).copy()
    boundary = _india_boundary_geojson()
    if summary.empty or boundary is None:
        return go.Figure()

    summary["state_key"] = summary["state_name"].map(_boundary_state_key)
    figure = px.choropleth(
        summary,
        geojson=boundary,
        locations="state_key",
        featureidkey="properties.state_key",
        color=metric,
        hover_name="state_name",
        hover_data={
            "state_key": False,
            "micro": ":,.0f",
            "small": ":,.0f",
            "medium": ":,.0f",
            "total": ":,.0f",
            "share_percent": ":.2f",
        },
        color_continuous_scale=["#33245D", "#4D6DDE", "#35C6D8", "#E7D46A"],
        labels={
            "micro": "Micro registrations",
            "small": "Small registrations",
            "medium": "Medium registrations",
            "total": "Total registrations",
            "share_percent": "Share of active view (%)",
        },
    )
    figure.update_traces(
        marker={"line": {"color": "rgba(217, 239, 255, 0.38)", "width": 0.35}},
        colorbar={"thickness": 12, "outlinewidth": 0, "bgcolor": "rgba(0,0,0,0)"},
    )
    figure.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        showlakes=False,
        showocean=True,
        oceancolor="#081225",
    )
    return _apply_dashboard_layout(figure, height=540)


def registration_mix_chart(frame: pd.DataFrame):
    """Show the observed micro, small and medium registration mix."""
    available = [column for column in REGISTRATION_COLUMNS[:-1] if column in frame.columns]
    values = pd.DataFrame(
        {
            "registration_type": [column.title() for column in available],
            "registrations": [pd.to_numeric(frame[column], errors="coerce").fillna(0).sum() for column in available],
        }
    )
    figure = px.pie(
        values,
        names="registration_type",
        values="registrations",
        hole=0.64,
        color="registration_type",
        color_discrete_sequence=["#67EBFF", "#A78BFA", "#F8C36A"],
    )
    figure.update_traces(textinfo="percent", textfont={"color": "#F3F7FF"})
    return _apply_dashboard_layout(figure, height=360)


def top_states_chart(frame: pd.DataFrame, limit: int = 10):
    """Rank the highest-registration states/UTs from the active data source."""
    summary = registration_state_summary(frame).head(limit).sort_values("total")
    figure = px.bar(
        summary,
        x="total",
        y="state_name",
        orientation="h",
        color="total",
        color_continuous_scale=["#4D3D9F", "#348FD9", "#46E2D3"],
        text="total",
        labels={"total": "Total registrations", "state_name": "State / UT"},
    )
    figure.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    figure.update_layout(coloraxis_showscale=False)
    return _apply_dashboard_layout(figure, height=420)


def state_registration_bubble_chart(frame: pd.DataFrame):
    """Compare actual state/UT micro and total registrations in two dimensions."""
    summary = registration_state_summary(frame)
    figure = px.scatter(
        summary,
        x="micro",
        y="total",
        size="small",
        color="share_percent",
        hover_name="state_name",
        hover_data={
            "micro": ":,.0f",
            "small": ":,.0f",
            "medium": ":,.0f",
            "total": ":,.0f",
            "share_percent": ":.2f",
        },
        color_continuous_scale=["#4D3D9F", "#348FD9", "#46E2D3", "#E7D46A"],
        size_max=48,
        labels={
            "micro": "Micro registrations",
            "total": "Total registrations",
            "small": "Small registrations",
            "share_percent": "Share of source (%)",
        },
    )
    figure.update_xaxes(type="log", gridcolor="rgba(157,189,255,0.12)")
    figure.update_yaxes(type="log", gridcolor="rgba(157,189,255,0.12)")
    return _apply_dashboard_layout(figure, height=440)


def registration_3d_chart(frame: pd.DataFrame):
    """Render the actual district registrations as an interactive 3D relationship."""
    required = {"micro", "small", "total"}
    if not required.issubset(frame.columns):
        return go.Figure()

    columns = [column for column in ("state_name", "district_name", "micro", "small", "medium", "total") if column in frame.columns]
    values = frame[columns].copy()
    for column in ("micro", "small", "medium", "total"):
        if column in values:
            values[column] = pd.to_numeric(values[column], errors="coerce").fillna(0).clip(lower=1)
    hover_name = "district_name" if "district_name" in values else None
    figure = px.scatter_3d(
        values,
        x="micro",
        y="small",
        z="total",
        color="total",
        hover_name=hover_name,
        hover_data={column: ":,.0f" for column in ("micro", "small", "medium", "total") if column in values},
        color_continuous_scale=["#4D3D9F", "#348FD9", "#46E2D3", "#E7D46A"],
        labels={
            "micro": "Micro registrations",
            "small": "Small registrations",
            "total": "Total registrations",
            "medium": "Medium registrations",
        },
    )
    figure.update_traces(marker={"size": 4.5, "opacity": 0.8, "line": {"width": 0.3, "color": "#DFFBFF"}})
    figure.update_scenes(
        xaxis={"type": "log", "gridcolor": "rgba(157,189,255,0.14)", "backgroundcolor": "#0B142B"},
        yaxis={"type": "log", "gridcolor": "rgba(157,189,255,0.14)", "backgroundcolor": "#0B142B"},
        zaxis={"type": "log", "gridcolor": "rgba(157,189,255,0.14)", "backgroundcolor": "#0B142B"},
        bgcolor="rgba(0,0,0,0)",
        camera={"eye": {"x": 1.5, "y": 1.4, "z": 0.9}},
    )
    return _apply_dashboard_layout(figure, height=620)


def missing_chart(frame: pd.DataFrame):
    missing = frame.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    return px.bar(x=missing.index, y=missing.values, labels={"x": "Column", "y": "Missing values"}, template="plotly_dark")


def distribution_chart(frame: pd.DataFrame, column: str):
    if pd.api.types.is_numeric_dtype(frame[column]):
        return px.histogram(frame, x=column, marginal="box", template="plotly_dark")
    counts = frame[column].astype("string").value_counts().head(30).rename_axis(column).reset_index(name="count")
    return px.bar(counts, x=column, y="count", template="plotly_dark")


def cluster_scatter(projection, labels):
    frame = pd.DataFrame(projection, columns=["Component 1", "Component 2"])
    frame["Cluster"] = pd.Series(labels).astype(str)
    return px.scatter(frame, x="Component 1", y="Component 2", color="Cluster", template="plotly_dark", opacity=0.8)


def heatmap(matrix: pd.DataFrame):
    return px.imshow(matrix, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", template="plotly_dark")


def radar_chart(profile: pd.DataFrame, columns: list[str]):
    figure = go.Figure()
    for cluster, row in profile.iterrows():
        values = [row.get(f"{column}__mean") for column in columns]
        figure.add_trace(go.Scatterpolar(r=values, theta=columns, fill="toself", name=f"Segment {cluster}"))
    figure.update_layout(
        polar={
            "bgcolor": "rgba(8,14,31,0.35)",
            "radialaxis": {"visible": True, "gridcolor": "rgba(157,189,255,0.18)"},
            "angularaxis": {"gridcolor": "rgba(157,189,255,0.18)"},
        }
    )
    return _apply_dashboard_layout(figure, height=460)
