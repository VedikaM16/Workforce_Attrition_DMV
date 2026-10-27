"""Executive Workforce Attrition BI application.

The app builds an in-memory star schema from the IBM HR attrition CSV and
provides five executive-facing analytical pages:

* Overview
* What Drives Attrition?
* Drill-down Explorer
* Data Model & Schema Audit
* Power BI Showcase
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE = Path(__file__).parent
CSV_FILE = BASE / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

PRIMARY_NAVY = "#1E3A8A"
COBALT = "#0078D4"
ACCENT = "#2B88D8"
PALETTE = [PRIMARY_NAVY, COBALT, ACCENT, "#60A5FA", "#93C5FD"]
NEUTRAL_GRAY = "#F3F4F6"
TEXT = "#111827"
GRID = "#E5E7EB"
GREEN = "#059669"
RED = "#DC2626"

AGE_ORDER = ["Under 25", "25-34", "35-44", "45-54", "55+"]
TENURE_ORDER = ["0-2 yrs", "3-5 yrs", "6-10 yrs", "11+ yrs"]
INCOME_ORDER = ["Low", "Medium", "High", "Very High"]
MARITAL_ORDER = ["Single", "Married", "Divorced"]

DIMENSION_SPECS = [
    ("Dim_Department", ["Department"], "department_key"),
    ("Dim_jobrole", ["JobRole", "JobLevel"], "jobrole_key"),
    ("Dim_Education", ["Education", "EducationField"], "education_key"),
    ("Dim_Demographic", ["AgeBand", "Gender",
     "MaritalStatus"], "demographic_key"),
    ("Dim_Tenure", ["TenureBand", "TenureSort"], "tenure_key"),
]

st.set_page_config(
    page_title="Workforce Attrition | HR Business Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    """Apply the application design system."""
    st.markdown(
        f"""
        <style>
        :root {{
            --navy: {PRIMARY_NAVY};
            --cobalt: {COBALT};
            --text: {TEXT};
            --grid: {GRID};
        }}
        .stApp {{
            background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 32%);
            color: var(--text);
        }}
        .stApp, .stApp p, .stApp label, .stApp [data-testid="stMarkdownContainer"] {{
            color: {TEXT};
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{
            color: {PRIMARY_NAVY};
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F255F 0%, {PRIMARY_NAVY} 100%);
        }}
        [data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
        [data-testid="stMetric"] {{
            background: #FFFFFF;
            border: 1px solid #DDE5F0;
            border-radius: 12px;
            padding: 14px 16px;
            box-shadow: 0 5px 16px rgba(30, 58, 138, .08);
        }}
        [data-testid="stMetricLabel"] {{ color: #64748B !important; }}
        [data-testid="stMetricValue"] {{ color: {PRIMARY_NAVY} !important; }}
        .hero {{
            background: linear-gradient(120deg, {PRIMARY_NAVY}, {COBALT});
            color: white; border-radius: 16px; padding: 24px 28px;
            margin-bottom: 18px; box-shadow: 0 10px 24px rgba(30, 58, 138, .18);
        }}
        .hero h1 {{ color: white; margin: 0; font-size: 2rem; }}
        .hero p {{ color: #DBEAFE; margin: 6px 0 0; }}
        .section-label {{
            color: {PRIMARY_NAVY}; font-size: .78rem; font-weight: 700;
            letter-spacing: .12em; text-transform: uppercase; margin: 14px 0 4px;
        }}
        .caption {{ color: #64748B; font-size: .9rem; }}
        div[data-testid="stExpander"] {{
            border: 1px solid #DDE5F0; border-radius: 10px;
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid #DDE5F0;
            border-radius: 8px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def make_dimension(df: pd.DataFrame, columns: List[str], key: str) -> pd.DataFrame:
    """Create a deterministic, 1-indexed dimension table."""
    dimension = df[columns].drop_duplicates().sort_values(
        columns).reset_index(drop=True)
    dimension.insert(0, key, range(1, len(dimension) + 1))
    return dimension


@st.cache_data(show_spinner="Building the workforce star schema…")
def build_model() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], pd.DataFrame]:
    """Load, standardize, and validate the star schema."""
    raw = pd.read_csv(CSV_FILE)
    work = raw.copy()
    work["AttritionFlag"] = work["Attrition"].eq("Yes").astype("int8")
    work["AgeBand"] = pd.cut(
        work["Age"], bins=[0, 25, 35, 45, 55, float("inf")],
        labels=AGE_ORDER, right=False,
    ).astype(str)
    work["TenureBand"] = pd.cut(
        work["YearsAtCompany"], bins=[-1, 3, 6, 11, float("inf")],
        labels=TENURE_ORDER, right=False,
    ).astype(str)
    work["TenureSort"] = work["TenureBand"].map(
        {band: i for i, band in enumerate(TENURE_ORDER, start=1)}
    ).astype("int8")
    work["IncomeBand"] = pd.cut(
        work["MonthlyIncome"], bins=[0, 3000, 5000, 9000, float("inf")],
        labels=INCOME_ORDER, right=False,
    ).astype(str)

    dimensions: Dict[str, pd.DataFrame] = {}
    fact = work.copy()
    descriptive_columns: List[str] = []
    for name, columns, key in DIMENSION_SPECS:
        dimension = make_dimension(work, columns, key)
        dimensions[name] = dimension
        fact = fact.merge(dimension, on=columns, how="left",
                          validate="many_to_one")
        descriptive_columns.extend(columns)

    # Fact grain is one row per employee. Keep numeric measures and flags, but
    # remove descriptive attributes now represented by dimension foreign keys.
    fact = fact.drop(columns=descriptive_columns + ["Attrition"])
    fact.insert(0, "employee_key", range(1, len(fact) + 1))

    joined = fact.copy()
    for name, columns, key in DIMENSION_SPECS:
        joined = joined.merge(
            dimensions[name], on=key, how="left", validate="many_to_one")

    foreign_keys = [spec[2] for spec in DIMENSION_SPECS]
    if fact[foreign_keys].isna().any().any():
        raise ValueError(
            "Star schema validation failed: null foreign keys detected.")
    if len(fact) != len(raw):
        raise ValueError("Star schema validation failed: fact grain changed.")
    if any((fact[key] < 1).any() for key in foreign_keys):
        raise ValueError(
            "Star schema validation failed: invalid surrogate key.")
    return raw, fact, dimensions, joined


def aggregate_by(data: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return common headcount, leaver, and attrition-rate measures."""
    if data.empty:
        return pd.DataFrame(columns=[column, "Employees", "Left", "AttritionRate"])
    return (
        data.groupby(column, dropna=False, as_index=False)
        .agg(Employees=("AttritionFlag", "size"), Left=("AttritionFlag", "sum"))
        .assign(AttritionRate=lambda frame: frame["Left"] / frame["Employees"])
    )


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_row(data: pd.DataFrame, include_income: bool = False) -> None:
    """Render the executive KPI row."""
    total = len(data)
    leavers = int(data["AttritionFlag"].sum()) if total else 0
    rate = leavers / total if total else 0
    columns = st.columns(4 if include_income else 3)
    columns[0].metric("Total Employees", f"{total:,}")
    columns[1].metric("Attrition Count", f"{leavers:,}")
    columns[2].metric("Attrition Rate", f"{rate:.2%}")
    if include_income:
        income = data["MonthlyIncome"].mean() if total else 0
        columns[3].metric("Avg Monthly Income", format_currency(income))


def style_figure(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Arial, sans-serif", "color": TEXT, "size": 12},
        title_font={"family": "Arial, sans-serif",
                    "color": PRIMARY_NAVY, "size": 16},
        margin={"l": 8, "r": 18, "t": 48, "b": 12},
        hoverlabel={"bgcolor": "white", "bordercolor": PRIMARY_NAVY,
                    "font": {"color": TEXT, "size": 13}},
        legend={"orientation": "h", "y": -0.15},
        uniformtext={"minsize": 10, "mode": "show"},
    )
    fig.update_xaxes(
        showline=False, gridcolor=GRID, zeroline=False,
        title_font={"color": TEXT, "size": 12},
        tickfont={"color": TEXT, "size": 11},
    )
    fig.update_yaxes(
        showline=False, gridcolor=GRID, zeroline=False,
        title_font={"color": TEXT, "size": 12},
        tickfont={"color": TEXT, "size": 11},
    )
    return fig


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def display_chart(fig: go.Figure) -> None:
    """Render a Plotly chart with the full interaction toolbar enabled."""
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def chart_hover(frame: pd.DataFrame, category: str) -> List[str]:
    """Build rich hover data for all grouped charts."""
    return [
        f"<b>{row[category]}</b><br>Employees: {int(row['Employees']):,}"
        f"<br>Leavers: {int(row['Left']):,}"
        f"<br>Attrition Rate: {row['AttritionRate']:.1%}<extra></extra>"
        for _, row in frame.iterrows()
    ]


def horizontal_rate_chart(
    frame: pd.DataFrame, category: str, title: str, highlight_max: bool = True
) -> go.Figure:
    ordered = frame.sort_values("AttritionRate", ascending=True).copy()
    colors = [COBALT] * len(ordered)
    if highlight_max and colors:
        colors[-1] = RED
    fig = go.Figure(
        go.Bar(
            x=ordered["AttritionRate"],
            y=ordered[category],
            orientation="h",
            marker_color=colors,
            text=ordered["AttritionRate"].map(lambda value: f"{value:.1%}"),
            textposition="outside",
            textfont={"color": TEXT, "size": 12},
            customdata=chart_hover(ordered, category),
            hovertemplate="%{customdata}",
        )
    )
    fig.update_xaxes(tickformat=".0%", title="Attrition rate")
    return style_figure(fig, title)


def vertical_rate_chart(
    frame: pd.DataFrame, category: str, title: str, order: Iterable[str] = ()
) -> go.Figure:
    ordered = frame.copy()
    if order:
        ordered[category] = pd.Categorical(
            ordered[category], categories=list(order), ordered=True)
        ordered = ordered.sort_values(category)
    fig = go.Figure(
        go.Bar(
            x=ordered[category],
            y=ordered["AttritionRate"],
            marker_color=COBALT,
            text=ordered["AttritionRate"].map(lambda value: f"{value:.1%}"),
            textposition="outside",
            textfont={"color": TEXT, "size": 12},
            customdata=chart_hover(ordered, category),
            hovertemplate="%{customdata}",
        )
    )
    fig.update_yaxes(tickformat=".0%", title="Attrition rate")
    return style_figure(fig, title)


def render_empty_state(message: str = "No employees match the current filters.") -> None:
    st.warning(message)


def render_overview(data: pd.DataFrame) -> None:
    page_header(
        "Workforce Attrition Overview",
        "Executive snapshot of workforce size, leavers, risk concentration, and pay.",
    )
    metric_row(data, include_income=True)
    st.markdown('<div class="section-label">Executive signals</div>',
                unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        by_department = aggregate_by(data, "Department").sort_values(
            "AttritionRate", ascending=False)
        display_chart(
            horizontal_rate_chart(
                by_department, "Department", "Attrition rate by department"
            ),
        )
        if not by_department.empty:
            highest = by_department.iloc[0]
            st.caption(
                f"Highest risk: **{highest['Department']}** at "
                f"**{highest['AttritionRate']:.1%}** ({int(highest['Left']):,} leavers)."
            )
    with right:
        by_gender = aggregate_by(data, "Gender")
        fig = go.Figure(
            go.Pie(
                labels=by_gender["Gender"],
                values=by_gender["Left"],
                hole=0.6,
                marker={"colors": [COBALT, ACCENT]},
                textinfo="label+value+percent",
                textfont={"color": TEXT, "size": 12},
                customdata=chart_hover(by_gender, "Gender"),
                hovertemplate="%{customdata}",
            )
        )
        fig.add_annotation(
            text=f"<b>{int(data['AttritionFlag'].sum()):,}</b><br><span style='font-size:11px'>leavers</span>",
            x=0.5, y=0.5, showarrow=False, font={"size": 20, "color": PRIMARY_NAVY},
        )
        display_chart(style_figure(fig, "Attrition count by gender"))


def render_drivers(data: pd.DataFrame) -> None:
    page_header(
        "What Drives Attrition?",
        "Compare risk by working pattern, income, role, and employee sentiment.",
    )
    top_left, top_right = st.columns(2)
    with top_left:
        overtime = aggregate_by(data, "OverTime").sort_values("AttritionRate")
        display_chart(horizontal_rate_chart(
            overtime, "OverTime", "Attrition rate by overtime"
        ))
    with top_right:
        income = aggregate_by(data, "IncomeBand")
        display_chart(vertical_rate_chart(
            income, "IncomeBand", "Attrition rate by income band", INCOME_ORDER
        ))

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        roles = aggregate_by(data, "JobRole").sort_values(
            "AttritionRate", ascending=False)
        display_chart(horizontal_rate_chart(
            roles, "JobRole", "Attrition rate by job role"
        ))
    with bottom_right:
        satisfaction = aggregate_by(data, "JobSatisfaction")
        satisfaction["JobSatisfaction"] = satisfaction["JobSatisfaction"].astype(
            int)
        satisfaction = satisfaction.sort_values("JobSatisfaction")
        total = pd.DataFrame(
            [{
                "Job Satisfaction": "Total",
                "Total Employees": len(data),
                "Leavers": int(data["AttritionFlag"].sum()),
                "Attrition Rate": data["AttritionFlag"].mean() if len(data) else 0,
            }]
        )
        table = satisfaction.rename(
            columns={
                "JobSatisfaction": "Job Satisfaction",
                "Employees": "Total Employees",
                "Left": "Leavers",
            }
        )[["Job Satisfaction", "Total Employees", "Leavers", "AttritionRate"]]
        table = table.rename(columns={"AttritionRate": "Attrition Rate"})
        table = pd.concat([table, total], ignore_index=True)
        st.markdown("#### Job satisfaction breakdown")
        st.dataframe(
            table.style.format({"Attrition Rate": "{:.1%}"}).background_gradient(
                subset=["Attrition Rate"], cmap="Blues"
            ).set_properties(**{"color": TEXT, "background-color": "#FFFFFF"}),
            hide_index=True,
            use_container_width=True,
        )


def filter_with_select_all(
    data: pd.DataFrame, label: str, column: str, options: List[str]
) -> List[str]:
    """Render a slicer with an explicit Select All control."""
    key = f"select_all_{column}"
    all_selected = st.checkbox(f"Select all {label}", value=True, key=key)
    return options if all_selected else st.multiselect(
        label, options, default=options, key=f"values_{column}"
    )


def render_drilldown(data: pd.DataFrame) -> None:
    page_header(
        "Drill-down Explorer",
        "Slice the workforce, inspect department-to-role risk, and compare demographics.",
    )
    st.sidebar.markdown("### Global slicers")
    selections = {}
    for label, column, values in [
        ("Department", "Department", sorted(
            data["Department"].dropna().unique())),
        ("Age band", "AgeBand", AGE_ORDER),
        ("Gender", "Gender", sorted(data["Gender"].dropna().unique())),
        ("Job role", "JobRole", sorted(data["JobRole"].dropna().unique())),
        ("Marital status", "MaritalStatus", MARITAL_ORDER),
    ]:
        selections[column] = filter_with_select_all(
            data, label, column, values)

    filtered = data.copy()
    for column, values in selections.items():
        filtered = filtered[filtered[column].isin(values)]
    if filtered.empty:
        render_empty_state(
            "No employees match these slicers. Re-select at least one value.")
        return

    metric_row(filtered)
    st.markdown('<div class="section-label">Department and role risk</div>',
                unsafe_allow_html=True)
    drill_mode = st.radio(
        "View", ["Department overview", "Role detail",
                 "Department > Job Role treemap"],
        horizontal=True,
    )
    if drill_mode == "Department overview":
        level = aggregate_by(filtered, "Department")
        display_chart(horizontal_rate_chart(
            level, "Department", "Filtered attrition by department"
        ))
    elif drill_mode == "Role detail":
        department = st.selectbox("Select department", sorted(
            filtered["Department"].unique()))
        level = aggregate_by(
            filtered[filtered["Department"] == department], "JobRole"
        )
        display_chart(horizontal_rate_chart(
            level, "JobRole", f"Role risk within {department}"
        ))
    else:
        tree = aggregate_by(filtered, "JobRole")
        tree["Department"] = filtered.groupby("JobRole")["Department"].first().reindex(
            tree["JobRole"]
        ).values
        fig = px.treemap(
            tree, path=["Department", "JobRole"], values="Employees",
            color="AttritionRate", color_continuous_scale="Blues",
            hover_data={"Employees": ":,",
                        "Left": ":,", "AttritionRate": ":.1%"},
        )
        fig.update_traces(
            texttemplate="<b>%{label}</b><br>%{customdata[2]:.1%}",
            hovertemplate="<b>%{label}</b><br>Employees: %{customdata[0]:,}"
            "<br>Leavers: %{customdata[1]:,}<br>Attrition Rate: %{customdata[2]:.1%}<extra></extra>",
        )
        display_chart(style_figure(
            fig, "Department > Job Role attrition treemap"))

    marital = aggregate_by(filtered, "MaritalStatus")
    display_chart(vertical_rate_chart(
        marital, "MaritalStatus", "Attrition rate by marital status", MARITAL_ORDER
    ))


def render_schema_audit(raw: pd.DataFrame, fact: pd.DataFrame,
                        dimensions: Dict[str, pd.DataFrame], joined: pd.DataFrame) -> None:
    page_header(
        "Data Model & Schema Audit",
        "A validated star schema with one employee per fact row and conformed dimensions.",
    )
    st.markdown(
        "**Grain:** one row per employee in `Fact_Employee`. "
        "Dimensions provide descriptive slicing while the fact retains surrogate keys, "
        "numeric measures, and additive metrics. Relationships are dimension (1) to fact (many)."
    )
    if (BASE / "model_view.png").exists():
        st.image(str(BASE / "model_view.png"),
                 caption="Power BI star-schema model view", use_container_width=True)

    foreign_keys = [spec[2] for spec in DIMENSION_SPECS]
    orphan_count = sum(
        int(fact[key].isna().sum())
        + int((~fact[key].isin(dimensions[name][key])).sum())
        for name, _, key in DIMENSION_SPECS
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows", f"{len(raw):,}")
    c2.metric("Fact rows", f"{len(fact):,}")
    c3.metric("Dimensions", f"{len(dimensions)}")
    c4.metric("RI orphan/null keys", f"{orphan_count:,}")
    if orphan_count == 0:
        st.success(
            "Referential integrity passed: zero orphan keys and zero null surrogate keys.")
    else:
        st.error(
            "Referential integrity failed. Review the foreign-key columns below.")

    with st.expander("Fact_Employee schema and sample", expanded=True):
        schema = pd.DataFrame({
            "Column": fact.columns,
            "Data type": [str(dtype) for dtype in fact.dtypes],
            "Non-null": [int(fact[column].notna().sum()) for column in fact.columns],
        })
        st.dataframe(schema, hide_index=True, use_container_width=True)
        st.dataframe(fact.head(10), hide_index=True, use_container_width=True)
    for name, dimension in dimensions.items():
        with st.expander(f"{name} · {len(dimension):,} rows"):
            schema = pd.DataFrame({
                "Column": dimension.columns,
                "Data type": [str(dtype) for dtype in dimension.dtypes],
                "Non-null": [int(dimension[column].notna().sum()) for column in dimension.columns],
            })
            st.dataframe(schema, hide_index=True, use_container_width=True)
            st.dataframe(dimension.head(10), hide_index=True,
                         use_container_width=True)


def render_showcase() -> None:
    page_header(
        "Power BI Showcase",
        "Reference report pages aligned to the Streamlit implementation.",
    )
    assets = [
        ("Overview", "page1.png", "Executive overview and workforce KPI page."),
        ("Drivers", "page2.png", "Attrition drivers and job satisfaction analysis."),
        ("Drill-down", "page3.png",
         "Interactive drill-down and demographic exploration."),
        ("Data model", "model_view.png", "Power BI model view for the star schema."),
    ]
    tabs = st.tabs([asset[0] for asset in assets])
    for tab, (_, filename, caption) in zip(tabs, assets):
        with tab:
            image = BASE / filename
            if image.exists():
                st.image(str(image), caption=caption, use_container_width=True)
            else:
                st.info(
                    f"{filename} is not available in the application folder.")


def main() -> None:
    inject_styles()
    if not CSV_FILE.exists():
        st.error("Dataset not found. Place the HR attrition CSV beside app.py.")
        st.stop()
    try:
        raw, fact, dimensions, joined = build_model()
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Unable to build the workforce star schema: {error}")
        st.stop()

    st.sidebar.markdown("## Workforce Attrition")
    st.sidebar.caption("HR Business Intelligence · Star schema analytics")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "What Drives Attrition?", "Drill-down Explorer",
         "Data Model & Schema Audit", "Power BI Showcase"],
    )
    if page == "Overview":
        render_overview(joined)
    elif page == "What Drives Attrition?":
        render_drivers(joined)
    elif page == "Drill-down Explorer":
        render_drilldown(joined)
    elif page == "Data Model & Schema Audit":
        render_schema_audit(raw, fact, dimensions, joined)
    else:
        render_showcase()


if __name__ == "__main__":
    main()
