"""
Pure Altair interactive dashboard for college football power rankings.

All controls (year, conference, team selection) live inside the chart itself.
Generates a self-contained HTML file — no Streamlit required.

Run with: python visualization/app.py
"""

import json
import os
import sys
import webbrowser
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import altair as alt

from team_colors import TEAM_COLORS, team_color
from fte_style import CONFERENCE_COLORS, BG_COLOR, GRID_COLOR, TEXT_COLOR, FONT_FAMILY

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    """Load all pre-computed CSVs and JSON files."""
    national = pd.read_csv(os.path.join(DATA_DIR, "power_rankings_national.csv"))
    predictions = pd.read_csv(os.path.join(DATA_DIR, "power_rankings_predictions.csv"))
    validation = pd.read_csv(os.path.join(DATA_DIR, "validation_results.csv"))

    with open(os.path.join(DATA_DIR, "model_metrics.json")) as f:
        metrics = json.load(f)

    fi_path = Path(DATA_DIR) / "feature_importances.json"
    feature_importances = {}
    if fi_path.exists():
        with open(fi_path) as f:
            feature_importances = json.load(f)

    nil_concentration = pd.DataFrame()
    nil_conc_path = Path(DATA_DIR) / "nil_concentration.csv"
    if nil_conc_path.exists():
        nil_concentration = pd.read_csv(nil_conc_path)

    nil_teams = pd.DataFrame()
    nil_team_path = Path(DATA_DIR) / "nil_team_analysis.csv"
    if nil_team_path.exists():
        nil_teams = pd.read_csv(nil_team_path)

    return national, predictions, validation, metrics, feature_importances, nil_concentration, nil_teams


def build_unified_df(national, predictions):
    """Merge historical and prediction data into a single DataFrame."""
    hist = national[["year", "team", "conference", "predicted_rating", "national_rank"]].copy()
    hist["is_projection"] = False
    hist["pred_lower"] = hist["predicted_rating"]
    hist["pred_upper"] = hist["predicted_rating"]

    pred = predictions[["year", "team", "conference", "predicted_rating", "national_rank",
                        "pred_lower", "pred_upper"]].copy()
    pred["is_projection"] = True

    unified = pd.concat([hist, pred], ignore_index=True)
    unified["team_color"] = unified["team"].map(team_color)
    unified["conf_color"] = unified["conference"].map(
        lambda c: CONFERENCE_COLORS.get(c, "#8b8b8b")
    )
    return unified


# ── Color scales ─────────────────────────────────────────────────────────────

team_color_domain = list(TEAM_COLORS.keys())
team_color_range = [TEAM_COLORS[t] for t in team_color_domain]

conf_color_domain = list(CONFERENCE_COLORS.keys())
conf_color_range = [CONFERENCE_COLORS[c] for c in conf_color_domain]

team_scale = alt.Scale(domain=team_color_domain, range=team_color_range)
conf_scale = alt.Scale(domain=conf_color_domain, range=conf_color_range)


# ── Chart builders ───────────────────────────────────────────────────────────

def _build_conf_legend(selection, title="Conference"):
    """Reusable small clickable conference legend."""
    return (
        alt.Chart(pd.DataFrame({
            "conference": conf_color_domain,
        }))
        .mark_circle(size=200)
        .encode(
            y=alt.Y("conference:N", title=None, axis=alt.Axis(labelFontSize=13)),
            color=alt.condition(
                selection,
                alt.Color("conference:N", scale=conf_scale, legend=None),
                alt.value("#d4d4d4"),
            ),
        )
        .add_params(selection)
        .properties(width=30, height=120, title=title)
    )


def build_rankings_chart(unified):
    """Bar chart of top 25 teams for a selected year, filtered by conference."""
    all_years = sorted(unified["year"].unique().tolist())

    year_dropdown = alt.binding_select(options=all_years, name="Year ")
    year_selection = alt.selection_point(
        fields=["year"],
        bind=year_dropdown,
        value=[{"year": all_years[-1]}],
        name="rank_year",
    )

    conf_selection = alt.selection_point(fields=["conference"], toggle=True, name="rank_conf")

    team_selection = alt.selection_point(fields=["team"], toggle=True, name="rank_team")

    conf_legend = _build_conf_legend(conf_selection)

    # Rankings bars — year + conference filtered, top 25 by rank
    base = alt.Chart(unified).transform_filter(
        year_selection
    ).transform_filter(
        conf_selection
    ).transform_window(
        filtered_rank="rank()",
        sort=[alt.SortField("predicted_rating", order="descending")],
    ).transform_filter(
        "datum.filtered_rank <= 25"
    ).transform_calculate(
        rank_label="datum.filtered_rank + '. ' + datum.team"
    )

    bars = base.mark_bar().encode(
        x=alt.X("predicted_rating:Q", title="Composite Score"),
        y=alt.Y("rank_label:N",
                sort=alt.EncodingSortField(field="filtered_rank", order="ascending"),
                title=None),
        color=alt.condition(
            team_selection,
            alt.Color("team:N", scale=team_scale, legend=None),
            alt.value("#d4d4d4"),
        ),
        tooltip=[
            alt.Tooltip("team:N", title="Team"),
            alt.Tooltip("conference:N", title="Conference"),
            alt.Tooltip("filtered_rank:Q", title="Rank"),
            alt.Tooltip("predicted_rating:Q", title="Score", format=".1f"),
            alt.Tooltip("pred_lower:Q", title="Lower Bound", format=".1f"),
            alt.Tooltip("pred_upper:Q", title="Upper Bound", format=".1f"),
        ],
    ).add_params(team_selection)

    error_bars = base.mark_errorbar(color="#3c3c3c").encode(
        x=alt.X("pred_lower:Q", title="Composite Score"),
        x2="pred_upper:Q",
        y=alt.Y("rank_label:N",
                sort=alt.EncodingSortField(field="filtered_rank", order="ascending")),
    )

    rankings = (
        (bars + error_bars)
        .add_params(year_selection)
        .properties(width=600, height=700, title="Power Rankings (Top 25)")
    )

    return conf_legend, rankings


def build_trajectory_chart(unified):
    """Line chart showing rating trajectories — each chart has its own filters."""
    conf_selection = alt.selection_point(fields=["conference"], toggle=True, name="traj_conf")
    team_selection = alt.selection_point(fields=["team"], toggle=True, name="traj_team")

    conf_legend = _build_conf_legend(conf_selection, title="Conference")

    base = alt.Chart(unified).transform_filter(
        conf_selection
    )

    # All teams — clickable, faded unless selected
    background = base.mark_line(strokeWidth=1).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("predicted_rating:Q", title="Composite Score"),
        detail="team:N",
        opacity=alt.condition(team_selection, alt.value(0.08), alt.value(0.08)),
        color=alt.value("#8b8b8b"),
    ).add_params(team_selection)

    # Selected teams highlighted
    foreground = base.mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("predicted_rating:Q", title="Composite Score"),
        color=alt.Color("team:N", scale=team_scale, legend=None),
        strokeDash=alt.StrokeDash(
            "is_projection:N",
            scale=alt.Scale(domain=[False, True], range=[[1, 0], [5, 5]]),
            legend=alt.Legend(title="Type", labelExpr="datum.value ? 'Projected' : 'Historical'"),
        ),
        tooltip=[
            alt.Tooltip("team:N"),
            alt.Tooltip("year:O"),
            alt.Tooltip("predicted_rating:Q", format=".1f", title="Score"),
        ],
    ).transform_filter(team_selection)

    # Prediction interval bands for selected teams
    bands = base.mark_area(opacity=0.12).encode(
        x=alt.X("year:O"),
        y=alt.Y("pred_lower:Q", title="Composite Score"),
        y2="pred_upper:Q",
        color=alt.Color("team:N", scale=team_scale, legend=None),
    ).transform_filter(
        team_selection
    ).transform_filter(
        "datum.is_projection"
    )

    trajectory = (
        (background + bands + foreground)
        .properties(width=600, height=350, title="Team Rating Trajectories (click lines to compare)")
    )

    return conf_legend, trajectory


def build_conference_distribution(unified):
    """Strip + box plot of conference distributions — own year and conference filters."""
    all_years = sorted(unified["year"].unique().tolist())

    year_dropdown = alt.binding_select(options=all_years, name="Year ")
    year_selection = alt.selection_point(
        fields=["year"],
        bind=year_dropdown,
        value=[{"year": all_years[-1]}],
        name="dist_year",
    )

    conf_selection = alt.selection_point(fields=["conference"], toggle=True, name="dist_conf")

    conf_legend = _build_conf_legend(conf_selection, title="Conference")

    base = alt.Chart(unified).transform_filter(
        year_selection
    ).transform_filter(
        conf_selection
    )

    boxplot = base.mark_boxplot(extent="min-max", size=30, opacity=0.3).encode(
        x=alt.X("conference:N", title="Conference"),
        y=alt.Y("predicted_rating:Q", title="Composite Score"),
        color=alt.Color("conference:N", scale=conf_scale, legend=None),
    )

    strip = base.mark_circle(size=50, opacity=0.7).encode(
        x=alt.X("conference:N", title="Conference"),
        y=alt.Y("predicted_rating:Q", title="Composite Score"),
        color=alt.Color("team:N", scale=team_scale, legend=None),
        tooltip=[
            alt.Tooltip("team:N"),
            alt.Tooltip("conference:N"),
            alt.Tooltip("predicted_rating:Q", format=".1f", title="Score"),
        ],
    )

    dist_chart = (boxplot + strip).add_params(year_selection).properties(
        width=300, height=350, title="Conference Distribution"
    )

    return conf_legend, dist_chart


def build_validation_chart(validation):
    """Predicted vs actual scatter with perfect-fit line."""
    val_data = validation.copy()

    min_val = min(val_data["actual_rating"].min(), val_data["predicted_rating"].min())
    max_val = max(val_data["actual_rating"].max(), val_data["predicted_rating"].max())

    scatter = (
        alt.Chart(val_data)
        .mark_circle(size=70, opacity=0.8)
        .encode(
            x=alt.X("actual_rating:Q", title="Actual 2025 SP+ Rating"),
            y=alt.Y("predicted_rating:Q", title="Predicted 2025 SP+ Rating"),
            color=alt.Color("team:N", scale=team_scale, legend=None),
            tooltip=[
                alt.Tooltip("team:N"),
                alt.Tooltip("actual_rating:Q", format=".1f"),
                alt.Tooltip("predicted_rating:Q", format=".1f"),
            ],
        )
    )

    perfect_line = (
        alt.Chart(pd.DataFrame({"x": [min_val, max_val], "y": [min_val, max_val]}))
        .mark_line(strokeDash=[5, 5], color="#fc4f30", strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q")
    )

    return (scatter + perfect_line).properties(
        width=400, height=350, title="Predicted vs Actual (2025 Validation)"
    )


def build_feature_importance_chart(feature_importances):
    """Horizontal bar chart of LightGBM feature importances."""
    fi_df = pd.DataFrame({
        "feature": list(feature_importances.keys()),
        "importance": list(feature_importances.values()),
    }).sort_values("importance", ascending=True)
    feature_order = fi_df["feature"].tolist()

    return (
        alt.Chart(fi_df)
        .mark_bar(color="#008fd5")
        .encode(
            x=alt.X("importance:Q", title="Importance (split count)"),
            y=alt.Y("feature:N", sort=feature_order, title="Feature"),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("importance:Q"),
            ],
        )
        .properties(width=400, height=max(250, len(fi_df) * 22), title="LightGBM Feature Importances")
    )


def build_nil_gini_chart(nil_concentration):
    """Line chart of Gini coefficient over time with NIL era marker."""
    nil_year = 2022

    line = (
        alt.Chart(nil_concentration)
        .mark_line(point=True, color="#008fd5", strokeWidth=2.5)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("gini_elite:Q", title="Gini Coefficient"),
            tooltip=[
                alt.Tooltip("year:O"),
                alt.Tooltip("gini_elite:Q", format=".4f"),
                alt.Tooltip("total_elite:Q", title="Total Elite"),
                alt.Tooltip("num_teams:Q", title="Teams"),
            ],
        )
    )

    rule = (
        alt.Chart(pd.DataFrame({"year": [str(nil_year)]}))
        .mark_rule(strokeDash=[5, 5], color="#fc4f30", strokeWidth=1.5)
        .encode(x="year:O")
    )

    text = (
        alt.Chart(pd.DataFrame({"year": [str(nil_year)], "label": ["NIL era"]}))
        .mark_text(align="left", dx=5, dy=-10, color="#fc4f30", fontSize=12)
        .encode(x="year:O", text="label:N")
    )

    return (line + rule + text).properties(
        width=400, height=300, title="Elite Talent Gini Coefficient by Year"
    )


def build_nil_gainers_losers_chart(nil_teams):
    """Bar chart of top gainers and losers post-NIL."""
    nil_teams = nil_teams.dropna(subset=["elite_delta"])
    top_gainers = nil_teams.head(10)
    top_losers = nil_teams.tail(10).sort_values("elite_delta")
    plot_df = pd.concat([top_gainers, top_losers]).copy()
    plot_df = plot_df.sort_values("elite_delta")
    team_order_nil = plot_df["team"].tolist()

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("elite_delta:Q", title="Change in Avg Elite Players (Post - Pre NIL)"),
            y=alt.Y("team:N", sort=team_order_nil, title=None),
            color=alt.Color("team:N", scale=team_scale, legend=None),
            tooltip=[
                alt.Tooltip("team:N"),
                alt.Tooltip("elite_delta:Q", format=".2f"),
            ],
        )
        .properties(width=400, height=450, title="Biggest Elite Talent Gainers & Losers (Post-NIL)")
    )


def build_metrics_table(metrics):
    """Text-based metrics display as an Altair chart."""
    rows = []
    for key, val in metrics.items():
        label = key.replace("_", " ").title()
        rows.append({"metric": label, "value": f"{val:.4f}" if isinstance(val, float) else str(val)})

    metrics_df = pd.DataFrame(rows)

    return (
        alt.Chart(metrics_df)
        .mark_text(align="left", fontSize=13, font=FONT_FAMILY)
        .encode(
            y=alt.Y("metric:N", title=None, axis=alt.Axis(labelFontSize=13, labelFontWeight="bold")),
            text="value:N",
        )
        .properties(width=200, height=len(rows) * 28, title="Model Evaluation Metrics")
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Increase max rows for large datasets
    alt.data_transformers.disable_max_rows()

    print("Loading data...")
    national, predictions, validation, metrics, feature_importances, nil_concentration, nil_teams = load_data()

    print("Building unified dataset...")
    unified = build_unified_df(national, predictions)

    print("Building charts...")

    # Row 1: Conference legend + Rankings
    rank_conf_legend, rankings = build_rankings_chart(unified)
    row1 = rank_conf_legend | rankings

    # Row 2: Trajectories + Conference distribution (each with own filters)
    traj_conf_legend, trajectory = build_trajectory_chart(unified)
    dist_conf_legend, conf_dist = build_conference_distribution(unified)
    row2 = (traj_conf_legend | trajectory) | (dist_conf_legend | conf_dist)

    # Row 3: Validation + Feature importances
    validation_chart = build_validation_chart(validation)
    charts_row3 = [validation_chart]
    if feature_importances:
        fi_chart = build_feature_importance_chart(feature_importances)
        charts_row3.append(fi_chart)
    row3 = alt.hconcat(*charts_row3)

    # Row 4: NIL analysis
    nil_charts = []
    if not nil_concentration.empty:
        nil_charts.append(build_nil_gini_chart(nil_concentration))
    if not nil_teams.empty:
        nil_charts.append(build_nil_gainers_losers_chart(nil_teams))

    # Row 5: Metrics table
    metrics_chart = build_metrics_table(metrics)

    # Assemble full dashboard
    rows = [row1, row2, row3]
    if nil_charts:
        row4 = alt.hconcat(*nil_charts)
        rows.append(row4)
    rows.append(metrics_chart)

    dashboard = alt.vconcat(
        *rows,
        title=alt.Title(
            "College Football Power Rankings",
            subtitle="LightGBM-powered projections with prediction intervals  |  Each chart has its own conference and team filters",
            fontSize=24,
            subtitleFontSize=14,
            color=TEXT_COLOR,
            subtitleColor="#666666",
        ),
    ).configure(
        background=BG_COLOR,
        font=FONT_FAMILY,
    ).configure_view(
        strokeWidth=0,
    ).configure_axis(
        gridColor=GRID_COLOR,
        labelColor=TEXT_COLOR,
        titleColor=TEXT_COLOR,
    ).configure_title(
        color=TEXT_COLOR,
    ).configure_legend(
        labelColor=TEXT_COLOR,
        titleColor=TEXT_COLOR,
    )

    # Save and open
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "power_rankings.html")
    print(f"Saving to {output_path}...")
    dashboard.save(output_path)
    print(f"Saved {output_path}")

    # Open in default browser
    webbrowser.open(output_path)
    print("Opened in browser.")


if __name__ == "__main__":
    main()
