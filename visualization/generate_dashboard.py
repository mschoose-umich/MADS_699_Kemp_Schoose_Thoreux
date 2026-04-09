import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import altair as alt
from team_colors import TEAM_COLORS, DEFAULT_COLOR

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

def main():
    # Load data
    df_hist = pd.read_csv(os.path.join(DATA_DIR, "power_rankings_national.csv"))
    df_pred = pd.read_csv(os.path.join(DATA_DIR, "power_rankings_predictions.csv"))
    
    cols = ["year", "team", "conference", "predicted_rating", "national_rank"]
    df_hist["type"] = "Historical"
    df_pred["type"] = "Predicted"
    
    df = pd.concat([df_hist[cols + ["type"]], df_pred[cols + ["type"]]], ignore_index=True)
    
    teams_in_data = df['team'].unique().tolist()
    color_domain = []
    color_range = []
    for team in teams_in_data:
        color_domain.append(team)
        color_range.append(TEAM_COLORS.get(team, DEFAULT_COLOR))
        
    color_scale = alt.Scale(domain=color_domain, range=color_range)

    # Helper for conference selector rects
    def make_conf_selector(selection, title="Conferences"):
        return alt.Chart(df).mark_rect(size=30).encode(
            y=alt.Y('conference:N', title=None, axis=alt.Axis(orient='left', labelFontSize=12)),
            color=alt.condition(selection, alt.value('#2c3e50'), alt.value('#e0e0e0')),
            tooltip='conference:N'
        ).transform_aggregate(
            count='count()',
            groupby=['conference']
        ).add_params(
            selection
        ).properties(
            width=50,
            title=title
        )

    # 1. Selections — independent per chart
    bar_conf = alt.selection_point(fields=['conference'], toggle=True, name='bar_conf')
    bar_team = alt.selection_point(fields=['team'], toggle=True, name='bar_team')

    line_conf = alt.selection_point(fields=['conference'], toggle=True, name='line_conf')
    line_team = alt.selection_point(fields=['team'], toggle=True, name='line_team')

    min_year = int(df['year'].min())
    max_year = int(df['year'].max())
    year_slider = alt.binding_range(min=min_year, max=max_year, step=1, name="Select Year for Rankings: ")
    year_selection = alt.selection_point(fields=['year'], bind=year_slider, value=max_year)

    # Conference selectors — one per chart section
    bar_conf_selector = make_conf_selector(bar_conf, title="Bar Conferences")
    line_conf_selector = make_conf_selector(line_conf, title="Line Conferences")

    # Bar Chart for Rankings (Filtered by Year and its own Conference)
    bars = alt.Chart(df).mark_bar().encode(
        y=alt.Y('team:N', sort=alt.EncodingSortField(field="predicted_rating", op="mean", order="descending"), title=None),
        x=alt.X('predicted_rating:Q', title="Composite Score"),
        color=alt.Color('team:N', scale=color_scale, legend=None),
        opacity=alt.condition(bar_team, alt.value(1.0), alt.value(0.3)),
        tooltip=['year', 'team', 'conference', 'predicted_rating', 'national_rank', 'type']
    ).add_params(
        year_selection,
        bar_team
    ).transform_filter(
        year_selection
    ).transform_filter(
        bar_conf
    ).properties(
        width=250,
        height=alt.Step(20),
        title="Rankings (Click bars to select teams)"
    )

    # Text labels for the bars
    text = bars.mark_text(
        align='left',
        baseline='middle',
        dx=3
    ).encode(
        text=alt.Text('predicted_rating:Q', format='.1f'),
        color=alt.value('black')
    )

    bar_chart = (bars + text)

    # Line Chart — its own conference and team selections
    line_chart = alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('year:O', title="Year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('predicted_rating:Q', title="Composite Score", scale=alt.Scale(zero=False)),
        color=alt.Color('team:N', scale=color_scale, legend=None),
        strokeDash=alt.condition(
            alt.datum.type == 'Predicted',
            alt.value([5, 5]),
            alt.value([0])
        ),
        tooltip=['year', 'team', 'conference', 'predicted_rating', 'national_rank', 'type'],
        opacity=alt.condition(line_team, alt.value(1.0), alt.value(0.05))
    ).add_params(
        line_team
    ).transform_filter(
        line_conf
    ).properties(
        width=600,
        height=400,
        title="Power Rankings Over Time (Solid = Historical, Dashed = Predicted)"
    )

    # Dashboard Assembly — each chart paired with its own conference selector
    bar_section = alt.hconcat(bar_conf_selector, bar_chart, spacing=10)
    line_section = alt.hconcat(line_conf_selector, line_chart, spacing=10)

    dashboard = alt.hconcat(
        bar_section,
        line_section,
        spacing=30
    ).resolve_scale(
        color='independent',
        y='independent'
    ).configure_view(
        stroke=None
    ).configure_title(
        fontSize=16,
        font='Arial',
        anchor='middle'
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "power_rankings_dashboard.html")
    dashboard.save(output_path)
    print(f"Dashboard generated successfully: {output_path}")

if __name__ == "__main__":
    main()
