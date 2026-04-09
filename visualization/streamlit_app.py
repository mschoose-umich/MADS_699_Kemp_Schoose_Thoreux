import json
import ast
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk

from team_colors import TEAM_COLORS, DEFAULT_COLOR

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

st.set_page_config(page_title="CFB Power Rankings & Recruiting", layout="wide")


@st.cache_data
def load_data():
    # Historical walk-forward rankings (2015-2025)
    hist_nat = pd.read_csv(os.path.join(DATA_DIR, "power_rankings_national.csv"))
    hist_nat["type"] = "Historical"
    hist_nat["pred_lower"] = hist_nat["predicted_rating"]
    hist_nat["pred_upper"] = hist_nat["predicted_rating"]

    hist_conf = pd.read_csv(os.path.join(DATA_DIR, "power_rankings_conference.csv"))[
        ["year", "team", "conference", "predicted_rating", "conference_rank"]
    ]
    hist_nat = hist_nat.merge(
        hist_conf[["year", "team", "conference_rank"]], on=["year", "team"], how="left"
    )

    # Multi-year predictions (2026-2028) with confidence intervals
    pred = pd.read_csv(os.path.join(DATA_DIR, "power_rankings_predictions.csv"))
    pred["type"] = "Predicted"

    keep = ["year", "team", "conference", "predicted_rating",
            "pred_lower", "pred_upper", "national_rank", "conference_rank", "type"]
    rankings = pd.concat(
        [hist_nat[keep], pred[keep]], ignore_index=True
    )

    # Recruiting pipeline map data
    recruits = pd.read_csv(os.path.join(DATA_DIR, "recruit_data.csv"))
    team_locs = pd.read_csv(os.path.join(DATA_DIR, "team_locations.csv"))

    def parse_hometown(x):
        try:
            if pd.isna(x):
                return None, None
            info = ast.literal_eval(x)
            return info.get("latitude"), info.get("longitude")
        except Exception:
            return None, None

    recruits["recruit_lat"], recruits["recruit_lon"] = zip(
        *recruits["hometownInfo"].apply(parse_hometown)
    )
    map_df = recruits.dropna(subset=["recruit_lat", "recruit_lon", "committedTo"])
    map_df = map_df.merge(
        team_locs[["team", "latitude", "longitude", "conference"]],
        left_on="committedTo", right_on="team", how="inner",
    )
    map_df = map_df.rename(columns={"latitude": "team_lat", "longitude": "team_lon"})
    map_df = map_df.dropna(subset=["team_lat", "team_lon"])

    # Model metrics
    with open(os.path.join(DATA_DIR, "model_metrics.json")) as f:
        metrics = json.load(f)

    with open(os.path.join(DATA_DIR, "feature_importances.json")) as f:
        importances = json.load(f)

    return rankings, map_df, metrics, importances


def color_scale(teams):
    domain = list(teams)
    rng = [TEAM_COLORS.get(t, DEFAULT_COLOR) for t in domain]
    return alt.Scale(domain=domain, range=rng)


def main():
    st.title("CFB Power Rankings & Recruiting")

    rankings, map_df, metrics, importances = load_data()

    # ── Shared data for filters ─────────────────────────────────────────────
    all_confs = sorted(rankings["conference"].dropna().unique())
    hist_years = sorted(rankings[rankings["type"] == "Historical"]["year"].unique())

    # ── Tab layout ───────────────────────────────────────────────────────────
    tab_rank, tab_proj, tab_model, tab_map = st.tabs([
        "Power Rankings", "Multi-Year Projections", "Model Performance", "Recruiting Map"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Power Rankings (historical + current)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_rank:
        # Per-tab filters
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            rank_confs = st.multiselect(
                "Conferences", all_confs, default=[],
                help="Leave empty for national view", key="rank_confs"
            )
        with rc2:
            selected_year = st.slider(
                "Year", int(min(hist_years)), int(max(hist_years)),
                int(max(hist_years)), key="rank_year"
            )
        with rc3:
            sort_order = st.radio(
                "Sort Order", ["Best to Worst", "Worst to Best"], key="rank_sort"
            )

        is_asc = sort_order == "Worst to Best"
        sort_field = "conference_rank" if rank_confs else "national_rank"
        chart_sort = "descending" if is_asc else "ascending"
        pd_asc = not is_asc

        st.subheader(f"{selected_year} Power Rankings")

        year_data = rankings[rankings["year"] == selected_year].copy()
        if rank_confs:
            year_data = year_data[year_data["conference"].isin(rank_confs)]
            year_data = year_data.sort_values("conference_rank", ascending=pd_asc)
            year_data["display_rank"] = (
                year_data["conference_rank"].astype(int).astype(str)
                + " (" + year_data["conference"] + ")"
            )
        else:
            year_data = year_data.sort_values("national_rank", ascending=pd_asc)
            year_data["display_rank"] = year_data["national_rank"].astype(int).astype(str)

        year_data = year_data.head(25)
        year_data["team_label"] = year_data["display_rank"] + ". " + year_data["team"]

        if year_data.empty:
            st.warning("No data for selected filters.")
        else:
            cs = color_scale(year_data["team"].unique())
            base = alt.Chart(year_data).encode(
                y=alt.Y(
                    "team_label:N",
                    sort=alt.EncodingSortField(field=sort_field, op="min", order=chart_sort),
                    title="",
                ),
                color=alt.Color("team:N", scale=cs, legend=None),
                tooltip=[
                    alt.Tooltip("national_rank:Q", title="National Rank"),
                    alt.Tooltip("team:N"),
                    alt.Tooltip("conference:N"),
                    alt.Tooltip("predicted_rating:Q", title="Predicted SP+ Rating", format=".2f"),
                    alt.Tooltip("type:N", title="Data Type"),
                ],
            )
            bars = base.mark_bar().encode(
                x=alt.X("predicted_rating:Q", title="Predicted SP+ Rating")
            )
            labels = base.mark_text(align="left", dx=3, fontSize=11).encode(
                x=alt.X("predicted_rating:Q"),
                text=alt.Text("predicted_rating:Q", format=".1f"),
                color=alt.value("black"),
            )
            title_prefix = "Bottom 25" if is_asc else "Top 25"
            chart = (bars + labels).properties(
                height=alt.Step(25),
                title=f"{title_prefix} Predicted SP+ Ratings ({selected_year})",
            ).configure_view(strokeWidth=0)
            st.altair_chart(chart, use_container_width=True)

        # Historical trend for top teams
        st.subheader("Historical Ratings Trend")
        top_teams = (
            rankings[rankings["year"] == selected_year]
            .nsmallest(10, "national_rank")["team"]
            .tolist()
        )
        trend_data = rankings[
            (rankings["team"].isin(top_teams)) & (rankings["type"] == "Historical")
        ].copy()

        if not trend_data.empty:
            trend = alt.Chart(trend_data).mark_line(point=True).encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("predicted_rating:Q", title="Predicted SP+ Rating"),
                color=alt.Color("team:N", scale=color_scale(top_teams)),
                tooltip=["year:O", "team:N", alt.Tooltip("predicted_rating:Q", format=".2f"),
                         "national_rank:Q"],
            ).properties(height=300, title="Top 10 Teams — Historical Ratings")
            st.altair_chart(trend, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Multi-Year Projections (2026-2028)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_proj:
        st.subheader("2026-2028 Power Rankings Projections")
        st.caption("90% prediction intervals reflect uncertainty compounding over forecast horizon.")

        # Per-tab filters
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            proj_confs = st.multiselect(
                "Conferences", all_confs, default=[],
                help="Leave empty for national view", key="proj_confs"
            )
        with pc2:
            show_intervals = st.checkbox("Show prediction intervals", value=True, key="proj_intervals")
        with pc3:
            top_n = st.slider("Teams to show", 5, 20, 10, key="proj_top_n")

        pred_data = rankings[rankings["type"] == "Predicted"].copy()
        if proj_confs:
            pred_data = pred_data[pred_data["conference"].isin(proj_confs)]
        top_proj_teams = (
            pred_data[pred_data["year"] == pred_data["year"].min()]
            .nsmallest(top_n, "national_rank")["team"]
            .tolist()
        )
        pred_filtered = pred_data[pred_data["team"].isin(top_proj_teams)]

        if pred_filtered.empty:
            st.warning("No projection data for selected conferences.")
        else:
            cs_proj = color_scale(top_proj_teams)

            # Bar chart for each projection year
            for yr in sorted(pred_filtered["year"].unique()):
                yr_data = pred_filtered[pred_filtered["year"] == yr].sort_values("national_rank")
                yr_data["team_label"] = yr_data["national_rank"].astype(int).astype(str) + ". " + yr_data["team"]

                base = alt.Chart(yr_data).encode(
                    y=alt.Y("team_label:N",
                            sort=alt.EncodingSortField(field="national_rank", op="min", order="ascending"),
                            title=""),
                    color=alt.Color("team:N", scale=cs_proj, legend=None),
                    tooltip=[
                        alt.Tooltip("national_rank:Q", title="Rank"),
                        "team:N", "conference:N",
                        alt.Tooltip("predicted_rating:Q", title="Projected Rating", format=".2f"),
                        alt.Tooltip("pred_lower:Q", title="90% Lower", format=".2f"),
                        alt.Tooltip("pred_upper:Q", title="90% Upper", format=".2f"),
                    ],
                )
                bars = base.mark_bar().encode(
                    x=alt.X("predicted_rating:Q", title="Projected SP+ Rating")
                )
                labels = base.mark_text(align="left", dx=3, fontSize=11).encode(
                    x="predicted_rating:Q",
                    text=alt.Text("predicted_rating:Q", format=".1f"),
                    color=alt.value("black"),
                )

                if show_intervals:
                    err = base.mark_errorbar(ticks=True).encode(
                        x=alt.X("pred_lower:Q", title="Projected SP+ Rating"),
                        x2="pred_upper:Q",
                    )
                    chart = (bars + err + labels).properties(
                        height=alt.Step(28), title=f"Projected {yr} Rankings (90% PI)"
                    ).configure_view(strokeWidth=0)
                else:
                    chart = (bars + labels).properties(
                        height=alt.Step(28), title=f"Projected {yr} Rankings"
                    ).configure_view(strokeWidth=0)

                st.altair_chart(chart, use_container_width=True)

            # Trajectory lines: historical + projected for top teams
            st.subheader("Rating Trajectory (Historical + Projected)")
            all_traj = rankings[rankings["team"].isin(top_proj_teams)].copy()
            hist_lines = alt.Chart(all_traj[all_traj["type"] == "Historical"]).mark_line(
                point=True, strokeDash=[0, 0]
            ).encode(
                x=alt.X("year:O", title="Year"),
                y=alt.Y("predicted_rating:Q", title="Predicted SP+ Rating"),
                color=alt.Color("team:N", scale=cs_proj),
                tooltip=["year:O", "team:N", alt.Tooltip("predicted_rating:Q", format=".2f")],
            )
            proj_lines = alt.Chart(all_traj[all_traj["type"] == "Predicted"]).mark_line(
                point=True, strokeDash=[4, 4]
            ).encode(
                x=alt.X("year:O"),
                y=alt.Y("predicted_rating:Q"),
                color=alt.Color("team:N", scale=cs_proj),
                tooltip=["year:O", "team:N",
                         alt.Tooltip("predicted_rating:Q", format=".2f"),
                         alt.Tooltip("pred_lower:Q", title="Lower 90%", format=".2f"),
                         alt.Tooltip("pred_upper:Q", title="Upper 90%", format=".2f")],
            )
            traj_chart = (hist_lines + proj_lines).properties(
                height=350, title="Solid = Historical | Dashed = Projected"
            )
            st.altair_chart(traj_chart, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Model Performance
    # ══════════════════════════════════════════════════════════════════════════
    with tab_model:
        st.subheader("Model Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R²", f"{metrics.get('r2', 'N/A'):.3f}")
        col2.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.3f}")
        col3.metric("Spearman r", f"{metrics.get('spearman_r', 'N/A'):.3f}")
        col4.metric("Interval Coverage", f"{metrics.get('interval_coverage', 'N/A'):.1%}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Rolling OOS RMSE", f"{metrics.get('rolling_val_rmse', 'N/A'):.3f}",
                    help="Walk-forward validation 2020-2025")
        col6.metric("vs. Persistence", f"{metrics.get('baseline_rmse', 'N/A'):.3f}",
                    help="Persistence baseline RMSE (next year = this year)")
        col7.metric("vs. Ridge", f"{metrics.get('ridge_rmse', 'N/A'):.3f}",
                    help="OLS Ridge baseline RMSE")
        col8.metric("Ranker Spearman r", f"{metrics.get('ranker_spearman_r', 'N/A'):.3f}",
                    help="LightGBM Ranker (ordinal) Spearman r — regression model wins")

        st.divider()
        st.subheader("Feature Importances")
        imp_df = pd.DataFrame(
            list(importances.items()), columns=["feature", "importance"]
        ).sort_values("importance", ascending=False).head(20)

        imp_chart = alt.Chart(imp_df).mark_bar().encode(
            x=alt.X("importance:Q", title="Importance Score"),
            y=alt.Y("feature:N", sort="-x", title=""),
            color=alt.value("#4C78A8"),
            tooltip=["feature:N", "importance:Q"],
        ).properties(height=400, title="Top 20 Feature Importances (LightGBM)")
        st.altair_chart(imp_chart, use_container_width=True)

        st.divider()
        st.subheader("Rolling Out-of-Sample RMSE by Year")
        rolling = metrics.get("rolling_val_per_year", {})
        if rolling:
            rolling_df = pd.DataFrame(
                [(int(yr), rmse) for yr, rmse in rolling.items()],
                columns=["year", "rmse"]
            )
            rolling_chart = alt.Chart(rolling_df).mark_bar(color="#E45756").encode(
                x=alt.X("year:O", title="Validation Year"),
                y=alt.Y("rmse:Q", title="RMSE", scale=alt.Scale(zero=False)),
                tooltip=["year:O", alt.Tooltip("rmse:Q", format=".3f")],
            ).properties(height=250, title="Walk-Forward Validation RMSE (LightGBM)")
            baseline_line = alt.Chart(pd.DataFrame({"y": [metrics["baseline_rmse"]]})).mark_rule(
                strokeDash=[4, 4], color="gray"
            ).encode(y="y:Q")
            st.altair_chart(rolling_chart + baseline_line, use_container_width=True)
            st.caption("Dashed line = persistence baseline RMSE")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Recruiting Map
    # ══════════════════════════════════════════════════════════════════════════
    with tab_map:
        st.subheader("Recruiting Pipeline")
        st.markdown("Arc from recruit's hometown to their committed school.")

        recruit_years = sorted(map_df["year"].dropna().unique().astype(int).tolist())
        if not recruit_years:
            st.info("No recruit data available.")
        else:
            # Per-tab filters
            mc1, mc2 = st.columns(2)
            with mc1:
                map_confs = st.multiselect(
                    "Conferences", all_confs, default=[],
                    help="Leave empty for all conferences", key="map_confs"
                )
            with mc2:
                map_year = st.select_slider(
                    "Recruit Class Year", options=recruit_years,
                    value=recruit_years[-1], key="map_year"
                )
            map_filtered = map_df[map_df["year"] == map_year]
            if map_confs:
                map_filtered = map_filtered[map_filtered["conference"].isin(map_confs)]

            if map_filtered.empty:
                st.warning(f"No recruit data for {map_year} with current filters.")
            else:
                def get_rgb(team_name):
                    h = TEAM_COLORS.get(team_name, DEFAULT_COLOR).lstrip("#")
                    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

                map_filtered = map_filtered.copy()
                map_filtered["color"] = map_filtered["committedTo"].apply(
                    lambda t: get_rgb(t) + [160]
                )

                arc_layer = pdk.Layer(
                    "ArcLayer",
                    data=map_filtered,
                    get_source_position=["recruit_lon", "recruit_lat"],
                    get_target_position=["team_lon", "team_lat"],
                    get_source_color=[255, 0, 0, 120],
                    get_target_color="color",
                    get_width="stars",
                    pickable=True,
                    auto_highlight=True,
                )
                scatter_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_filtered,
                    get_position=["team_lon", "team_lat"],
                    get_color="color",
                    get_radius=15000,
                    pickable=True,
                )

                mid_lat = map_filtered["team_lat"].mean()
                mid_lon = map_filtered["team_lon"].mean()
                view = pdk.ViewState(
                    latitude=mid_lat if not pd.isna(mid_lat) else 39.83,
                    longitude=mid_lon if not pd.isna(mid_lon) else -98.58,
                    zoom=3.5, pitch=45,
                )
                st.pydeck_chart(pdk.Deck(
                    layers=[arc_layer, scatter_layer],
                    initial_view_state=view,
                    tooltip={"text": "{name} ({stars}★)\nFrom: {city}, {stateProvince}\nTo: {committedTo}"},
                    map_style=pdk.map_styles.LIGHT,
                ))


if __name__ == "__main__":
    main()
