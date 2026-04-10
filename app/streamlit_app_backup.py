import ast
import os
import sys
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import streamlit as st
import pandas as pd
import numpy as np
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

    # Win probability data
    wp_games_path = os.path.join(DATA_DIR, "2025_games_clean.csv")
    wp_rosters_path = os.path.join(DATA_DIR, "2025_rosters.csv")
    wp_model_path = os.path.join(DATA_DIR, "production_model.pkl")

    if os.path.exists(wp_games_path) and os.path.exists(wp_rosters_path):
        wp_games = pd.read_csv(wp_games_path)
        wp_games["startDate"] = pd.to_datetime(wp_games["startDate"], errors="coerce")
        wp_rosters = pd.read_csv(wp_rosters_path)
    else:
        wp_games, wp_rosters = None, None

    if os.path.exists(wp_model_path):
        with open(wp_model_path, "rb") as f:
            wp_model = pickle.load(f)
    else:
        wp_model = None

    return rankings, map_df, wp_games, wp_rosters, wp_model


def color_scale(teams):
    domain = list(teams)
    rng = [TEAM_COLORS.get(t, DEFAULT_COLOR) for t in domain]
    return alt.Scale(domain=domain, range=rng)


def main():
    st.title("CFB Power Rankings & Recruiting")

    rankings, map_df, wp_games, wp_rosters, wp_model = load_data()

    # ── Shared data for filters ─────────────────────────────────────────────
    all_confs = sorted(rankings["conference"].dropna().unique())
    hist_years = sorted(rankings[rankings["type"] == "Historical"]["year"].unique())

    # ── Tab layout ───────────────────────────────────────────────────────────
    tab_rank, tab_proj, tab_map, tab_wp = st.tabs([
        "Power Rankings", "Multi-Year Projections",
        "Recruiting Map", "Win Probability Simulator",
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


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — Win Probability Simulator
    # ══════════════════════════════════════════════════════════════════════════
    with tab_wp:
        if wp_games is None or wp_rosters is None or wp_model is None:
            st.warning(
                "Win probability data not found. Run the full pipeline first:\n"
                "  `data_pipeline/get_team_starters.py` → "
                "`data_pipeline/build_matchup_features.py` → "
                "`analysis/build_win_model.py`"
            )
        else:
            _wp_positions = [
                "QB1", "RB1", "WR1", "WR2", "TE1",
                "OL1", "OL2", "OL3", "OL4", "OL5",
                "DL1", "DL2", "DL3", "DL4",
                "LB1", "LB2", "LB3",
                "DB1", "DB2", "DB3", "DB4",
                "PK1", "P1",
            ]
            _wp_features = [
                "QB_diff", "OL_adv_1", "OL_adv_2", "WR_adv_1", "WR_adv_2",
                "RB_adv_1", "RB_adv_2", "TE_diff", "PK_diff", "P_diff",
                "QB1_vs_DB_2", "QB2_vs_DB_1", "PASS_SYNERGY_diff", "team_1_is_home",
            ]
            _rating_options = [0.0, 2.0, 3.0, 4.0, 5.0]

            def _wp_get_roster(team_name):
                exact = wp_rosters.loc[wp_rosters["team"] == team_name]
                if not exact.empty:
                    return exact.iloc[0].to_dict()
                canon = str(team_name).strip()
                fallback = wp_rosters.loc[wp_rosters["team"].str.strip() == canon]
                if not fallback.empty:
                    return fallback.iloc[0].to_dict()
                return None

            def _wp_aggregates(row):
                row = row.copy()
                row["OL_avg"] = np.mean([row["OL1"], row["OL2"], row["OL3"], row["OL4"], row["OL5"]])
                row["WR_avg"] = np.mean([row["WR1"], row["WR2"]])
                row["DL_avg"] = np.mean([row["DL1"], row["DL2"], row["DL3"], row["DL4"]])
                row["LB_avg"] = np.mean([row["LB1"], row["LB2"], row["LB3"]])
                row["DB_avg"] = np.mean([row["DB1"], row["DB2"], row["DB3"], row["DB4"]])
                return row

            def _wp_matchup(t1, t2, t1_home):
                s1 = (2.0 * t1["QB1"] + 1.0 * t1["WR_avg"] + 0.8 * t1["TE1"]) / 3.8
                s2 = (2.0 * t2["QB1"] + 1.0 * t2["WR_avg"] + 0.8 * t2["TE1"]) / 3.8
                return {
                    "QB_diff": t1["QB1"] - t2["QB1"],
                    "OL_adv_1": t1["OL_avg"] - t2["DL_avg"],
                    "OL_adv_2": t2["OL_avg"] - t1["DL_avg"],
                    "WR_adv_1": t1["WR_avg"] - t2["DB_avg"],
                    "WR_adv_2": t2["WR_avg"] - t1["DB_avg"],
                    "RB_adv_1": t1["RB1"] - t2["LB_avg"],
                    "RB_adv_2": t2["RB1"] - t1["LB_avg"],
                    "TE_diff": t1["TE1"] - t2["TE1"],
                    "PK_diff": t1["PK1"] - t2["PK1"],
                    "P_diff": t1["P1"] - t2["P1"],
                    "QB1_vs_DB_2": t1["QB1"] - t2["DB_avg"],
                    "QB2_vs_DB_1": t2["QB1"] - t1["DB_avg"],
                    "PASS_SYNERGY_diff": s1 - s2,
                    "team_1_is_home": int(t1_home),
                }

            def _wp_predict(team_profile, team_games_df, selected):
                rows, valid_idx, skipped = [], [], []
                for idx, game in team_games_df.iterrows():
                    is_home = game["homeTeam"] == selected
                    opponent = game["awayTeam"] if is_home else game["homeTeam"]
                    opp_row = _wp_get_roster(opponent)
                    if opp_row is None:
                        skipped.append(opponent)
                        continue
                    opp_row = _wp_aggregates(opp_row)
                    rows.append(_wp_matchup(team_profile, opp_row, is_home))
                    valid_idx.append(idx)
                if not rows:
                    return pd.Series(dtype=float), [], skipped
                X = pd.DataFrame(rows)[_wp_features]
                probs = wp_model.predict_proba(X)[:, 1]
                return pd.Series(probs, index=valid_idx), valid_idx, skipped

            st.subheader("Win Probability Simulator")

            wp_teams = sorted(
                set(wp_games["homeTeam"]).union(set(wp_games["awayTeam"]))
            )
            selected_team = st.selectbox("Select a Team", wp_teams, key="wp_team")
            team_roster = _wp_get_roster(selected_team)

            if team_roster is None:
                st.error(f"No roster found for {selected_team}")
            else:
                team_games_filtered = wp_games[
                    (wp_games["homeTeam"] == selected_team)
                    | (wp_games["awayTeam"] == selected_team)
                ].sort_values("startDate")

                if team_games_filtered.empty:
                    st.warning(f"No games found for {selected_team}.")
                else:
                    left_col, right_col = st.columns([1.0, 1.8])

                    with left_col:
                        st.markdown("**Adjust Roster Ratings**")
                        edited = {}
                        sel_cols = st.columns(2)
                        for i, pos in enumerate(_wp_positions):
                            cur = float(team_roster.get(pos, 0.0))
                            default_idx = (
                                _rating_options.index(cur) if cur in _rating_options else 0
                            )
                            with sel_cols[i % 2]:
                                edited[pos] = st.selectbox(
                                    pos, options=_rating_options,
                                    index=default_idx,
                                    key=f"wp_{selected_team}_{pos}",
                                )

                    original = _wp_aggregates(team_roster.copy())
                    modified = _wp_aggregates({**team_roster, **edited})

                    changes = []
                    for pos in _wp_positions:
                        orig_val = float(team_roster.get(pos, 0.0))
                        new_val = float(edited[pos])
                        if orig_val != new_val:
                            changes.append({
                                "Position": pos, "Original": orig_val,
                                "Updated": new_val, "Delta": new_val - orig_val,
                            })
                    has_changes = len(changes) > 0

                    orig_probs, orig_idx, skip1 = _wp_predict(
                        original, team_games_filtered, selected_team
                    )
                    results = team_games_filtered.loc[orig_idx].copy()

                    if results.empty:
                        st.error("No valid predictions for this team.")
                    else:
                        results["original_prob"] = orig_probs.loc[orig_idx].values

                        if has_changes:
                            upd_probs, upd_idx, skip2 = _wp_predict(
                                modified, team_games_filtered, selected_team
                            )
                            valid = sorted(set(orig_idx) & set(upd_idx))
                            results = team_games_filtered.loc[valid].copy()
                            results["original_prob"] = orig_probs.loc[valid].values
                            results["updated_prob"] = upd_probs.loc[valid].values
                            results["delta"] = results["updated_prob"] - results["original_prob"]
                            skipped = sorted(set(skip1) | set(skip2))
                        else:
                            results["updated_prob"] = np.nan
                            results["delta"] = np.nan
                            skipped = sorted(set(skip1))

                        if skipped:
                            st.warning("Skipped (missing roster): " + ", ".join(skipped))

                        results["opponent"] = results.apply(
                            lambda r: f"vs {r['awayTeam']}"
                            if r["homeTeam"] == selected_team
                            else f"@ {r['homeTeam']}",
                            axis=1,
                        )

                        with right_col:
                            st.markdown("**Win Probability by Game**")
                            orig_wins = int((results["original_prob"] > 0.5).sum())
                            st.markdown(
                                f"Original projected record: **{orig_wins}-{len(results) - orig_wins}**"
                            )
                            if has_changes:
                                new_wins = int((results["updated_prob"] > 0.5).sum())
                                st.markdown(
                                    f"Updated projected record: **{new_wins}-{len(results) - new_wins}** "
                                    f"({new_wins - orig_wins:+} wins)"
                                )

                            chart_df = results.reset_index(drop=True).copy()
                            bars = alt.Chart(chart_df).mark_bar().encode(
                                x=alt.X("opponent:N", sort=None, title="Opponent"),
                                y=alt.Y("original_prob:Q", title="Win Probability",
                                         scale=alt.Scale(domain=[0, 1])),
                                color=alt.value("lightgray"),
                                tooltip=[
                                    alt.Tooltip("startDate:T", title="Date"),
                                    alt.Tooltip("homeTeam:N", title="Home"),
                                    alt.Tooltip("awayTeam:N", title="Away"),
                                    alt.Tooltip("original_prob:Q", title="Prob", format=".3f"),
                                ],
                            )

                            if has_changes:
                                chart_df["delta_start"] = chart_df[
                                    ["original_prob", "updated_prob"]
                                ].min(axis=1)
                                chart_df["delta_end"] = chart_df[
                                    ["original_prob", "updated_prob"]
                                ].max(axis=1)
                                delta_bars = alt.Chart(chart_df).mark_bar().encode(
                                    x=alt.X("opponent:N", sort=None, title="Opponent"),
                                    y=alt.Y("delta_end:Q", scale=alt.Scale(domain=[0, 1])),
                                    y2=alt.Y2("delta_start:Q"),
                                    color=alt.condition(
                                        alt.datum.updated_prob > alt.datum.original_prob,
                                        alt.value("green"), alt.value("red"),
                                    ),
                                    tooltip=[
                                        alt.Tooltip("startDate:T", title="Date"),
                                        alt.Tooltip("original_prob:Q", title="Original", format=".3f"),
                                        alt.Tooltip("updated_prob:Q", title="Updated", format=".3f"),
                                        alt.Tooltip("delta:Q", title="Delta", format=".3f"),
                                    ],
                                )
                                st.altair_chart(bars + delta_bars, use_container_width=True)
                            else:
                                st.altair_chart(bars, use_container_width=True)

                        # Schedule table
                        st.subheader("Schedule Probabilities")
                        if has_changes:
                            disp = results[["startDate", "homeTeam", "awayTeam",
                                            "original_prob", "updated_prob", "delta"]].copy()
                            disp["updated_prob"] = disp["updated_prob"].round(3)
                            disp["delta"] = disp["delta"].round(3)
                        else:
                            disp = results[["startDate", "homeTeam", "awayTeam",
                                            "original_prob"]].copy()
                        disp["startDate"] = pd.to_datetime(
                            disp["startDate"], errors="coerce"
                        ).dt.strftime("%Y-%m-%d")
                        disp["original_prob"] = disp["original_prob"].round(3)
                        st.dataframe(disp, use_container_width=True)

                        # Position changes table
                        if has_changes:
                            st.subheader("Position Changes")
                            st.dataframe(pd.DataFrame(changes), use_container_width=True)


if __name__ == "__main__":
    main()
