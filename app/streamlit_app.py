import ast
import os
import sys
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import warnings
from io import StringIO

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import folium
from streamlit_folium import st_folium

from team_colors import TEAM_COLORS, DEFAULT_COLOR

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

st.set_page_config(page_title="CFB Power Rankings & Recruiting", layout="wide")
alt.data_transformers.disable_max_rows()
warnings.filterwarnings('ignore')


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

    # Recruiting dashboard data (alex_branch)
    dash_path = os.path.join(DATA_DIR, "dashboard_df.csv")
    recruit_dash_path = os.path.join(DATA_DIR, "recruit_df.csv")
    teams_dash_path = os.path.join(DATA_DIR, "teams_df.csv")
    nil_path = os.path.join(DATA_DIR, "pre_post_nil_df.csv")

    if all(os.path.exists(p) for p in [dash_path, recruit_dash_path, teams_dash_path, nil_path]):
        dashboard_df = pd.read_csv(dash_path)
        recruit_dash_df = pd.read_csv(recruit_dash_path)
        teams_dash_df = pd.read_csv(teams_dash_path)
        pre_post_nil_df = pd.read_csv(nil_path)
    else:
        dashboard_df, recruit_dash_df, teams_dash_df, pre_post_nil_df = None, None, None, None

    return rankings, wp_games, wp_rosters, wp_model, dashboard_df, recruit_dash_df, teams_dash_df, pre_post_nil_df


def color_scale(teams):
    domain = list(teams)
    rng = [TEAM_COLORS.get(t, DEFAULT_COLOR) for t in domain]
    return alt.Scale(domain=domain, range=rng)


def main():
    st.title("CFB Power Rankings & Recruiting")

    (rankings, wp_games, wp_rosters, wp_model,
     dashboard_df, recruit_dash_df, teams_dash_df, pre_post_nil_df) = load_data()

    # ── Shared data for filters ─────────────────────────────────────────────
    all_confs = sorted(rankings["conference"].dropna().unique())
    hist_years = sorted(rankings[rankings["type"] == "Historical"]["year"].unique())

    # ── Tab layout ───────────────────────────────────────────────────────────
    tab_rank, tab_proj, tab_wp, tab_recruit, tab_nil = st.tabs([
        "Power Rankings", "Multi-Year Projections",
        "Win Probability Simulator",
        "Recruiting Dashboard", "NIL Impact",
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

            st.title("College Football Win Probability Simulator")

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
                        st.subheader("Adjust Team Roster")
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
                            st.subheader("Win Probability by Game")
                            orig_wins = int((results["original_prob"] > 0.5).sum())
                            st.markdown(f"**Original projected record:** {orig_wins}-{len(results) - orig_wins}")
                            if has_changes:
                                new_wins = int((results["updated_prob"] > 0.5).sum())
                                st.markdown(f"**Updated projected record:** {new_wins}-{len(results) - new_wins}")
                                st.markdown(f"**Additional wins:** {new_wins - orig_wins:+}")
                            else:
                                st.markdown("**Updated projected record:** —")
                                st.markdown("**Additional wins:** —")

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
                        st.subheader("Position Changes")
                        if has_changes:
                            st.dataframe(pd.DataFrame(changes), use_container_width=True)
                        else:
                            st.write("No position changes have been made.")


    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — Recruiting Dashboard (from alex_branch)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_recruit:
        if dashboard_df is None or recruit_dash_df is None or teams_dash_df is None:
            st.warning("Recruiting dashboard data not found. Run `app/clean_dashboard_data.py` first.")
        else:
            _RD_METRICS = ['averageRating', 'totalRating', 'commits', 'averageStars']

            _RD_METRIC_LABELS = {
                'averageRating': 'Average Rating',
                'totalRating':   'Total Rating',
                'commits':       'Commits',
                'averageStars':  'Average Stars',
            }

            _RD_STAR_COLORS = {5: '#FFD700', 4: '#C0C0C0', 3: '#CD7F32', 2: '#888888', 1: '#888888'}

            _RD_POSITIONS_SORT = [
                'All Positions', 'Defensive Back', 'Defensive Line', 'Linebacker',
                'Offensive Line', 'Quarterback', 'Receiver', 'Running Back', 'Special Teams',
            ]

            NIL_YEAR = 2021

            # ── Session state ─────────────────────────────────────────────────────
            if 'sel_team' not in st.session_state:
                st.session_state.sel_team = 'Michigan'
            if 'sel_conf' not in st.session_state:
                mich_conf = dashboard_df.loc[dashboard_df['team'] == 'Michigan', 'conference']
                st.session_state.sel_conf = mich_conf.iloc[0] if not mich_conf.empty else 'All Conferences'
            if 'sel_year' not in st.session_state:
                st.session_state.sel_year = 2025
            if 'sel_metric' not in st.session_state:
                st.session_state.sel_metric = 'averageRating'

            # ── Sidebar filters ───────────────────────────────────────────────────
            st.sidebar.header('Filters')

            conferences = ['All Conferences'] + sorted(dashboard_df['conference'].dropna().unique())
            conf_index = conferences.index(st.session_state.sel_conf) if st.session_state.sel_conf in conferences else 0
            sel_conf = st.sidebar.selectbox('Conference', conferences, index=conf_index)
            if sel_conf != st.session_state.sel_conf:
                st.session_state.sel_team = None
                st.session_state.last_click_coords = None
            st.session_state.sel_conf = sel_conf

            years = sorted(dashboard_df['year'].dropna().unique())
            sel_year_value = st.session_state.sel_year if st.session_state.sel_year in years else years[-1]
            sel_year = st.sidebar.select_slider('Year', options=years, value=sel_year_value)
            st.session_state.sel_year = sel_year

            metric_index = _RD_METRICS.index(st.session_state.sel_metric) if st.session_state.sel_metric in _RD_METRICS else 0
            sel_metric = st.sidebar.selectbox('Metric', _RD_METRICS,
                                              format_func=lambda m: _RD_METRIC_LABELS[m],
                                              index=metric_index)
            st.session_state.sel_metric = sel_metric

            # ── Derived column names ──────────────────────────────────────────────
            school_col = f'school{sel_metric[0].upper()}{sel_metric[1:]}'
            conf_col   = f'conf{sel_metric[0].upper()}{sel_metric[1:]}'
            diff_col   = f'{sel_metric}Diff'

            # ── Filter dashboard ──────────────────────────────────────────────────
            df = dashboard_df[dashboard_df['year'] == sel_year].copy()
            if sel_conf != 'All Conferences':
                df = df[df['conference'] == sel_conf]

            # ── Map data: one row per team ────────────────────────────────────────
            rd_map_df = (
                df.groupby(['team', 'conference', 'lat', 'lon', 'logo'])
                .size().reset_index(name='n')
                .dropna(subset=['lat', 'lon'])
            )

            # ── Recruits for selected year ────────────────────────────────────────
            year_recruits = recruit_dash_df[recruit_dash_df['year'] == sel_year].copy()

            # ── Build folium map ──────────────────────────────────────────────────
            @st.cache_data
            def build_map(map_df_json, recruits_json, sel_team):
                map_data = pd.read_json(StringIO(map_df_json))
                rec_df = pd.read_json(StringIO(recruits_json))

                m = folium.Map(location=[38.5, -96.5], zoom_start=4, tiles='CartoDB positron')

                if sel_team and not rec_df.empty:
                    team_recruits = rec_df[rec_df['committedTo'] == sel_team]
                    school_row = map_data[map_data['team'] == sel_team]

                    if not school_row.empty and not team_recruits.empty:
                        s_lat = school_row.iloc[0]['lat']
                        s_lon = school_row.iloc[0]['lon']

                        for _, r in team_recruits.iterrows():
                            stars = int(r.get('stars', 0))
                            color = _RD_STAR_COLORS.get(stars, '#888888')
                            star_str = '\u2b50' * stars if stars > 0 else 'N/A'

                            ht_in = r.get('height', None)
                            ht_str = (f"{int(ht_in)//12}'{int(ht_in)%12}\""
                                      if pd.notna(ht_in) else 'N/A')
                            wt = r.get('weight', None)
                            wt_str = f"{int(wt)} lbs" if pd.notna(wt) else 'N/A'
                            rank = r.get('ranking', None)
                            rank_str = f"#{int(rank)}" if pd.notna(rank) else 'N/A'
                            rating = r.get('rating', None)
                            rating_str = f"{rating:.4f}" if pd.notna(rating) else 'N/A'

                            tooltip_html = (
                                f"<b>{r['name']}</b><br>"
                                f"Position: {r.get('position', 'N/A')}<br>"
                                f"Stars: {star_str}<br>"
                                f"Rating: {rating_str}<br>"
                                f"Ranking: {rank_str}<br>"
                                f"Hometown: {r.get('city', '')}, {r.get('stateProvince', '')}<br>"
                                f"Height: {ht_str} &nbsp;|&nbsp; Weight: {wt_str}"
                            )

                            folium.PolyLine(
                                locations=[[r['home_lat'], r['home_lon']], [s_lat, s_lon]],
                                color=color, weight=2, opacity=0.7,
                                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                            ).add_to(m)

                            folium.CircleMarker(
                                location=[r['home_lat'], r['home_lon']],
                                radius=4, color=color, fill=True,
                                fill_color=color, fill_opacity=0.9,
                                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                            ).add_to(m)

                for _, row in map_data.iterrows():
                    is_selected = row['team'] == sel_team
                    has_selection = sel_team is not None
                    opacity = 1.0 if (is_selected or not has_selection) else 0.25
                    size = (44, 44) if is_selected else (32, 32)

                    icon_html = (
                        f"<div style='opacity:{opacity};'>"
                        f"<img src='{row['logo']}' width='{size[0]}' height='{size[1]}' "
                        f"style=\"filter:{'drop-shadow(0 0 6px #003087)' if is_selected else 'none'}\"/>"
                        f"</div>"
                    )
                    folium.Marker(
                        location=[row['lat'], row['lon']],
                        icon=folium.DivIcon(html=icon_html, icon_size=size,
                                            icon_anchor=(size[0]//2, size[1]//2)),
                        tooltip=row['team'],
                    ).add_to(m)

                return m

            st.subheader('Click a team logo to see its recruiting breakdown')

            recruit_cols = ['committedTo', 'name', 'position', 'stars', 'rating',
                            'ranking', 'city', 'stateProvince', 'height', 'weight',
                            'home_lat', 'home_lon']

            m = build_map(
                rd_map_df.to_json(),
                year_recruits[recruit_cols].to_json(),
                st.session_state.sel_team,
            )

            click_data = st_folium(m, use_container_width=True, height=520, key='team_map')

            if click_data and click_data.get('last_object_clicked_tooltip'):
                clicked = click_data['last_object_clicked_tooltip']
                clicked_coords = click_data.get('last_object_clicked')

                if clicked in rd_map_df['team'].values:
                    if (clicked == st.session_state.sel_team and
                            clicked_coords == st.session_state.get('last_click_coords')):
                        st.session_state.sel_team = None
                        st.session_state.last_click_coords = None
                    else:
                        st.session_state.sel_team = clicked
                        st.session_state.last_click_coords = clicked_coords
                    st.rerun()

            sel_team = st.session_state.sel_team

            # ── Star-color legend ─────────────────────────────────────────────────
            if sel_team:
                legend_parts = ' &nbsp;&nbsp; '.join(
                    f"<span style='color:{c}; font-size:20px;'>\u25cf</span> {s}\u2605"
                    for s, c in sorted(_RD_STAR_COLORS.items(), reverse=True)
                )
                st.markdown(f"**Line color by recruit star rating:** &nbsp; {legend_parts}",
                            unsafe_allow_html=True)

            # ── Bar chart ─────────────────────────────────────────────────────────
            metric_label = _RD_METRIC_LABELS[sel_metric]
            st.subheader(f'{metric_label} Diff by Position Group')

            if sel_team is None:
                st.info('Click a team logo on the map to see its position-group breakdown.')
            else:
                st.write(f'**{sel_team}** \u00b7 {sel_year} \u00b7 {metric_label}')
                chart_df = df[df['team'] == sel_team].copy()

                if chart_df.empty:
                    st.warning(f'No position-group data for {sel_team} in {sel_year}.')
                else:
                    chart_df['color'] = chart_df[diff_col].apply(lambda x: 'Above' if x >= 0 else 'Below')
                    chart_df['positionGroup'] = pd.Categorical(
                        chart_df['positionGroup'], categories=_RD_POSITIONS_SORT, ordered=True)
                    chart_df = chart_df.sort_values('positionGroup')

                    bars = alt.Chart(chart_df).mark_bar().encode(
                        y=alt.Y('positionGroup:N', title='Position Group', sort=_RD_POSITIONS_SORT),
                        x=alt.X(
                            f'{diff_col}:Q',
                            title=f'Diff vs Conference Average ({metric_label})',
                        ),
                        color=alt.Color(
                            'color:N',
                            scale=alt.Scale(domain=['Above', 'Below'], range=['steelblue', 'tomato']),
                            legend=alt.Legend(title='vs Conf Avg'),
                        ),
                        tooltip=['positionGroup:N', alt.Tooltip(f'{diff_col}:Q', format='.4f')],
                    )

                    st.altair_chart(alt.layer(bars).properties(height=400), use_container_width=True)

            st.subheader(f'{metric_label} Over Time \u2014 {sel_team} vs Conference Average')

            if sel_team:
                time_df = dashboard_df[
                    (dashboard_df['team'] == sel_team) &
                    (dashboard_df['positionGroup'] == 'All Positions')
                ].copy()

                if not time_df.empty:
                    long_df = pd.melt(
                        time_df,
                        id_vars=['year'],
                        value_vars=[school_col, conf_col],
                        var_name='series',
                        value_name='value',
                    )
                    long_df['series'] = long_df['series'].map({
                        school_col: sel_team,
                        conf_col: 'Conference Avg',
                    })

                    years_all = sorted(dashboard_df['year'].unique())

                    line = alt.Chart(long_df).mark_line(point=True).encode(
                        x=alt.X('year:O', title='Year', axis=alt.Axis(values=years_all)),
                        y=alt.Y('value:Q', title=metric_label, scale=alt.Scale(zero=False)),
                        color=alt.Color(
                            'series:N',
                            title='',
                            scale=alt.Scale(
                                domain=[sel_team, 'Conference Avg'],
                                range=['steelblue', 'gray'],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip('series:N', title=''),
                            alt.Tooltip('year:O', title='Year'),
                            alt.Tooltip('value:Q', title=metric_label, format='.4f'),
                        ],
                    )

                    nil_rule = alt.Chart(pd.DataFrame({'year': [NIL_YEAR]})).mark_rule(
                        color='red', strokeDash=[6, 3], strokeWidth=2,
                    ).encode(x=alt.X('year:O'))

                    nil_label = alt.Chart(pd.DataFrame({'year': [NIL_YEAR], 'label': ['NIL Era']})).mark_text(
                        align='left', dx=6, dy=-130,
                        color='red', fontSize=12, fontWeight='bold',
                    ).encode(x=alt.X('year:O'), text='label:N')

                    st.altair_chart(
                        alt.layer(line, nil_rule, nil_label).properties(height=400),
                        use_container_width=True,
                    )
                else:
                    st.warning(f'No time-series data for {sel_team}.')

            # ── Individual recruit table ──────────────────────────────────────────
            if sel_team:
                team_rec = year_recruits[year_recruits['committedTo'] == sel_team].copy()
                if not team_rec.empty:
                    with st.expander(f"Individual recruits \u2014 {sel_team} {sel_year} ({len(team_rec)} commits)"):
                        display_cols = ['ranking', 'name', 'position', 'stars', 'rating',
                                        'city', 'stateProvince', 'height', 'weight']
                        available = [c for c in display_cols if c in team_rec.columns]
                        st.dataframe(
                            team_rec[available].sort_values('ranking').reset_index(drop=True),
                            use_container_width=True,
                        )

            # ── Raw aggregated data ───────────────────────────────────────────────
            with st.expander('Raw filtered data'):
                cols = ['team', 'conference', 'positionGroup', 'year', school_col, conf_col, diff_col]
                st.dataframe(df[cols].reset_index(drop=True), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7 — NIL Impact (from alex_branch)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_nil:
        if pre_post_nil_df is None:
            st.warning("NIL impact data not found. Run `app/clean_dashboard_data.py` first.")
        else:
            st.subheader("NIL Impact on College Football Recruiting")
            st.caption("Post-NIL (2021+) minus Pre-NIL (\u20132020)")

            _NIL_METRICS = ['avg_stars', 'num_commits', 'avg_rating', 'gini', 'rank', 'points']
            TOP_N = 10
            CONF_COLORS = {"Power 4": "#FFCB05", "G5 / Ind.": "#00274C"}

            sel_nil_metric = st.selectbox('Metric', _NIL_METRICS, key='nil_metric')

            filtered = pre_post_nil_df[pre_post_nil_df['metric'] == sel_nil_metric].copy()
            winners = filtered[filtered['category'] == 'Winner'].sort_values('delta', ascending=False).head(TOP_N)
            losers = filtered[filtered['category'] == 'Loser'].sort_values('delta', ascending=True).head(TOP_N)

            rule_df = pd.DataFrame({"v": [0]})

            def panel(df_panel, sort_order, title, title_color):
                y = alt.Y('team:N', sort=df_panel['team'].tolist(),
                          axis=alt.Axis(title=None, labelFontSize=12, labelLimit=180))
                x = alt.X('delta:Q',
                          axis=alt.Axis(title='Post-NIL minus Pre-NIL', format="+.3f", gridColor="#eee"))
                bars = alt.Chart(df_panel).mark_bar(
                    height=20, cornerRadiusTopRight=4, cornerRadiusBottomRight=4
                ).encode(
                    x=x, y=y,
                    color=alt.Color("group:N",
                                    scale=alt.Scale(domain=list(CONF_COLORS),
                                                    range=list(CONF_COLORS.values())),
                                    legend=alt.Legend(title="Conference")),
                    tooltip=["team:N", "group:N", alt.Tooltip("delta:Q", format="+.4f")],
                )
                labels = alt.Chart(df_panel).mark_text(
                    align="left" if sort_order == "descending" else "right",
                    dx=5 if sort_order == "descending" else -5,
                    fontSize=10, color="#333"
                ).encode(x=x, y=y, text="delta_label:N")
                rule = alt.Chart(rule_df).mark_rule(color="#aaa", strokeDash=[4, 3]).encode(x="v:Q")
                return alt.layer(rule, bars, labels).properties(
                    width=340, height=300,
                    title=alt.TitleParams(title, color=title_color, fontWeight="bold", fontSize=14),
                )

            chart = (
                alt.hconcat(
                    panel(winners, "descending", "Top 10 Gains", "#2a7a2a"),
                    panel(losers, "ascending", "Top 10 Losses", "#b22222"),
                )
                .configure_view(stroke=None)
                .configure_axis(domainColor="#ccc")
            )

            st.altair_chart(chart, use_container_width=False)


if __name__ == "__main__":
    main()
