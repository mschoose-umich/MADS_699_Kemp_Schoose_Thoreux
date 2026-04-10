import streamlit as st
import pandas as pd
import numpy as np
import pickle
import altair as alt

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    games = pd.read_csv("data/2025_games_clean.csv")
    rosters = pd.read_csv("data/2025_rosters.csv")
    games["startDate"] = pd.to_datetime(games["startDate"], errors="coerce")
    return games, rosters


@st.cache_resource
def load_model():
    with open("data/production_model.pkl", "rb") as f:
        return pickle.load(f)


games_df, rosters_df = load_data()
model = load_model()

# =========================
# HELPERS
# =========================
def canonical_team_name(x):
    return str(x).strip()


def get_team_roster(team_name, rosters):
    exact = rosters.loc[rosters["team"] == team_name]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    canon = canonical_team_name(team_name)
    roster_names = rosters["team"].map(canonical_team_name)
    fallback = rosters.loc[roster_names == canon]

    if not fallback.empty:
        return fallback.iloc[0].to_dict()

    return None


def compute_team_aggregates(team_row):
    team_row = team_row.copy()

    team_row["OL_avg"] = np.mean([
        team_row["OL1"], team_row["OL2"], team_row["OL3"],
        team_row["OL4"], team_row["OL5"]
    ])

    team_row["WR_avg"] = np.mean([team_row["WR1"], team_row["WR2"]])

    team_row["DL_avg"] = np.mean([
        team_row["DL1"], team_row["DL2"],
        team_row["DL3"], team_row["DL4"]
    ])

    team_row["LB_avg"] = np.mean([
        team_row["LB1"], team_row["LB2"], team_row["LB3"]
    ])

    team_row["DB_avg"] = np.mean([
        team_row["DB1"], team_row["DB2"],
        team_row["DB3"], team_row["DB4"]
    ])

    return team_row


def build_matchup_features(team_1, team_2, team_1_is_home):
    team_1_synergy = (
        2.0 * team_1["QB1"] +
        1.0 * team_1["WR_avg"] +
        0.8 * team_1["TE1"]
    ) / 3.8

    team_2_synergy = (
        2.0 * team_2["QB1"] +
        1.0 * team_2["WR_avg"] +
        0.8 * team_2["TE1"]
    ) / 3.8

    return {
        "QB_diff": team_1["QB1"] - team_2["QB1"],
        "OL_adv_1": team_1["OL_avg"] - team_2["DL_avg"],
        "OL_adv_2": team_2["OL_avg"] - team_1["DL_avg"],
        "WR_adv_1": team_1["WR_avg"] - team_2["DB_avg"],
        "WR_adv_2": team_2["WR_avg"] - team_1["DB_avg"],
        "RB_adv_1": team_1["RB1"] - team_2["LB_avg"],
        "RB_adv_2": team_2["RB1"] - team_1["LB_avg"],
        "TE_diff": team_1["TE1"] - team_2["TE1"],
        "PK_diff": team_1["PK1"] - team_2["PK1"],
        "P_diff": team_1["P1"] - team_2["P1"],
        "QB1_vs_DB_2": team_1["QB1"] - team_2["DB_avg"],
        "QB2_vs_DB_1": team_2["QB1"] - team_1["DB_avg"],
        "PASS_SYNERGY_diff": team_1_synergy - team_2_synergy,
        "team_1_is_home": int(team_1_is_home)
    }


FEATURES = [
    "QB_diff",
    "OL_adv_1", "OL_adv_2",
    "WR_adv_1", "WR_adv_2",
    "RB_adv_1", "RB_adv_2",
    "TE_diff",
    "PK_diff", "P_diff",
    "QB1_vs_DB_2", "QB2_vs_DB_1",
    "PASS_SYNERGY_diff",
    "team_1_is_home"
]

POSITIONS = [
    "QB1", "RB1", "WR1", "WR2", "TE1",
    "OL1", "OL2", "OL3", "OL4", "OL5",
    "DL1", "DL2", "DL3", "DL4",
    "LB1", "LB2", "LB3",
    "DB1", "DB2", "DB3", "DB4",
    "PK1", "P1"
]

RATING_OPTIONS = [0.0, 2.0, 3.0, 4.0, 5.0]

# =========================
# UI
# =========================
st.title("🏈 College Football Win Probability Simulator")

teams = sorted(
    set(games_df["homeTeam"]).union(set(games_df["awayTeam"]))
)

selected_team = st.selectbox("Select a Team", teams)

team_roster = get_team_roster(selected_team, rosters_df)

if team_roster is None:
    st.error(f"No roster found for selected team: {selected_team}")
    st.stop()

# =========================
# BUILD SCHEDULE
# =========================
team_games = games_df[
    (games_df["homeTeam"] == selected_team) |
    (games_df["awayTeam"] == selected_team)
].sort_values("startDate")

if team_games.empty:
    st.warning(f"No games found for {selected_team}.")
    st.stop()

# =========================
# PREDICTIONS
# =========================
def get_predictions(team_profile):
    rows = []
    valid_indices = []
    skipped_opponents = []

    for idx, game in team_games.iterrows():
        is_home = game["homeTeam"] == selected_team
        opponent = game["awayTeam"] if is_home else game["homeTeam"]

        opponent_row = get_team_roster(opponent, rosters_df)
        if opponent_row is None:
            skipped_opponents.append(opponent)
            continue

        opponent_row = compute_team_aggregates(opponent_row)
        features = build_matchup_features(team_profile, opponent_row, is_home)

        rows.append(features)
        valid_indices.append(idx)

    if not rows:
        return pd.Series(dtype=float), [], skipped_opponents

    X = pd.DataFrame(rows)[FEATURES]
    probs = model.predict_proba(X)[:, 1]

    return pd.Series(probs, index=valid_indices), valid_indices, skipped_opponents


# =========================
# LAYOUT: LEFT CONTROLS / RIGHT CHART
# =========================
left_col, right_col = st.columns([1.0, 1.8])

with left_col:
    st.subheader("Adjust Team Roster")

    edited_roster = {}
    selector_cols = st.columns(2)

    for i, pos in enumerate(POSITIONS):
        current_value = float(team_roster[pos])
        default_index = RATING_OPTIONS.index(current_value) if current_value in RATING_OPTIONS else 0

        with selector_cols[i % 2]:
            edited_roster[pos] = st.selectbox(
                pos,
                options=RATING_OPTIONS,
                index=default_index,
                key=f"{selected_team}_{pos}"
            )

original_team = compute_team_aggregates(team_roster.copy())
edited_team = compute_team_aggregates({**team_roster, **edited_roster})

# determine whether any positions changed
changes = []
for pos in POSITIONS:
    original_val = float(team_roster[pos])
    updated_val = float(edited_roster[pos])
    if original_val != updated_val:
        changes.append({
            "Position": pos,
            "Original": original_val,
            "Updated": updated_val,
            "Delta": updated_val - original_val
        })

has_changes = len(changes) > 0

# baseline always shown
original_probs, original_idx, skipped_1 = get_predictions(original_team)
results = team_games.loc[original_idx].copy()

if results.empty:
    st.error("No valid predictions could be generated for this team.")
    st.stop()

results["original_prob"] = original_probs.loc[original_idx].values

# updated only if changed
if has_changes:
    updated_probs, updated_idx, skipped_2 = get_predictions(edited_team)
    valid_idx = sorted(set(original_idx).intersection(updated_idx))
    results = team_games.loc[valid_idx].copy()
    results["original_prob"] = original_probs.loc[valid_idx].values
    results["updated_prob"] = updated_probs.loc[valid_idx].values
    results["delta"] = results["updated_prob"] - results["original_prob"]
    skipped = sorted(set(skipped_1).union(skipped_2))
else:
    results["updated_prob"] = np.nan
    results["delta"] = np.nan
    skipped = sorted(set(skipped_1))

if skipped:
    st.warning("Skipped games with missing opponent rosters: " + ", ".join(skipped))

def get_opponent_label(row):
    if row["homeTeam"] == selected_team:
        return f"vs {row['awayTeam']}"
    return f"@ {row['homeTeam']}"

results["opponent"] = results.apply(get_opponent_label, axis=1)

# =========================
# RIGHT PANEL: SUMMARY + CHART
# =========================
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

    base = alt.Chart(chart_df).encode(
        x=alt.X(
            "opponent:N",
            sort=None,
            title="Game (Opponent)"
        )
    )

    bars = base.mark_bar().encode(
        y=alt.Y(
            "original_prob:Q",
            title="Win Probability",
            scale=alt.Scale(domain=[0, 1])
        ),
        color=alt.value("lightgray"),
        tooltip=[
            alt.Tooltip("startDate:T", title="Date"),
            alt.Tooltip("homeTeam:N", title="Home"),
            alt.Tooltip("awayTeam:N", title="Away"),
            alt.Tooltip("original_prob:Q", title="Original Prob", format=".3f"),
        ]
    )

    if has_changes:
        chart_df["delta_start"] = chart_df[["original_prob", "updated_prob"]].min(axis=1)
        chart_df["delta_end"] = chart_df[["original_prob", "updated_prob"]].max(axis=1)

        delta = base.mark_bar().encode(
            y=alt.Y("delta_end:Q", scale=alt.Scale(domain=[0, 1])),
            y2=alt.Y2("delta_start:Q"),
            color=alt.condition(
                alt.datum.updated_prob > alt.datum.original_prob,
                alt.value("green"),
                alt.value("red")
            ),
            tooltip=[
                alt.Tooltip("startDate:T", title="Date"),
                alt.Tooltip("homeTeam:N", title="Home"),
                alt.Tooltip("awayTeam:N", title="Away"),
                alt.Tooltip("original_prob:Q", title="Original Prob", format=".3f"),
                alt.Tooltip("updated_prob:Q", title="Updated Prob", format=".3f"),
                alt.Tooltip("delta:Q", title="Delta", format=".3f"),
            ]
        )

        st.altair_chart(
            (alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("opponent:N", sort=None, title="Game (Opponent)"),
                y=alt.Y(
                    "original_prob:Q",
                    title="Win Probability",
                    scale=alt.Scale(domain=[0, 1])
                ),
                color=alt.value("lightgray"),
                tooltip=[
                    alt.Tooltip("startDate:T", title="Date"),
                    alt.Tooltip("homeTeam:N", title="Home"),
                    alt.Tooltip("awayTeam:N", title="Away"),
                    alt.Tooltip("original_prob:Q", title="Original Prob", format=".3f"),
                ]
            ) + alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("opponent:N", sort=None, title="Game (Opponent)"),
                y=alt.Y("delta_end:Q", scale=alt.Scale(domain=[0, 1])),
                y2=alt.Y2("delta_start:Q"),
                color=alt.condition(
                    alt.datum.updated_prob > alt.datum.original_prob,
                    alt.value("green"),
                    alt.value("red")
                ),
                tooltip=[
                    alt.Tooltip("startDate:T", title="Date"),
                    alt.Tooltip("homeTeam:N", title="Home"),
                    alt.Tooltip("awayTeam:N", title="Away"),
                    alt.Tooltip("original_prob:Q", title="Original Prob", format=".3f"),
                    alt.Tooltip("updated_prob:Q", title="Updated Prob", format=".3f"),
                    alt.Tooltip("delta:Q", title="Delta", format=".3f"),
                ]
            )),
            use_container_width=True
        )
    else:
        st.altair_chart(bars, use_container_width=True)

# =========================
# TABLE UNDER BOTH
# =========================
st.subheader("Schedule Probabilities")

if has_changes:
    display_df = results[[
        "startDate", "homeTeam", "awayTeam",
        "original_prob", "updated_prob", "delta"
    ]].copy()
    display_df["updated_prob"] = display_df["updated_prob"].round(3)
    display_df["delta"] = display_df["delta"].round(3)
else:
    display_df = results[[
        "startDate", "homeTeam", "awayTeam",
        "original_prob"
    ]].copy()

display_df["startDate"] = pd.to_datetime(display_df["startDate"], errors="coerce").dt.strftime("%Y-%m-%d")
display_df["original_prob"] = display_df["original_prob"].round(3)

st.dataframe(display_df, use_container_width=True)

# =========================
# POSITION CHANGES TABLE
# =========================
st.subheader("Position Changes")

if has_changes:
    changes_df = pd.DataFrame(changes)
    st.dataframe(changes_df, use_container_width=True)
else:
    st.write("No position changes have been made.")
