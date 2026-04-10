import streamlit as st
import pandas as pd
import altair as alt
import folium
from streamlit_folium import st_folium
import ast
import warnings
from io import StringIO

st.set_page_config(page_title='CFB Recruiting', layout='wide')
alt.data_transformers.disable_max_rows()
warnings.filterwarnings('ignore')

st.title("CFB Recruit Ratings")

positions_sort = [
    'All Positions', 'Defensive Back', 'Defensive Line', 'Linebacker',
    'Offensive Line', 'Quarterback', 'Receiver', 'Running Back', 'Special Teams',
]

NIL_YEAR = 2021
POWER4 = {'SEC', 'Big Ten', 'Big 12', 'ACC'}



@st.cache_data
def load_data():
    dashboard_df = pd.read_csv('data/dashboard_df.csv')
    teams_df = pd.read_csv('data/teams_df.csv')
    recruits_df = pd.read_csv('data/recruit_df.csv')
    pre_post_nil_df = pd.read_csv('data/pre_post_nil_df.csv')
    return dashboard_df, teams_df, recruits_df, pre_post_nil_df

dashboard_df, teams_df, recruits_df, pre_post_nil_df = load_data()

tab1, tab2 = st.tabs(['Recruiting Dashboard', 'NIL Chart'])

with tab1:
    METRICS = ['averageRating', 'totalRating', 'commits', 'averageStars']

    METRIC_LABELS = {
        'averageRating': 'Average Rating',
        'totalRating':   'Total Rating',
        'commits':       'Commits',
        'averageStars':  'Average Stars',
    }

    STAR_COLORS = {5: '#FFD700', 4: '#C0C0C0', 3: '#CD7F32', 2: '#888888', 1: '#888888'}

    # ── Session state ─────────────────────────────────────────────────────────────
    if 'sel_team' not in st.session_state:
        st.session_state.sel_team = 'Michigan'
    if 'sel_conf' not in st.session_state:
        mich_conf = dashboard_df.loc[dashboard_df['team'] == 'Michigan', 'conference']
        st.session_state.sel_conf = mich_conf.iloc[0] if not mich_conf.empty else 'All Conferences'
    if 'sel_year' not in st.session_state:
        st.session_state.sel_year = 2025
    if 'sel_metric' not in st.session_state:
        st.session_state.sel_metric = 'averageRating'

    # ── Sidebar filters ───────────────────────────────────────────────────────────
    st.sidebar.header('Filters')

    conferences = ['All Conferences'] + sorted(dashboard_df['conference'].dropna().unique())
    conf_index = conferences.index(st.session_state.sel_conf) if st.session_state.sel_conf in conferences else 0
    sel_conf = st.sidebar.selectbox('Conference', conferences, index=conf_index)
    if sel_conf != st.session_state.sel_conf:
        st.session_state.sel_team = None
        st.session_state.last_click_coords = None
    st.session_state.sel_conf = sel_conf
    st.session_state.sel_conf = sel_conf

    years = sorted(dashboard_df['year'].dropna().unique())
    sel_year_value = st.session_state.sel_year if st.session_state.sel_year in years else years[-1]
    sel_year = st.sidebar.select_slider('Year', options=years, value=sel_year_value)
    st.session_state.sel_year = sel_year

    metric_index = METRICS.index(st.session_state.sel_metric) if st.session_state.sel_metric in METRICS else 0
    sel_metric = st.sidebar.selectbox('Metric', METRICS, format_func=lambda m: METRIC_LABELS[m],
                                       index=metric_index)
    st.session_state.sel_metric = sel_metric

    # ── Derived column names ──────────────────────────────────────────────────────
    school_col = f'school{sel_metric[0].upper()}{sel_metric[1:]}'
    conf_col   = f'conf{sel_metric[0].upper()}{sel_metric[1:]}'
    diff_col   = f'{sel_metric}Diff'

    # ── Filter dashboard ──────────────────────────────────────────────────────────
    df = dashboard_df[dashboard_df['year'] == sel_year].copy()
    if sel_conf != 'All Conferences':
        df = df[df['conference'] == sel_conf]

    # ── Map data: one row per team ────────────────────────────────────────────────
    map_df = (
        df.groupby(['team', 'conference', 'lat', 'lon', 'logo'])
        .size().reset_index(name='n')
        .dropna(subset=['lat', 'lon'])
    )

    # ── Recruits for selected year ────────────────────────────────────────────────
    year_recruits = recruits_df[recruits_df['year'] == sel_year].copy()

    # ── Build folium map ──────────────────────────────────────────────────────────
    @st.cache_data
    def build_map(map_df_json, recruits_json, sel_team):

        map_df  = pd.read_json(StringIO(map_df_json))
        rec_df  = pd.read_json(StringIO(recruits_json))

        m = folium.Map(location=[38.5, -96.5], zoom_start=4, tiles='CartoDB positron')

        # Draw recruit lines only for the selected team
        if sel_team and not rec_df.empty:
            team_recruits = rec_df[rec_df['committedTo'] == sel_team]
            school_row    = map_df[map_df['team'] == sel_team]

            if not school_row.empty and not team_recruits.empty:
                s_lat = school_row.iloc[0]['lat']
                s_lon = school_row.iloc[0]['lon']

                for _, r in team_recruits.iterrows():
                    stars    = int(r.get('stars', 0))
                    color    = STAR_COLORS.get(stars, '#888888')
                    star_str = '⭐' * stars if stars > 0 else 'N/A'

                    ht_in  = r.get('height', None)
                    ht_str = (f"{int(ht_in)//12}'{int(ht_in)%12}\""
                              if pd.notna(ht_in) else 'N/A')
                    wt     = r.get('weight', None)
                    wt_str = f"{int(wt)} lbs" if pd.notna(wt) else 'N/A'
                    rank   = r.get('ranking', None)
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

                    # Line from hometown → school
                    folium.PolyLine(
                        locations=[[r['home_lat'], r['home_lon']], [s_lat, s_lon]],
                        color=color,
                        weight=2,
                        opacity=0.7,
                        tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    ).add_to(m)

                    # Dot at hometown
                    folium.CircleMarker(
                        location=[r['home_lat'], r['home_lon']],
                        radius=4,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.9,
                        tooltip=folium.Tooltip(tooltip_html, sticky=True),
                    ).add_to(m)

        # School logo markers (drawn on top of lines)
        for _, row in map_df.iterrows():
            is_selected   = row['team'] == sel_team
            has_selection = sel_team is not None
            opacity = 1.0 if (is_selected or not has_selection) else 0.25
            size    = (44, 44) if is_selected else (32, 32)

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
        map_df.to_json(),
        year_recruits[recruit_cols].to_json(),
        st.session_state.sel_team,
    )

    click_data = st_folium(m, use_container_width=True, height=520, key='team_map')

    # ── Update selected team from map click ──────────────────────────────────────
    # if click_data and click_data.get('last_object_clicked_tooltip'):
    #     clicked = click_data['last_object_clicked_tooltip']
    #     # Only update if the tooltip is a plain team name (not a recruit HTML tooltip)
    #     if clicked in map_df['team'].values and clicked != st.session_state.sel_team:
    #         st.session_state.sel_team = clicked
    #         st.rerun()
    if click_data and click_data.get('last_object_clicked_tooltip'):
        clicked = click_data['last_object_clicked_tooltip']
        clicked_coords = click_data.get('last_object_clicked')

        if clicked in map_df['team'].values:
            if (clicked == st.session_state.sel_team and
                    clicked_coords == st.session_state.get('last_click_coords')):
                st.session_state.sel_team = None
                st.session_state.last_click_coords = None
            else:
                st.session_state.sel_team = clicked
                st.session_state.last_click_coords = clicked_coords
            st.rerun()


    sel_team = st.session_state.sel_team

    # ── Star-color legend ─────────────────────────────────────────────────────────
    if sel_team:
        legend_parts = ' &nbsp;&nbsp; '.join(
            f"<span style='color:{c}; font-size:20px;'>●</span> {s}★"
            for s, c in sorted(STAR_COLORS.items(), reverse=True)
        )
        st.markdown(f"**Line color by recruit star rating:** &nbsp; {legend_parts}",
                    unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────────
    metric_label = METRIC_LABELS[sel_metric]
    st.subheader(f'{metric_label} Diff by Position Group')

    if sel_team is None:
        st.info('Click a team logo on the map to see its position-group breakdown.')
    else:
        st.write(f'**{sel_team}** · {sel_year} · {metric_label}')
        chart_df = df[
            (df['team'] == sel_team)
        ].copy()
        print(chart_df.drop_duplicates())

        if chart_df.empty:
            st.warning(f'No position-group data for {sel_team} in {sel_year}.')
        else:
            chart_df['color'] = chart_df[diff_col].apply(lambda x: 'Above' if x >= 0 else 'Below')
            chart_df['positionGroup'] = pd.Categorical(
                chart_df['positionGroup'], categories=positions_sort, ordered=True)
            chart_df = chart_df.sort_values('positionGroup')

            #max_abs = chart_df[diff_col].abs().max() * 1.05

            bars = alt.Chart(chart_df).mark_bar().encode(
                y=alt.Y('positionGroup:N', title='Position Group', sort=positions_sort),
                x=alt.X(
                    f'{diff_col}:Q',
                    title=f'Diff vs Conference Average ({metric_label})',
                    #scale=alt.Scale(domain=[-max_abs, max_abs]),
                ),
                color=alt.Color(
                    'color:N',
                    scale=alt.Scale(domain=['Above', 'Below'], range=['steelblue', 'tomato']),
                    legend=alt.Legend(title='vs Conf Avg'),
                ),
                tooltip=['positionGroup:N', alt.Tooltip(f'{diff_col}:Q', format='.4f')],
            )

            # zero_df = pd.DataFrame({'x': [0]})
            # zero = alt.Chart(zero_df).mark_rule(color='black', strokeDash=[4, 4]).encode(
            #     x=alt.X('x:Q', scale=alt.Scale(domain=[-max_abs, max_abs])),
            # )

            #st.altair_chart(alt.layer(bars, zero).properties(height=400), use_container_width=True)
            st.altair_chart(alt.layer(bars).properties(height=400), use_container_width=True)

    st.subheader(f'{metric_label} Over Time — {sel_team} vs Conference Average')

    if sel_team:
        time_df = dashboard_df[
            (dashboard_df['team'] == sel_team) &
            (dashboard_df['positionGroup'] == 'All Positions')
        ].copy()

        if not time_df.empty:
            # Reshape to long format so each year has a School and Conference row
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
            ).encode(
                x=alt.X('year:O'),
            )

            nil_label = alt.Chart(pd.DataFrame({'year': [NIL_YEAR], 'label': ['NIL Era']})).mark_text(
                align='left', dx=6, dy=-130,
                color='red', fontSize=12, fontWeight='bold',
            ).encode(
                x=alt.X('year:O'),
                text='label:N',
            )

            st.altair_chart(
                alt.layer(line, nil_rule, nil_label).properties(height=400),
                use_container_width=True,
            )
        else:
            st.warning(f'No time-series data for {sel_team}.')

    # ── Individual recruit table ──────────────────────────────────────────────────
    if sel_team:
        team_rec = year_recruits[year_recruits['committedTo'] == sel_team].copy()
        if not team_rec.empty:
            with st.expander(f"Individual recruits — {sel_team} {sel_year} ({len(team_rec)} commits)"):
                display_cols = ['ranking', 'name', 'position', 'stars', 'rating',
                                'city', 'stateProvince', 'height', 'weight']
                available = [c for c in display_cols if c in team_rec.columns]
                st.dataframe(
                    team_rec[available].sort_values('ranking').reset_index(drop=True),
                    use_container_width=True,
                )

    # ── Raw aggregated data ───────────────────────────────────────────────────────
    with st.expander('Raw filtered data'):
        cols = ['team', 'conference', 'positionGroup', 'year', school_col, conf_col, diff_col]
        st.dataframe(df[cols].reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("NIL Impact on College Football Recruiting")
    st.caption("Post-NIL (2021+) minus Pre-NIL (–2020)")

    METRICS = ['avg_stars', 'num_commits', 'avg_rating', 'gini', 'rank', 'points']
    TOP_N = 10
    CONF_COLORS = {"Power 4": "#FFCB05", "G5 / Ind.": "#00274C"}

    pre_post_nil_df = pd.read_csv('data/pre_post_nil_df.csv')

    sel_metric = st.selectbox('Metric', METRICS, key='nil_metric')

    filtered = pre_post_nil_df[pre_post_nil_df['metric'] == sel_metric].copy()
    winners = filtered[filtered['category'] == 'Winner'].sort_values('delta', ascending=False).head(TOP_N)
    losers = filtered[filtered['category'] == 'Loser'].sort_values('delta', ascending=True).head(TOP_N)

    rule_df = pd.DataFrame({"v": [0]})


    def panel(df, sort_order, title, title_color):
        y = alt.Y('team:N', sort=df['team'].tolist(), axis=alt.Axis(title=None, labelFontSize=12, labelLimit=180))
        x = alt.X('delta:Q', axis=alt.Axis(title='Post-NIL minus Pre-NIL', format="+.3f", gridColor="#eee"))
        bars = alt.Chart(df).mark_bar(height=20, cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=x, y=y,
            color=alt.Color("group:N", scale=alt.Scale(domain=list(CONF_COLORS), range=list(CONF_COLORS.values())),
                            legend=alt.Legend(title="Conference")),
            tooltip=["team:N", "group:N", alt.Tooltip("delta:Q", format="+.4f")],
        )
        labels = alt.Chart(df).mark_text(align="left" if sort_order == "descending" else "right",
                                         dx=5 if sort_order == "descending" else -5,
                                         fontSize=10, color="#333").encode(x=x, y=y, text="delta_label:N")
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