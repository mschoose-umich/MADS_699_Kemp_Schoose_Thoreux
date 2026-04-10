"""FiveThirtyEight-inspired Plotly theme and conference color map."""

# Background and text colors
BG_COLOR = "#f0f0f0"
PLOT_BG_COLOR = "#f0f0f0"
GRID_COLOR = "#d4d4d4"
TEXT_COLOR = "#3c3c3c"
FONT_FAMILY = "Arial, Helvetica, sans-serif"

# Conference color palette
CONFERENCE_COLORS = {
    "Big Ten": "#008fd5",
    "SEC": "#fc4f30",
    "Big 12": "#e5ae38",
    "ACC": "#6d904f",
}

# Sequential palette for general use
PALETTE = ["#008fd5", "#fc4f30", "#e5ae38", "#6d904f", "#8b8b8b"]


def apply_fte_theme(fig):
    """Apply FiveThirtyEight visual theme to a Plotly figure."""
    fig.update_layout(
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=BG_COLOR,
        font=dict(
            family=FONT_FAMILY,
            size=13,
            color=TEXT_COLOR,
        ),
        title_font=dict(
            family=FONT_FAMILY,
            size=20,
            color=TEXT_COLOR,
        ),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            gridwidth=1,
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            gridwidth=1,
            showline=False,
            zeroline=False,
        ),
        margin=dict(l=60, r=30, t=60, b=40),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
    )
    return fig


def conference_color(conf):
    """Return the color for a conference, with a gray fallback."""
    return CONFERENCE_COLORS.get(conf, "#8b8b8b")
