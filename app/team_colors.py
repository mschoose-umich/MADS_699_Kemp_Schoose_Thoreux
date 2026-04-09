"""Primary colors for college football teams in the dataset."""

TEAM_COLORS = {
    # SEC
    "Alabama": "#9E1B32",
    "Arkansas": "#9D2235",
    "Auburn": "#0C2340",
    "Florida": "#0021A5",
    "Georgia": "#BA0C2F",
    "Kentucky": "#0033A0",
    "LSU": "#461D7C",
    "Mississippi State": "#660000",
    "Missouri": "#F1B82D",
    "Oklahoma": "#841617",
    "Ole Miss": "#CE1126",
    "South Carolina": "#73000A",
    "Tennessee": "#FF8200",
    "Texas": "#BF5700",
    "Texas A&M": "#500000",
    "Vanderbilt": "#CFAE70",
    # Big Ten
    "Illinois": "#E84A27",
    "Indiana": "#990000",
    "Iowa": "#FFCD00",
    "Maryland": "#E03A3E",
    "Michigan": "#00274C",
    "Michigan State": "#18453B",
    "Minnesota": "#7A0019",
    "Nebraska": "#D00000",
    "Northwestern": "#4E2A84",
    "Ohio State": "#BB0000",
    "Oregon": "#154733",
    "Penn State": "#041E42",
    "Purdue": "#CEB888",
    "Rutgers": "#CC0033",
    "UCLA": "#2D68C4",
    "USC": "#990000",
    "Washington": "#4B2E83",
    "Wisconsin": "#C5050C",
    # Big 12
    "Arizona": "#CC0033",
    "Arizona State": "#8C1D40",
    "Baylor": "#154734",
    "BYU": "#002E5D",
    "Cincinnati": "#E00122",
    "Colorado": "#CFB87C",
    "Houston": "#C8102E",
    "Iowa State": "#C8102E",
    "Kansas": "#0051BA",
    "Kansas State": "#512888",
    "Oklahoma State": "#FF6600",
    "TCU": "#4D1979",
    "Texas Tech": "#CC0000",
    "UCF": "#BA9B37",
    "Utah": "#CC0000",
    "West Virginia": "#002855",
    # ACC
    "Boston College": "#98002E",
    "California": "#003262",
    "Clemson": "#F56600",
    "Duke": "#003087",
    "Florida State": "#782F40",
    "Georgia Tech": "#B3A369",
    "Louisville": "#AD0000",
    "Miami": "#F47321",
    "NC State": "#CC0000",
    "North Carolina": "#7BAFD4",
    "Pittsburgh": "#003594",
    "SMU": "#CC0035",
    "Stanford": "#8C1515",
    "Syracuse": "#F76900",
    "Virginia": "#232D4B",
    "Virginia Tech": "#630031",
    "Wake Forest": "#9E7E38",
}

# Fallback for unknown teams
DEFAULT_COLOR = "#8b8b8b"


def team_color(team):
    """Return the primary color for a team."""
    return TEAM_COLORS.get(team, DEFAULT_COLOR)