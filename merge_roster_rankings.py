from pathlib import Path
import pandas as pd

def convert_recruitIds_to_int(df):
    # convert [] values to nan
    df["recruitIds"] = (
        df["recruitIds"]
        .replace("[]", pd.NA)           # treat [] as missing
        )

    # convert recruitIds to numeric
    df["recruitId"] = pd.to_numeric(
        df["recruitIds"].str.extract(r"(\d+)")[0],
        errors="coerce"
        )

    return df



recruit_csv = 'data/recruit_data.csv'
rosters_path = Path('data/rosters')
out_path = Path('data/merged_rosters')
out_path.mkdir(parents=True, exist_ok=True)

recruit_df = pd.read_csv(recruit_csv)

roster_files = sorted(rosters_path.glob("*.csv"))

for roster_file in roster_files:
    roster_df = pd.read_csv(roster_file)
    roster_df = convert_recruitIds_to_int(roster_df)

    # left join on roster_file recruitId and recruit_df.id
    merged = roster_df.merge(
        recruit_df[["id", "stars", "rating"]],
          how="left",
          left_on="recruitId",
          right_on="id"
          )
    
    # fix duplicate id columns
    id = merged.pop('id_x')
    merged.insert(0, 'id', value=id)
    merged = merged.drop('id_y', axis=1)
    
    
    out_file = out_path / roster_file.name  # keep same filename

    merged.to_csv(out_file, index=False)
    
