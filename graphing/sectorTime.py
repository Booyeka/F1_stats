import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from dotenv import load_dotenv
from db.db_connection import get_db_conn_alc
import pandas as pd
import numpy as np
from db.models import Race, Lap, Driver, CarTelemetry, CarPosition, Weather, TrackCorners



load_dotenv()  # Load from .env
engine = get_db_conn_alc() # connect to mysql f1_data database
Session = sessionmaker(bind=engine)

with Session() as session:
    laps = session.query(Lap).filter(
        Lap.race_id == 1,
        Lap.session == "race",
        Lap.driver_abr == "VER",
    )

lap_dicts = [obj.__dict__ for obj in laps]
for d in lap_dicts:
    d.pop('_sa_instance_state', None)

lap_df = pd.DataFrame(lap_dicts)

# sort by lap number

lap_df.sort_values(by="lap_number")

sector_data = []
tire_compounds = []
for lap_num in range(len(lap_df["lap_number"])):
    lap_number = int(lap_df.iloc[lap_num]["lap_number"])
    sector1 = float(lap_df.iloc[lap_num]["sector1Time"]) if lap_df.iloc[lap_num]["sector1Time"] is not None else 0.0
    sector2 = float(lap_df.iloc[lap_num]["sector2Time"]) if lap_df.iloc[lap_num]["sector2Time"] is not None else 0.0
    sector3 = float(lap_df.iloc[lap_num]["sector3Time"]) if lap_df.iloc[lap_num]["sector3Time"] is not None else 0.0
    tire_compound = str(lap_df.iloc[lap_num]["compound"])
    sector_data.append((lap_number, sector1, sector2, sector3))
    tire_compounds.append(tire_compound)

# print(tire_compounds)







import matplotlib.pyplot as plt

# Example data (replace with your actual sector times)
# Each tuple: (lap_number, sector1_time, sector2_time, sector3_time)
sector_times = [
    (1, 12.5, 34.2, 28.0),
    (2, 12.4, 34.5, 27.9),
    (3, 12.3, 34.1, 27.8),
    (4, 12.2, 33.9, 27.7),
    # ... up to 57 laps
]

# Color mapping
compound_colors = {
    'SOFT': '#ff4d4d',
    'MEDIUM': '#ffd633',
    'HARD': '#cccccc',
    "INTERMEDIATE": "#4dd67b",
}

# Extract individual sector times
sector1 = [s1 for _, s1, _, _ in sector_data]
sector2 = [s2 for _, _, s2, _ in sector_data]
sector3 = [s3 for _, _, _, s3 in sector_data]

# Create artificial X positions
lap_nums = [lap for lap, *_ in sector_data]
x1 = lap_nums
x2 = [lap + len(lap_nums) for lap in lap_nums]  # shift to right
x3 = [lap + 2 * len(lap_nums) for lap in lap_nums]  # shift further right

# Plotting
fig, ax = plt.subplots(figsize=(14, 6))


# --- Draw tire compound background spans ---
for i, compound in enumerate(tire_compounds):
    color = compound_colors.get(compound, 'white')
    ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.2)  # Sector 1
    ax.axvspan(i + len(lap_nums) - 0.5, i + len(lap_nums) + 0.5, color=color, alpha=0.2)  # Sector 2
    ax.axvspan(i + 2 * len(lap_nums) - 0.5, i + 2 * len(lap_nums) + 0.5, color=color, alpha=0.2)  # Sector 3


ax.scatter(x1, sector1, color='red', label='Sector 1')
ax.scatter(x2, sector2, color='blue', label='Sector 2')
ax.scatter(x3, sector3, color='green', label='Sector 3')

# Add vertical separators between sectors
sep1 = len(lap_nums)
sep2 = 2 * len(lap_nums)
ax.axvline(sep1 - 0.5, color='gray', linestyle='--')
ax.axvline(sep2 - 0.5, color='gray', linestyle='--')

# Labeling
ax.set_title("Driver Sector Times Across Laps")
ax.set_ylabel("Time (s)")
ax.set_xticks([
    len(lap_nums)//2,                      # Midpoint of Sector 1 section
    len(lap_nums) + len(lap_nums)//2,      # Midpoint of Sector 2 section
    2 * len(lap_nums) + len(lap_nums)//2   # Midpoint of Sector 3 section
])
ax.set_xticklabels(["Sector 1", "Sector 2", "Sector 3"])

ax.legend()
from matplotlib.patches import Patch
handles = [
    Patch(color=color, label=compound) for compound, color in compound_colors.items()
]
ax.legend(handles=handles + ax.get_legend().legend_handles)
ax.grid(True)
plt.tight_layout()
plt.show()