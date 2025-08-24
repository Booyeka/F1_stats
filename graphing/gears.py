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
db_session = Session()


with Session() as session:
    telem = session.query(CarTelemetry).filter(
        CarTelemetry.driver_abr == "VER",
        CarTelemetry.race_id == 1,
        CarTelemetry.session == "race",
        CarTelemetry.sessionTime >= 5754.77,
        CarTelemetry.sessionTime <= 5754.77 + 91.591
    ).all()
    pos = session.query(CarPosition).filter(
        CarPosition.driver_abr == "VER",
        CarPosition.race_id == 1,
        CarPosition.session == "race",
        CarPosition.sessionTime >= 5754.77,
        CarPosition.sessionTime <= 5846.05
    )

# Step 1: Convert each ORM object to a dict
telem_dicts = [obj.__dict__ for obj in telem]
pos_dicts = [obj.__dict__ for obj in pos]

#  Step 2: Remove SQLAlchemy's internal state key
for d in telem_dicts:
    d.pop('_sa_instance_state', None)
for d in pos_dicts:
    d.pop('_sa_instance_state', None)

# Step 3: Convert to DataFrame
telem_df = pd.DataFrame(telem_dicts)
pos_df = pd.DataFrame(pos_dicts)


# merge together based on time
# Ensure both DataFrames are sorted by time
telem_df = telem_df.sort_values("time")
pos_df = pos_df.sort_values("time")

# Merge gear onto position (or vice versa)
df_merged = pd.merge_asof(
    pos_df,
    telem_df[["time", "gear"]],
    on="time",
    direction="nearest"  # can be 'backward', 'forward', or 'nearest'
)

print(df_merged.iloc[0])
# 

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# import numpy as np

# Normalize gear for colormap (e.g. gears from 1 to 8)
norm = mcolors.Normalize(vmin=df_merged["gear"].min(), vmax=df_merged["gear"].max())
# cmap = cm.get_cmap("rainbow")  # or "plasma" "viridis", "cool", "rainbow", etc.
cmap = cm._colormaps['rainbow']

# Build segments for plotting
x_pos = df_merged["x_pos"].values
y_pos = df_merged["y_pos"].values
gear = df_merged["gear"].values

# Create a colored line using segments
points = np.array([x_pos, y_pos]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

from matplotlib.collections import LineCollection

lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(gear)
lc.set_linewidth(3)

fig, ax = plt.subplots(figsize=(8, 6))
line = ax.add_collection(lc)
ax.set_xlim(x_pos.min(), x_pos.max())
ax.set_ylim(y_pos.min(), y_pos.max())
fig.colorbar(line, ax=ax, label="Gear")

ax.set_title("Gear Changes Around the Lap")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
plt.axis("equal")
plt.show()




# 5754.77. start

# 5846.05. end