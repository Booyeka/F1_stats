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
from matplotlib.patches import Patch



load_dotenv()  # Load from .env
engine = get_db_conn_alc() # connect to mysql f1_data database
Session = sessionmaker(bind=engine)


def get_brake_segments(driver_abr, race_id):
    with Session() as session:
        telem = session.query(CarTelemetry).filter(
            CarTelemetry.race_id == race_id,
            CarTelemetry.session == "race",
            CarTelemetry.driver_abr == driver_abr,
            CarTelemetry.brake == True
        ).all()
        laps = session.query(Lap).filter(
            Lap.race_id == 2,
            Lap.session == "race",
            Lap.driver_abr == driver_abr,
        )

    telem_dicts = [obj.__dict__ for obj in telem]
    lap_dicts = [obj.__dict__ for obj in laps]

    for d in telem_dicts:
        d.pop('_sa_instance_state', None)
    for d in lap_dicts:
        d.pop('_sa_instance_state', None)

    telem_df = pd.DataFrame(telem_dicts)
    lap_df = pd.DataFrame(lap_dicts)
    # print(lap_df.iloc[0])

    lap_df = lap_df.sort_values(by="lap_number") # sort lap_df by lap numer
    # print(lap_df)
    tire_compounds = [compound for compound in lap_df["compound"]]

    brake_segments = []
    lap_distances = {}

    for lap in range(len(lap_df["lap_number"])):
        lap_start = lap_df.iloc[lap]["lapStartTime"]
        lap_end = lap_start + lap_df.iloc[lap]["lap_time"]
        lap_data = session.query(CarTelemetry).filter(
            CarTelemetry.driver_abr == 'VER',
            CarTelemetry.race_id == 2,
            CarTelemetry.session == 'race',
            CarTelemetry.sessionTime >= lap_start,
            CarTelemetry.sessionTime <= lap_end
        ).order_by(CarTelemetry.time).all()

        # for d in lap_data:
        #     d.pop('_sa_instance_state', None)
        distances = []
        for data in lap_data:
            # print(data)
            distances.append(data.distance)
        # print(distances)
        if len(distances) == 0:
            continue
        start_dist = distances[0]
        end_dist = distances[-1]
        lap_distances[lap+1] = (start_dist, end_dist)
        
        braking = False
        start_distance = None
        for point in lap_data:
            if point.brake and not braking:
                braking = True
                start_distance = point.distance
            elif not point.brake and braking:
                braking = False
                end_distance = point.distance
                brake_segments.append((int(lap_df.iloc[lap]["lap_number"]), start_distance, end_distance))
    
    return brake_segments, lap_distances, tire_compounds

# print(brake_segments)
# brake_dict = {}
# for thing in brake_segments:
#     if thing[0] not in brake_dict.keys():
#         brake_dict.update({
#             thing[0] : []
#         })

#     brake_dict[thing[0]].append((thing[1],thing[2]))

# print(brake_dict)

import matplotlib.pyplot as plt
import matplotlib.patches as patches


driver1_data = get_brake_segments(driver_abr="NOR", race_id=1)
driver1_brake_segments = driver1_data[0]
driver1_lap_distances = driver1_data[1]
driver1_tire_compounds = driver1_data[2]



normalized_brake_segments1 = []
normalized_lap_distances1 = []


# standard_lap_length1 = max(end - start for start, end in driver1_lap_distances.values())
standard_lap_length1 = 5278
# standard_lap_length2 = max(end - start for start, end in driver2_lap_distances.values())

for lap, start_dist, end_dist in driver1_brake_segments:
    lap_start, lap_end = driver1_lap_distances[lap]
    lap_length = lap_end - lap_start

    norm_start = (start_dist - lap_start) / lap_length * standard_lap_length1
    norm_end = (end_dist - lap_start) / lap_length * standard_lap_length1

    # lap_distance = lap_length
    # normalized_distance = (lap_distance / lap_length) * 5278
    # normalized_brake_segments1.append((lap, 0, normalized_distance))
    normalized_brake_segments1.append((lap, norm_start, norm_end))



brake_segments_driver1 = normalized_brake_segments1


# Unpack all lap numbers and distances to compute axis limits
lap_numbers1 = [lap for lap, _, _ in brake_segments_driver1]
start_distances1 = [start for _, start, _ in brake_segments_driver1]
end_distances1 = [end for _, _, end in brake_segments_driver1]

min_dist1 = min(start_distances1)
max_dist1 = max(end_distances1)
min_lap1 = min(lap_numbers1)
max_lap1 = max(lap_numbers1)




# get corner info/dist
with Session() as session:
    corners = session.query(TrackCorners).filter(
        TrackCorners.race_id == 1
    ).all()

corner_dicts = [obj.__dict__ for obj in corners]

for d in corner_dicts:
    d.pop('_sa_instance_state', None)

corner_df = pd.DataFrame(corner_dicts)

corner_df = corner_df.sort_values(by="number")
corner_dist = corner_df["distance"].to_list()




# Color mapping
compound_colors = {
    'SOFT': '#ff4d4d',
    'MEDIUM': '#ffd633',
    'HARD': '#cccccc',
    "INTERMEDIATE": "#4dd67b",
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(9, 6))

# # --- Background color bands by lap ---
for i, compound in enumerate(driver1_tire_compounds):
    color = compound_colors.get(compound, 'white')
    ax.axhspan(i + 0.5, i + 1.5, color=color, alpha=0.2)
    # ax.axvspan(i + 0.5, i + 1.5, color=color, alpha=0.2)




# Loop through each brake segment
for lap_number, start_dist, end_dist in brake_segments_driver1:
    ax.add_patch(
        patches.Rectangle(
            (start_dist, lap_number -0.4 ),  # (x, y): distance and vertical center of lap row
            # (lap_number - 0.4, start_dist),  # (x, y): distance and vertical center of lap row
            end_dist - start_dist,          # width (brake zone length)
            # 0.8,
            0.8,                            # height (how thick the row bar is)
            # end_dist - start_dist,
            color='black'
        )
    )
    
# Plot turn markers as vertical lines
for turn_x in corner_dist:
    ax.axvline(x=turn_x, color='blue', linestyle='--', alpha=0.7, label='Turn')
    # ax.axhline(y=turn_x, color='blue', linestyle='--', alpha=0.7, label='Turn')


# --- Add tire compound legend ---
handles = [Patch(color=color, label=compound) for compound, color in compound_colors.items()]
ax.legend(handles=handles + [ax.lines[0]])

# Set axis limits based on data

ax.set_xlim(min_dist1 - 10, max_dist1 + 10)
# ax.set_xlim(min_dist1 - 10, max_dist1 /4)
ax.set_ylim(min_lap1 - 1, max_lap1 // 1.5)
# ax.set_ylim(min_dist - 10, max_dist + 10)
# ax.set_xlim(min_lap - 1, max_lap + 1)

# Labels and formatting
ax.set_xlabel("Track Distance (m)")
ax.set_ylabel("Lap Number")
ax.set_title("Braking Zones per Lap")
# ax.set_yticks(sorted(set(lap for lap, _, _ in brake_segments)))
# ax.set_xticks(sorted(set(lap for lap, _, _ in brake_segments)))
# ax.invert_yaxis()
ax.grid(True)
plt.tight_layout()
plt.show()






# # merge together based on time
# # Ensure both DataFrames are sorted by time
# telem_df = telem_df.sort_values("time")
# lap_df = lap_df.sort_values("time")

# # Merge gear onto position (or vice versa)
# df_merged = pd.merge_asof(
#     lap_df,
#     telem_df[["time", "brake", "distance"]],
#     on="time",
#     direction="nearest"  # can be 'backward', 'forward', or 'nearest'
# )

# print(df_merged.iloc[0])


