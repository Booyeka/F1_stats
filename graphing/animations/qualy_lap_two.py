import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import interp1d
from matplotlib.patches import Rectangle
import math
import json

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from dotenv import load_dotenv
from db.db_connection import get_db_conn_alc
import pandas as pd
import numpy as np
from db.models import Race, Lap, Driver, CarTelemetry, CarPosition, Weather
import matplotlib.patches as patches


if __name__ == "__main__":

    load_dotenv()  # Load from .env
    engine = get_db_conn_alc() # connect to mysql f1_data database
    Session = sessionmaker(bind=engine)

    with open("/Users/neilg/Documents/Repos/F1_stats/graphing/animations/race_id.json", "r") as openfile:
        race_id = json.load(openfile)
    race_id = race_id[0]



    # get driver with pole position
    with Session() as session:
        driver_pole = session.query(Driver).filter(
            Driver.race_id == race_id,
            Driver.grid_pos == 1,
            Driver.session == "Race"
        ).all()
        second_pos = session.query(Driver).filter(
            Driver.race_id == race_id,
            Driver.grid_pos == 2,
            Driver.session == "Race"
        ).all()
    race_dicts1 = [obj.__dict__ for obj in driver_pole]
    race_dicts2 = [obj.__dict__ for obj in second_pos]
    for d in race_dicts1:
        d.pop('_sa_instance_state', None)    
    pole_driver = race_dicts1[0]["abbreviation"]
    for d in race_dicts2:
        d.pop('_sa_instance_state', None)
    second_driver = race_dicts2[0]["abbreviation"]




    def prepare_lap_data(position_data, telemetry_data, num_points=500):
        """
        Syncs position and telemetry datasets to have the same length using lap distance.
        """
        num_points = len(position_data["x"])
        # Extract position info
        seshtime = np.array(position_data['times'])
        x_vals = np.array(position_data['x'])
        y_vals = np.array(position_data['y'])
        
        # Extract telemetry info
        sessiontime = np.array(telemetry_data["sessionTime"])
        # tel_dist = np.array(telemetry_data['lap_distance'])
        speed = np.array(telemetry_data['speed'])
        throttle = np.array(telemetry_data['throttle'])
        brake = np.array(telemetry_data['brake'])
        rpm = np.array(telemetry_data['rpm'])
        gear = np.array(telemetry_data['gear'])
        tyre = np.array(telemetry_data['tyre'])  # e.g. string or int

        # Create common lap time array
        # max_dist = min(pos_dist.max(), tel_dist.max())  # safest so both cover range
        max_dist = seshtime.max()
        common_dist = np.linspace(seshtime.min(), max_dist, num_points)

        # Interpolation helper
        def interp(series_dist, series_vals):
            return interp1d(series_dist, series_vals, kind='linear', fill_value="extrapolate")(common_dist)

        # # Resample position
        # x_resampled = interp(pos_dist, x_vals)
        # y_resampled = interp(pos_dist, y_vals)

        # Resample telemetry
        speed_resampled = interp(sessiontime, speed)
        throttle_resampled = interp(sessiontime, throttle)
        brake_resampled = interp(sessiontime, brake)
        rpm_resampled = interp(sessiontime, rpm)
        gear_resampled = interp(sessiontime, gear)

        # For tyre compound (categorical), pick nearest value
        # tyre_resampled = np.array([
        #     tyre[np.abs(sessiontime - d).argmin()] for d in common_dist
        # ])

        # Return single merged dict
        return {
            # 'distance': common_dist,
            "times": seshtime,
            'x': x_vals,
            'y': y_vals,
            'speed': speed_resampled,
            'throttle': throttle_resampled,
            'brake': brake_resampled,
            'rpm': rpm_resampled,
            'gear': gear_resampled,
            # 'tyre': tyre_resampled
        }


    def offset_track_edges(x, y, width=5.0):
        # Convert to arrays
        x, y = np.array(x), np.array(y)

        # Compute tangents (dx, dy)
        dx = np.gradient(x)
        dy = np.gradient(y)

        # Normalize tangents
        length = np.hypot(dx, dy)
        dx /= length
        dy /= length

        # Compute normals (perpendicular to tangents)
        nx = -dy
        ny = dx

        # Offset edges
        left_x = x + nx * (width / 2)
        left_y = y + ny * (width / 2)
        right_x = x - nx * (width / 2)
        right_y = y - ny * (width / 2)

        return left_x, left_y, right_x, right_y



    


    with Session() as session:
        race_info = session.query(Race).filter(
            Race.id == race_id
        ).all()
    race_dicts = [obj.__dict__ for obj in race_info]
    for d in race_dicts:
        d.pop('_sa_instance_state', None)
    # print(race_dicts)

    # PULL DRIVER RESULT TO FIND Q3 TIME
    with Session() as session:
        driver_res = session.query(Driver).filter(
            Driver.session == "qualifying",
            # Driver.session != "sprint qualifying",
            Driver.race_id == race_id,
            Driver.abbreviation == f"{pole_driver}",
        ).all()
        driver_2_res = session.query(Driver).filter(
            Driver.session == "qualifying",
            # Driver.session != "sprint qualifying",
            Driver.race_id == race_id,
            Driver.abbreviation == f"{second_driver}",
        ).all()
        

    driver_dicts1 = [obj.__dict__ for obj in driver_res]
    for d in driver_dicts1:
        d.pop('_sa_instance_state', None)
    driver_df = pd.DataFrame(driver_dicts1)
    laptime1 = round(driver_df.iloc[0]["Q3"],3)

    driver_dicts2 = [obj.__dict__ for obj in driver_2_res]
    for d in driver_dicts2:
        d.pop('_sa_instance_state', None)
    driver_df = pd.DataFrame(driver_dicts2)
    laptime2 = round(driver_df.iloc[0]["Q3"],3)
  

    # PULL LAPS AND FILTER FOR Q3 TIME
    tolerance = 0.001
    with Session() as session:
        fastest_lap1 = session.query(Lap).filter(
            Lap.session == "qualifying",
            # Lap.session != "sprint qualifying",
            Lap.race_id == race_id,
            Lap.driver_abr == f"{pole_driver}",
            Lap.lap_time.between(laptime1 - tolerance, laptime1 + tolerance)
        ).all()
        fastest_lap2 = session.query(Lap).filter(
            Lap.session == "qualifying",
            # Lap.session != "sprint qualifying",
            Lap.race_id == race_id,
            Lap.driver_abr == f"{second_driver}",
            Lap.lap_time.between(laptime2 - tolerance, laptime2 + tolerance)
        ).all()

    lap_dicts1 = [obj.__dict__ for obj in fastest_lap1]
    for d in lap_dicts1:
        d.pop('_sa_instance_state', None)
    lap_df1 = pd.DataFrame(lap_dicts1)

    lap_dicts2 = [obj.__dict__ for obj in fastest_lap2]
    for d in lap_dicts2:
        d.pop('_sa_instance_state', None)
    lap_df2 = pd.DataFrame(lap_dicts2)


    # GET TIME VALUES TO PULL TELEMTRY FOR LAP
    lap_start1 = lap_df1.iloc[0]["lapStartTime"]
    lap_end1 = lap_df1.iloc[0]["time"]

    lap_start2 = lap_df2.iloc[0]["lapStartTime"]
    lap_end2 = lap_df2.iloc[0]["time"]

    with Session() as session:
        telem1 = session.query(CarTelemetry).filter(
            CarTelemetry.race_id == race_id,
            CarTelemetry.session == "qualifying",
            # CarTelemetry.session != "sprint qualifying",
            CarTelemetry.driver_abr == f"{pole_driver}",
            CarTelemetry.sessionTime.between(lap_start1, lap_end1)
        ).all()
        telem2 = session.query(CarTelemetry).filter(
            CarTelemetry.race_id == race_id,
            CarTelemetry.session == "qualifying",
            # CarTelemetry.session != "sprint qualifying",
            CarTelemetry.driver_abr == f"{second_driver}",
            CarTelemetry.sessionTime.between(lap_start2, lap_end2)
        ).all()


    telem_dicts1 = [obj.__dict__ for obj in telem1]
    for d in telem_dicts1:
        d.pop('_sa_instance_state', None)
    telem_df1 = pd.DataFrame(telem_dicts1)
    telem_df1.sort_values(by=["sessionTime"])

    telem_dicts2 = [obj.__dict__ for obj in telem2]
    for d in telem_dicts2:
        d.pop('_sa_instance_state', None)
    telem_df2 = pd.DataFrame(telem_dicts2)
    telem_df2.sort_values(by=["sessionTime"])
 

    # USING SAME TIME VALUES, GET X/Y DATA:

    with Session() as session:
        pos1 = session.query(CarPosition).filter(
            CarPosition.race_id == race_id,
            CarPosition.session == "qualifying",
            # CarPosition.session != "sprint qualifying",
            CarPosition.driver_abr == f"{pole_driver}",
            CarPosition.sessionTime.between(lap_start1, lap_end1)
        ).all()
        pos2 = session.query(CarPosition).filter(
            CarPosition.race_id == race_id,
            CarPosition.session == "qualifying",
            # CarPosition.session != "sprint qualifying",
            CarPosition.driver_abr == f"{second_driver}",
            CarPosition.sessionTime.between(lap_start2, lap_end2)
        ).all()


    pos_dicts1 = [obj.__dict__ for obj in pos1]
    for d in pos_dicts1:
        d.pop('_sa_instance_state', None)
    pos_df1 = pd.DataFrame(pos_dicts1)
    pos_df1.sort_values(by=["sessionTime"])
   
    pos_dicts2 = [obj.__dict__ for obj in pos2]
    for d in pos_dicts2:
        d.pop('_sa_instance_state', None)
    pos_df2 = pd.DataFrame(pos_dicts2)
    pos_df2.sort_values(by=["sessionTime"])

    with Session() as session:
        weather = session.query(Weather).filter(
            Weather.race_id == race_id,
            Weather.session == "qualifying",
            Weather.session != "sprint qualifying",
            Weather.time.between(lap_start1, lap_end1)
        )
    weather_dicts = [obj.__dict__ for obj in weather]
    for d in weather_dicts:
        d.pop('_sa_instance_state', None)

    weather_df = pd.DataFrame(weather_dicts)
    weather_df.sort_values(by=["time"])
    # print(weather_df)

    lap_pos1 = {
        "x_pos" : [],
        "y_pos" : [],
        "seshtime" : []
    }
    lap_telem1 = {
        "speed" : [],
        "throttle" : [],
        "brake" : [],
        "rpm" : [],
        "gear" : [],
        "tyre" : [],
        "tyre_life" : [],
        "lap_num" : [],
        "lap_time" : [],
        "sessionTime" : []
    }
    lap_pos2 = {
        "x_pos" : [],
        "y_pos" : [],
        "seshtime" : []
    }
    lap_telem2 = {
        "speed" : [],
        "throttle" : [],
        "brake" : [],
        "rpm" : [],
        "gear" : [],
        "tyre" : [],
        "tyre_life" : [],
        "lap_num" : [],
        "lap_time" : [],
        "sessionTime" : []
    }
    for row in range(len(pos_df1["session"])):
        lap_pos1["x_pos"].append(
            pos_df1.iloc[row]["x_pos"]
        )
        lap_pos1["y_pos"].append(
            pos_df1.iloc[row]["y_pos"]
        )
        lap_pos1["seshtime"].append(
            pos_df1.iloc[row]["sessionTime"]
        )
    for row in range(len(pos_df2["session"])):
        lap_pos2["x_pos"].append(
            pos_df2.iloc[row]["x_pos"]
        )
        lap_pos2["y_pos"].append(
            pos_df2.iloc[row]["y_pos"]
        )
        lap_pos2["seshtime"].append(
            pos_df2.iloc[row]["sessionTime"]
        )

    for row in range(len(telem_df1["session"])):
        lap_telem1["sessionTime"].append(
                telem_df1.iloc[row]["sessionTime"]
        )
        lap_telem1["speed"].append(
                telem_df1.iloc[row]["speed"]
        )
        lap_telem1["throttle"].append(
                telem_df1.iloc[row]["throttle"]
        )
        lap_telem1["brake"].append(
                telem_df1.iloc[row]["brake"]
        )
        lap_telem1["rpm"].append(
                telem_df1.iloc[row]["rpm"]
        )
        lap_telem1["gear"].append(
                telem_df1.iloc[row]["gear"]
        )
        lap_telem1["tyre"].append(
                lap_df1.iloc[0]["compound"]
        )
        lap_telem1["tyre_life"].append(
                lap_df1.iloc[0]["tyreLife"]
        )
        lap_telem1["lap_num"].append(
                lap_df1.iloc[0]["lap_number"]
        )
        lap_telem1["lap_time"].append(
                lap_df1.iloc[0]["lap_time"]
        )

    for row in range(len(telem_df2["session"])):
        lap_telem2["sessionTime"].append(
                telem_df2.iloc[row]["sessionTime"]
        )
        lap_telem2["speed"].append(
                telem_df2.iloc[row]["speed"]
        )
        lap_telem2["throttle"].append(
                telem_df2.iloc[row]["throttle"]
        )
        lap_telem2["brake"].append(
                telem_df2.iloc[row]["brake"]
        )
        lap_telem2["rpm"].append(
                telem_df2.iloc[row]["rpm"]
        )
        lap_telem2["gear"].append(
                telem_df2.iloc[row]["gear"]
        )
        lap_telem2["tyre"].append(
                lap_df2.iloc[0]["compound"]
        )
        lap_telem2["tyre_life"].append(
                lap_df2.iloc[0]["tyreLife"]
        )
        lap_telem2["lap_num"].append(
                lap_df2.iloc[0]["lap_number"]
        )
        lap_telem2["lap_time"].append(
                lap_df2.iloc[0]["lap_time"]
        )


    # Assume lap_data is a list of telemetry rows with x, y, time (or sessionTime)
    position_data1 = {
        "x" : np.array(lap_pos1["x_pos"]),
        "y" : np.array(lap_pos1["y_pos"]),
        "times" : np.array(lap_pos1["seshtime"])
    }

    telemetry_data1 ={
        "sessionTime" : np.array(lap_telem1["sessionTime"]),
        "speed" : np.array(lap_telem1["speed"]),         # km/h
        "throttle" : np.array(lap_telem1["throttle"]),     # %
        "brake" : np.array(lap_telem1["brake"]),        # %
        "rpm" : np.array(lap_telem1["rpm"]),           # engine rpm
        "gear" : np.array(lap_telem1["gear"]),           # gear number
        "tyre" : np.array(lap_telem1["tyre"]),                    # single value or list
        "tyre_life" : np.array(lap_telem1["tyre_life"]),   
        "lap_num" : np.array(lap_telem1["lap_num"]),          # gear number
        "lap_time" : np.array(lap_telem1["lap_time"]),  
    }
    position_data2 = {
        "x" : np.array(lap_pos2["x_pos"]),
        "y" : np.array(lap_pos2["y_pos"]),
        "times" : np.array(lap_pos2["seshtime"])
    }

    telemetry_data2 ={
        "sessionTime" : np.array(lap_telem2["sessionTime"]),
        "speed" : np.array(lap_telem2["speed"]),         # km/h
        "throttle" : np.array(lap_telem2["throttle"]),     # %
        "brake" : np.array(lap_telem2["brake"]),        # %
        "rpm" : np.array(lap_telem2["rpm"]),           # engine rpm
        "gear" : np.array(lap_telem2["gear"]),           # gear number
        "tyre" : np.array(lap_telem2["tyre"]),                    # single value or list
        "tyre_life" : np.array(lap_telem2["tyre_life"]),   
        "lap_num" : np.array(lap_telem2["lap_num"]),          # gear number
        "lap_time" : np.array(lap_telem2["lap_time"]),  
    }

    # # Normalize time to start from 0
    # start_time = position_data["times"][0]
    # times = [t - start_time for t in position_data["times"]]



    lap_data1 = prepare_lap_data(position_data1, telemetry_data1)
    lap_data2 = prepare_lap_data(position_data2, telemetry_data2)
    


    # Function to map speed to color (red slow → green fast)
    def speed_to_color(s):
        norm_speed1 = (s - min(lap_data1["speed"])) / (max(lap_data1["speed"]) - min(lap_data1["speed"]))  # normalize 0–1
        norm_speed2 = (s - min(lap_data2["speed"])) / (max(lap_data2["speed"]) - min(lap_data2["speed"]))  # normalize 0–1
        return (plt.cm.RdYlGn(norm_speed1), plt.cm.RdYlGn(norm_speed2))

    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))


    # Create track edges
    left_x1, left_y1, right_x1, right_y1 = offset_track_edges(lap_data1['x'], lap_data1['y'], width=120.0)
    left_x2, left_y2, right_x2, right_y2 = offset_track_edges(lap_data2['x'], lap_data2['y'], width=120.0)

    # Plot static edges
    ax.plot(left_x1, left_y1, color='gray', linewidth=2, zorder=0)
    ax.plot(right_x1, right_y1, color='gray', linewidth=2, zorder=0)

    ax.plot(left_x2, left_y2, color='gray', linewidth=2, zorder=0)
    ax.plot(right_x2, right_y2, color='gray', linewidth=2, zorder=0)



    ax.set_aspect('equal')
    ax.set_title(f'-- 1st VERUS 2nd QUALIFYING -- ')

    # ax.set_aspect('equal')
    # ax.set_title(f'{lap_df2.iloc[0]["driver_abr"]} -- 2nd QUALIFYING POSITION -- ')

    # Car rectangle
    trail1, = ax.plot([], [], 'k-', lw=8)        # black trail
    car1, = ax.plot([], [], 'bo', markersize=10)  # red circle for now

    trail2, = ax.plot([], [], 'k-', lw=8)        # black trail
    car2, = ax.plot([], [], 'ko', markersize=10)  # red circle for now


    # Telemetry text
    telemetry_text1 = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.axis('off')

    telemetry_text2 = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.axis('off')


    # Inset axes for overlay (values are relative to figure, not data)
    overlay_ax1 = fig.add_axes([0.0, 0.15, 0.2, 0.2])  # [x0, y0, width, height]
    overlay_ax1.set_xlim(0, 1)
    overlay_ax1.set_ylim(0, 1)
    overlay_ax1.axis('off')  # Hide ticks

    overlay_ax2 = fig.add_axes([0.5, 0.15, 0.2, 0.2])  # [x0, y0, width, height]
    overlay_ax2.set_xlim(0, 1)
    overlay_ax2.set_ylim(0, 1)
    overlay_ax2.axis('off')  # Hide ticks

    wind_dir_deg = weather_df.iloc[0]["windDirection"]
    # Convert FROM direction to TO direction for arrow
    arrow_angle_rad = np.deg2rad((wind_dir_deg + 180) % 360)
    # wind_rad =  np.deg2rad(weather_df.iloc[0]["windDirection"])

    # Arrow length
    length = 0.15

    # dx, dy for the arrow (direction arrow is pointing *to*)
    dx = length * np.cos(arrow_angle_rad)
    dy = length * np.sin(arrow_angle_rad)

    # Draw the arrow
    # overlay_ax1.arrow(0.2, 3.2, dx, dy, transform=overlay_ax1.transAxes,width=0.04,head_width=0.08, head_length=0.08, fc='black', ec='black',clip_on=False)
    # wind_text1 = overlay_ax1.text(0.4, 3.25, 'Wind Direction', transform=overlay_ax1.transAxes, fontsize=12, verticalalignment='top')
    # wind_speed1 = overlay_ax1.text(0.4, 3.0, 'Wind Speed: ', transform=overlay_ax1.transAxes, fontsize=12, verticalalignment='top')

    # overlay_ax2.arrow(0.2, 3.2, dx, dy, transform=overlay_ax2.transAxes,width=0.04,head_width=0.08, head_length=0.08, fc='black', ec='black',clip_on=False)
    # wind_text2 = overlay_ax2.text(0.4, 3.25, 'Wind Direction', transform=overlay_ax1.transAxes, fontsize=12, verticalalignment='top')
    # wind_speed2 = overlay_ax2.text(0.4, 3.0, 'Wind Speed: ', transform=overlay_ax1.transAxes, fontsize=12, verticalalignment='top')


    throttle_text1 = overlay_ax1.text(0.55, -0.05, '', transform=overlay_ax1.transAxes, fontsize=12, verticalalignment='top')
    brake_text1 = overlay_ax1.text(0.05, -0.05, '', transform=overlay_ax1.transAxes, fontsize=12, verticalalignment='top')
    other_text1 = overlay_ax1.text(0.10, 3.0, '', transform=overlay_ax1.transAxes, fontsize=12, verticalalignment='top')
    telem_text1 = overlay_ax1.text(2.25, 0.50, '', transform=overlay_ax1.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right', color='blue')
    driver_text1 = overlay_ax1.text(0.10, 1.20, '', transform=overlay_ax1.transAxes, fontsize=12, verticalalignment='top', color='blue')

    throttle_text2 = overlay_ax2.text(1.85, -0.05, '', transform=overlay_ax2.transAxes, fontsize=12, verticalalignment='top')
    brake_text2 = overlay_ax2.text(1.35, -0.05, '', transform=overlay_ax2.transAxes, fontsize=12, verticalalignment='top')
    other_text2 = overlay_ax2.text(1.75, 3.0, '', transform=overlay_ax2.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='left')
    telem_text2 = overlay_ax2.text(-0.15, 0.50, '', transform=overlay_ax2.transAxes, fontsize=12, verticalalignment='top')
    driver_text2 = overlay_ax2.text(1.50, 1.20, '', transform=overlay_ax2.transAxes, fontsize=12, verticalalignment='top')


    # Outlines for throttle/brake (full range boxes)
    throttle_outline1 = patches.Rectangle(
        (0.6, 0.0), 0.2, 1.0, linewidth=1.5, edgecolor='black', facecolor='none'
    )
    brake_outline1 = patches.Rectangle(
        (0.1, 0.0), 0.2, 1.0, linewidth=1.5, edgecolor='black', facecolor='none'
    )
    overlay_ax1.add_patch(throttle_outline1)
    overlay_ax1.add_patch(brake_outline1)
    throttle_outline2 = patches.Rectangle(
        (1.95, 0.0), 0.2, 1.0, linewidth=1.5, edgecolor='black', facecolor='none' , clip_on=False)
    brake_outline2 = patches.Rectangle(
        (1.45, 0.0), 0.2, 1.0, linewidth=1.5, edgecolor='black', facecolor='none', clip_on=False)
    overlay_ax2.add_patch(throttle_outline2)
    overlay_ax2.add_patch(brake_outline2)

    # Fill bars (start empty)
    throttle_bar1 = patches.Rectangle((0.6, 0.0), 0.2, 0, facecolor='green')
    brake_bar1 = patches.Rectangle((0.1, 0.0), 0.2, 0, facecolor='red')
    overlay_ax1.add_patch(throttle_bar1)
    overlay_ax1.add_patch(brake_bar1)

    throttle_bar2 = patches.Rectangle((1.95, 0.0), 0.2, 0, facecolor='green', clip_on=False)
    brake_bar2 = patches.Rectangle((1.45, 0.0), 0.2, 0, facecolor='red', clip_on=False)
    overlay_ax2.add_patch(throttle_bar2)
    overlay_ax2.add_patch(brake_bar2)

    # Labels
    # overlay_ax.text(0.1, 0.45, "THROTTLE", color='white', fontsize=8)
    # overlay_ax.text(0.5, 0.45, "BRAKE", color='white', fontsize=8)

    # Init function
    def init():
        car1.set_data([], [])
        car2.set_data([], [])
        trail1.set_data([], [])
        trail2.set_data([], [])
        throttle_bar1.set_height(0)
        throttle_bar2.set_height(0)
        brake_bar1.set_height(0)
        brake_bar2.set_height(0)
        return car1, trail1, throttle_bar1, brake_bar1, car2, trail2, throttle_bar2, brake_bar2

    # Update function
    def update(frame):
        car1.set_data([lap_data1['x'][frame]], [lap_data1['y'][frame]])
        trail1.set_data([lap_data1['x'][:frame+1]], [lap_data1['y'][:frame+1]])
        # Change trail color based on current speed
        trail1.set_color(speed_to_color(lap_data1["speed"][frame])[0])
        # Update throttle and brake bars
        throttle_height1 = (lap_data1["throttle"][frame] ) / 100
        brake_height1 = (lap_data1["brake"][frame] ) * 100
        throttle_bar1.set_height(throttle_height1)
        brake_bar1.set_height(brake_height1)
        throttle_text1.set_text(
            f"Throttle: {lap_data1["throttle"][frame]:.0f}%\n"
        )
        brake_text1.set_text(
            f"Brake: {lap_data1["brake"][frame]*100:.0f}%\n"
        )
        other_text1.set_text(
            f"Lap Time: {telemetry_data1["lap_time"][0]}\n\n"
            f"Tyre: {telemetry_data1["tyre"][0]}\n\n"
            f"Tyre life: {int(telemetry_data1["tyre_life"][0])} Laps \n\n" 
        )
        telem_text1.set_text(
            f"{lap_data1["speed"][frame]:.1f}\n\n"
            f"{lap_data1["rpm"][frame]:.0f}\n\n"
            f"{math.floor(lap_data1["gear"][frame])}\n\n"
        )
        driver_text1.set_text(
            f"POLE: {pole_driver} - {race_dicts1[0]["team"]}"
        )

         # Other car - only update if it has data left
        if frame < len(lap_data2['x']):
            car2.set_data([lap_data2['x'][frame]], [lap_data2['y'][frame]])
            trail2.set_data([lap_data2['x'][:frame+1]], [lap_data2['y'][:frame+1]])
            trail2.set_color(speed_to_color(lap_data2["speed"][frame])[1])
            throttle_height2 = (lap_data2["throttle"][frame] ) / 100
    
            brake_height2 = (lap_data2["brake"][frame] ) 
            
            throttle_bar2.set_height(throttle_height2)
            
            brake_bar2.set_height(brake_height2)
            throttle_text2.set_text(
            f"Throttle: {lap_data2["throttle"][frame]:.0f}%\n"
            )
            
            brake_text2.set_text(
                f"Brake: {lap_data2["brake"][frame]*100:.0f}%\n"
            )
            
            other_text2.set_text(
                f"Lap Time: {telemetry_data2["lap_time"][0]}\n\n"
                f"Tyre: {telemetry_data2["tyre"][0]}\n\n"
                f"Tyre life: {int(telemetry_data2["tyre_life"][0])} Laps \n\n" 
            )
            
            telem_text2.set_text(
                f"SPEED    {lap_data2["speed"][frame]:.1f}\n\n"
                f"RPM      {lap_data2["rpm"][frame]:.0f}\n\n"
                f"GEAR     {math.floor(lap_data2["gear"][frame])}\n\n"
            )
            
            driver_text2.set_text(
                f"2nd: {second_driver} - {race_dicts2[0]["team"]}"
            )
        else:
            # Hold position at last known point
            car2.set_data([lap_data2['x'][-1]], [lap_data2['y'][-1]])
            trail2.set_data([lap_data2['x'][:-1]], [lap_data2['y'][:-1]])
            trail2.set_color(speed_to_color(lap_data2["speed"][-1])[1])
            throttle_height2 = (lap_data2["throttle"][-1] ) / 100
    
            brake_height2 = (lap_data2["brake"][-1] ) 
            
            throttle_bar2.set_height(throttle_height2)
            
            brake_bar2.set_height(brake_height2)
            throttle_text2.set_text(
            f"Throttle: {lap_data2["throttle"][-1]:.0f}%\n"
            )
            
            brake_text2.set_text(
                f"Brake: {lap_data2["brake"][-1]*100:.0f}%\n"
            )
            
            other_text2.set_text(
                f"Lap Time: {telemetry_data2["lap_time"][0]}\n\n"
                f"Tyre: {telemetry_data2["tyre"][0]}\n\n"
                f"Tyre life: {int(telemetry_data2["tyre_life"][0])} Laps \n\n" 
            )
            
            telem_text2.set_text(
                f"SPEED    {lap_data2["speed"][-1]:.1f}\n\n"
                f"RPM      {lap_data2["rpm"][-1]:.0f}\n\n"
                f"GEAR     {math.floor(lap_data2["gear"][-1])}\n\n"
            )
            
            driver_text2.set_text(
                f"2nd: {second_driver} - {race_dicts2[0]["team"]}"
            )

    
        
        
        # wind_speed1.set_text(
        #     f"Wind Speed: {weather_df.iloc[0]["windSpeed"]}"  #{lap_data["throttle"][frame]:.0f}%\n"
        # )
        # wind_speed2.set_text(
        #     f"Wind Speed: {weather_df.iloc[0]["windSpeed"]}"  #{lap_data["throttle"][frame]:.0f}%\n"
        # )
        
        
        


        # Set the camera box size (smaller values = more zoomed in) # use this for more zoomed in battles
        zoom_radius = 1550
        ax.set_xlim(lap_data1['x'][frame] - zoom_radius, lap_data1['x'][frame] + zoom_radius)
        ax.set_ylim(lap_data1['y'][frame] - zoom_radius, lap_data1['y'][frame] + zoom_radius)

        # ax2.set_xlim(lap_data2['x'][frame] - zoom_radius, lap_data2['x'][frame] + zoom_radius)
        # ax2.set_ylim(lap_data2['y'][frame] - zoom_radius, lap_data2['y'][frame] + zoom_radius)
        return car1, trail1, telemetry_text1, throttle_bar1, brake_bar1, car2, trail2, telemetry_text2, throttle_bar2, brake_bar2



    def create_animation():
        # Speed up 2x — assume telemetry is at 60 Hz, adjust if needed
        frame_interval_ms1 = (lap_data1["times"][1] - lap_data1["times"][0]) * 1350 / 1.00
        frame_interval_ms2 = (lap_data2["times"][1] - lap_data2["times"][0]) * 1350 / 1.00

        ani = animation.FuncAnimation(
            fig, update, frames=len(lap_data1['x']),
            init_func=init, blit=False, interval=frame_interval_ms1
        )

        writer = FFMpegWriter(fps=6, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(f"graphing/animations/finsihed_animations/{race_id}_{race_dicts[0]["location"]}_{race_dicts[0]["country"]}.mp4", writer=writer)
        # plt.show()


    create_animation()