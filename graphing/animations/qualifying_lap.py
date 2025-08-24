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

    # get driver with pole position
    with Session() as session:
        driver_pole = session.query(Driver).filter(
            Driver.race_id == 1,
            Driver.grid_pos == 1
        ).all()
    race_dicts = [obj.__dict__ for obj in driver_pole]
    for d in race_dicts:
        d.pop('_sa_instance_state', None)
    # print(race_dicts)
    pole_driver = race_dicts[0]["abbreviation"]

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
            Race.id == 1
        ).all()
    race_dicts = [obj.__dict__ for obj in race_info]
    for d in race_dicts:
        d.pop('_sa_instance_state', None)
    # print(race_dicts)

    # PULL DRIVER RESULT TO FIND Q3 TIME
    with Session() as session:
        driver_res = session.query(Driver).filter(
            Driver.session == "qualifying",
            Driver.race_id == 1,
            Driver.abbreviation == f"{pole_driver}",
        ).all()

    driver_dicts = [obj.__dict__ for obj in driver_res]
    for d in driver_dicts:
        d.pop('_sa_instance_state', None)

    driver_df = pd.DataFrame(driver_dicts)
    # print(driver_df.iloc[0])
    laptime = round(driver_df.iloc[0]["Q3"],3)
    # print(laptime)

    # PULL LAPS AND FILTER FOR Q3 TIME
    tolerance = 0.001
    with Session() as session:
        fastest_lap = session.query(Lap).filter(
            Lap.session == "qualifying",
            Lap.race_id == 1,
            Lap.driver_abr == f"{pole_driver}",
            Lap.lap_time.between(laptime - tolerance, laptime + tolerance)
        ).all()

    lap_dicts = [obj.__dict__ for obj in fastest_lap]
    for d in lap_dicts:
        d.pop('_sa_instance_state', None)

    lap_df = pd.DataFrame(lap_dicts)
    # print(lap_df.iloc[0])


    # GET TIME VALUES TO PULL TELEMTRY FOR LAP
    lap_start = lap_df.iloc[0]["lapStartTime"]
    lap_end = lap_df.iloc[0]["time"]

    with Session() as session:
        telem = session.query(CarTelemetry).filter(
            CarTelemetry.race_id == 1,
            CarTelemetry.session == "qualifying",
            CarTelemetry.driver_abr == f"{pole_driver}",
            CarTelemetry.sessionTime.between(lap_start, lap_end)
        ).all()


    telem_dicts = [obj.__dict__ for obj in telem]
    for d in telem_dicts:
        d.pop('_sa_instance_state', None)

    telem_df = pd.DataFrame(telem_dicts)
    telem_df.sort_values(by=["sessionTime"])
    # print(telem_df)

    # USING SAME TIME VALUES, GET X/Y DATA:

    with Session() as session:
        pos = session.query(CarPosition).filter(
            CarPosition.race_id == 1,
            CarPosition.session == "qualifying",
            CarPosition.driver_abr == f"{pole_driver}",
            CarPosition.sessionTime.between(lap_start, lap_end)
        ).all()


    pos_dicts = [obj.__dict__ for obj in pos]
    for d in pos_dicts:
        d.pop('_sa_instance_state', None)

    pos_df = pd.DataFrame(pos_dicts)
    pos_df.sort_values(by=["sessionTime"])
    # print(pos_df)

    with Session() as session:
        weather = session.query(Weather).filter(
            Weather.race_id == 1,
            Weather.session == "qualifying",
            Weather.time.between(lap_start, lap_end)
        )
    weather_dicts = [obj.__dict__ for obj in weather]
    for d in weather_dicts:
        d.pop('_sa_instance_state', None)

    weather_df = pd.DataFrame(weather_dicts)
    weather_df.sort_values(by=["time"])
    # print(weather_df)

    lap_pos = {
        "x_pos" : [],
        "y_pos" : [],
        "seshtime" : []
    }
    lap_telem = {
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
    for row in range(len(pos_df["session"])):
        lap_pos["x_pos"].append(
            pos_df.iloc[row]["x_pos"]
        )
        lap_pos["y_pos"].append(
            pos_df.iloc[row]["y_pos"]
        )
        lap_pos["seshtime"].append(
            pos_df.iloc[row]["sessionTime"]
        )

    for row in range(len(telem_df["session"])):
        lap_telem["sessionTime"].append(
                telem_df.iloc[row]["sessionTime"]
        )
        lap_telem["speed"].append(
                telem_df.iloc[row]["speed"]
        )
        lap_telem["throttle"].append(
                telem_df.iloc[row]["throttle"]
        )
        lap_telem["brake"].append(
                telem_df.iloc[row]["brake"]
        )
        lap_telem["rpm"].append(
                telem_df.iloc[row]["rpm"]
        )
        lap_telem["gear"].append(
                telem_df.iloc[row]["gear"]
        )
        lap_telem["tyre"].append(
                lap_df.iloc[0]["compound"]
        )
        lap_telem["tyre_life"].append(
                lap_df.iloc[0]["tyreLife"]
        )
        lap_telem["lap_num"].append(
                lap_df.iloc[0]["lap_number"]
        )
        lap_telem["lap_time"].append(
                lap_df.iloc[0]["lap_time"]
        )


    # Assume lap_data is a list of telemetry rows with x, y, time (or sessionTime)
    position_data = {
        "x" : np.array(lap_pos["x_pos"]),
        "y" : np.array(lap_pos["y_pos"]),
        "times" : np.array(lap_pos["seshtime"])

    }

    telemetry_data ={
        "sessionTime" : np.array(lap_telem["sessionTime"]),
        "speed" : np.array(lap_telem["speed"]),         # km/h
        "throttle" : np.array(lap_telem["throttle"]),     # %
        "brake" : np.array(lap_telem["brake"]),        # %
        "rpm" : np.array(lap_telem["rpm"]),           # engine rpm
        "gear" : np.array(lap_telem["gear"]),           # gear number
        "tyre" : np.array(lap_telem["tyre"]),                    # single value or list
        "tyre_life" : np.array(lap_telem["tyre_life"]),   
        "lap_num" : np.array(lap_telem["lap_num"]),          # gear number
        "lap_time" : np.array(lap_telem["lap_time"]),  

    }

    # # Normalize time to start from 0
    # start_time = position_data["times"][0]
    # times = [t - start_time for t in position_data["times"]]



    lap_data = prepare_lap_data(position_data, telemetry_data)

    # Function to map speed to color (red slow → green fast)
    def speed_to_color(s):
        norm_speed = (s - min(lap_data["speed"])) / (max(lap_data["speed"]) - min(lap_data["speed"]))  # normalize 0–1
        return plt.cm.RdYlGn(norm_speed)

    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))


    # Create track edges
    left_x, left_y, right_x, right_y = offset_track_edges(lap_data['x'], lap_data['y'], width=100.0)

    # Plot static edges
    ax.plot(left_x, left_y, color='gray', linewidth=2, zorder=0)
    ax.plot(right_x, right_y, color='gray', linewidth=2, zorder=0)



    ax.set_aspect('equal')
    ax.set_title(f'{lap_df.iloc[0]["driver_abr"]} -- POLE QUALIFYING POSITION -- {race_dicts[0]["location"]}')

    # Car rectangle

    trail, = ax.plot([], [], 'k-', lw=8)        # black trail
    car, = ax.plot([], [], 'ko', markersize=10)  # red circle for now
    # Telemetry text
    telemetry_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.axis('off')


    # Inset axes for overlay (values are relative to figure, not data)
    overlay_ax = fig.add_axes([0.25, 0.15, 0.2, 0.15])  # [x0, y0, width, height]
    overlay_ax.set_xlim(0, 1)
    overlay_ax.set_ylim(0, 1)
    overlay_ax.axis('off')  # Hide ticks

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
    overlay_ax.arrow(0.1, 3.2, dx, dy, transform=overlay_ax.transAxes,width=0.04,head_width=0.08, head_length=0.08, fc='black', ec='black',clip_on=False)
    wind_text = overlay_ax.text(0.4, 3.25, 'Wind Direction', transform=overlay_ax.transAxes, fontsize=12, verticalalignment='top')
    wind_speed = overlay_ax.text(0.4, 3.0, 'Wind Speed: ', transform=overlay_ax.transAxes, fontsize=12, verticalalignment='top')

    throttle_text = overlay_ax.text(0.55, -0.05, '', transform=overlay_ax.transAxes, fontsize=12, verticalalignment='top')
    brake_text = overlay_ax.text(0.05, -0.05, '', transform=overlay_ax.transAxes, fontsize=12, verticalalignment='top')
    other_text = overlay_ax.text(0.1, 2.5, '', transform=overlay_ax.transAxes, fontsize=12, verticalalignment='top')

    # Outlines for throttle/brake (full range boxes)
    throttle_outline = patches.Rectangle(
        (0.6, 0.0), 0.2, 1.0, linewidth=1.5, edgecolor='black', facecolor='none'
    )
    brake_outline = patches.Rectangle(
        (0.1, 0.0), 0.2, 1.0, linewidth=1.5, edgecolor='black', facecolor='none'
    )
    overlay_ax.add_patch(throttle_outline)
    overlay_ax.add_patch(brake_outline)

    # Fill bars (start empty)
    throttle_bar = patches.Rectangle((0.6, 0.0), 0.2, 0, facecolor='green')
    brake_bar = patches.Rectangle((0.1, 0.0), 0.2, 0, facecolor='red')
    overlay_ax.add_patch(throttle_bar)
    overlay_ax.add_patch(brake_bar)

    # Labels
    # overlay_ax.text(0.1, 0.45, "THROTTLE", color='white', fontsize=8)
    # overlay_ax.text(0.5, 0.45, "BRAKE", color='white', fontsize=8)

    # Init function
    def init():
        car.set_data([], [])
        trail.set_data([], [])
        throttle_bar.set_height(0)
        brake_bar.set_height(0)
        return car, trail, throttle_bar, brake_bar

    # Update function
    def update(frame):
        car.set_data([lap_data['x'][frame]], [lap_data['y'][frame]])
        trail.set_data([lap_data['x'][:frame+1]], [lap_data['y'][:frame+1]])
        # Change trail color based on current speed
        trail.set_color(speed_to_color(lap_data["speed"][frame]))

        # Update throttle and brake bars
        throttle_height = (lap_data["throttle"][frame] ) / 100
        brake_height = (lap_data["brake"][frame] ) * 100
        throttle_bar.set_height(throttle_height)
        brake_bar.set_height(brake_height)
        wind_speed.set_text(
            f"Wind Speed: {weather_df.iloc[0]["windSpeed"]}"  #{lap_data["throttle"][frame]:.0f}%\n"
        )
        throttle_text.set_text(
            f"Throttle: {lap_data["throttle"][frame]:.0f}%\n"
        )
        brake_text.set_text(
            f"Brake: {lap_data["brake"][frame]*100:.0f}%\n"
        )
        other_text.set_text(
            f"Lap Time: {telemetry_data["lap_time"][0]}\n\n"
            f"Tyre: {telemetry_data["tyre"][0]}\n\n"
            f"Tyre life: {int(telemetry_data["tyre_life"][0])} Laps \n\n"
            f"Speed: {lap_data["speed"][frame]:.1f} km/h\n\n"
            f"RPM: {lap_data["rpm"][frame]:.0f}\n\n"
            f"Gear: {math.floor(lap_data["gear"][frame])}\n\n"
            
        )


        # Set the camera box size (smaller values = more zoomed in) # use this for more zoomed in battles
        zoom_radius = 1850
        ax.set_xlim(lap_data['x'][frame] - zoom_radius, lap_data['x'][frame] + zoom_radius)
        ax.set_ylim(lap_data['y'][frame] - zoom_radius, lap_data['y'][frame] + zoom_radius)
        return car, trail, telemetry_text, throttle_bar, brake_bar

    def create_animation():
        # Speed up 2x — assume telemetry is at 60 Hz, adjust if needed
        frame_interval_ms = (lap_data["times"][1] - lap_data["times"][0]) * 1350 / 1.00

        ani = animation.FuncAnimation(
            fig, update, frames=len(lap_data['x']),
            init_func=init, blit=False, interval=frame_interval_ms
        )

        writer = FFMpegWriter(fps=4, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("lap_animation_test.mp4", writer=writer)
        # plt.show()


    create_animation()