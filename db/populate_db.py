import fastf1
fastf1.Cache.enable_cache('f1_cache') # stores data so no need to redownload
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from dotenv import load_dotenv
from db_connection import get_db_conn_alc
from models import Race, Lap, Driver, CarTelemetry, CarPosition, Weather, TrackCorners
import pandas as pd
import numpy as np

load_dotenv()  # Load from .env
engine = get_db_conn_alc() # connect to mysql f1_data database
Session = sessionmaker(bind=engine)
db_session = Session()




''' Populate race table  - should only run once a year'''
def events():
    races = []
    # get events for the year
    events = fastf1.get_event_schedule(year=2025)
    # print(events)
    for roundNumber in events["RoundNumber"]:
        if roundNumber == 0: # skip testing session
            continue
        event = events.iloc[roundNumber]
        races.append(
            Race(
                id=int(event["RoundNumber"]),
                RoundNumber=str(event["RoundNumber"]),
                name=str(event["OfficialEventName"]),
                eventFormat=str(event["EventFormat"]),
                date=event["EventDate"],
                location=str(event["Location"]),
                country=str(event["Country"]),
                session1=str(event["Session1"]),
                session1Date=event["Session1DateUtc"],
                session2=str(event["Session2"]),
                session2Date=event["Session2DateUtc"],
                session3=str(event["Session3"]),
                session3Date=event["Session3DateUtc"],
                session4=str(event["Session4"]),
                session4Date=event["Session4DateUtc"],
                session5=str(event["Session5"]),
                session5Date=event["Session5DateUtc"],
            ))
        
    # insert races into races table in f1_stats database
    db_session.add_all(races)
    db_session.commit()

    





def pull_weekly_updates():
    ''''
    get events remaining - use to see what races to pull data from
            -- check to see what races we have already - get data from the difference
    '''
    events_remaining = fastf1.get_events_remaining()
    # print(events_remaining.iloc[0])
    next_event = events_remaining.iloc[0]

    '''
    loop through laps to see what races already in downloaded
    '''
    # Check to see what the highest number event is in the laps db
    max_race_id = db_session.query(func.max(Lap.race_id)).scalar()
    if max_race_id == None:
        max_race_id = 0
    # print(max_race_id)

    # Check to see what number the next_event is
    next_race_id = next_event["RoundNumber"]

    with Session() as session:
        race_events = pd.read_sql(session.query(Race).statement, engine)
    # print(race_events.iloc[0])

    # loop through race events that arnt in db yet
    rounds = [i for i in range(max_race_id+1, next_race_id)]
    for round in rounds:
        session_list = [
            race_events.iloc[round-1]["session1"],
            race_events.iloc[round-1]["session2"],
            race_events.iloc[round-1]["session3"],
            race_events.iloc[round-1]["session4"],
            race_events.iloc[round-1]["session5"]
        ]
        lap_data=[]
        driver_data = []
        car_data = []
        car_pos_data = []
        weather_data = []
        # loop through sessions for each race event
        for event_session in session_list:
            event = fastf1.get_session(2025, round, f"{event_session}")
            event.load()
            '''
            INSERTING LAP DATA
            '''
            laps = event.laps
            laps = laps.replace({np.nan: None}) # converting NAN to NONE values
            # lap_data=[]
            for lap in range(len(laps["Time"])):
                data_row = laps.iloc[lap]
                lap_data.append(
                    Lap(
                        race_id=round,
                        session=event_session,
                        driver_number=int(data_row.get("DriverNumber")),
                        driver_abr=data_row.get("Driver"),
                        time=data_row.get("Time").total_seconds(),
                        lap_time=data_row.get("LapTime").total_seconds() if data_row.get("LapTime") is not None else None,
                        lap_number=data_row.get("LapNumber"),
                        stint=data_row.get("Stint"),
                        pitOutTime=data_row.get("PitOutTime").total_seconds() if data_row.get("PitOutTime") is not None else None,
                        pitInTime=data_row.get("PitInTime").total_seconds() if data_row.get("PitInTime") is not None else None,
                        sector1Time=data_row.get("Sector1Time").total_seconds() if data_row.get("Sector1Time") is not None else None,
                        sector2Time=data_row.get("Sector2Time").total_seconds() if data_row.get("Sector2Time") is not None else None,
                        sector3Time=data_row.get("Sector3Time").total_seconds() if data_row.get("Sector3Time") is not None else None,
                        sector1SessionTime=data_row.get("Sector1SessionTime").total_seconds() if data_row.get("Sector1SessionTime") is not None else None,
                        secto2SessionTime=data_row.get("Sector2SessionTime").total_seconds() if data_row.get("Sector2SessionTime") is not None else None,
                        sector3SessionTime=data_row.get("Sector3SessionTime").total_seconds() if data_row.get("Sector3SessionTime") is not None else None,
                        speedI1=data_row.get("SpeedI1") if data_row.get("SpeedI1") is not None else None,
                        speedI2=data_row.get("SpeedI2") if data_row.get("SpeedI2") is not None else None,
                        speedFL=data_row.get("SpeedFL") if data_row.get("SpeedFL") is not None else None,
                        speedST=data_row.get("SpeedST") if data_row.get("SpeedST") is not None else None,
                        isPersonalBest=data_row.get("IsPersonalBest") if data_row.get("IsPersonalBest") is not None else None,
                        compound=data_row.get("Compound"),
                        tyreLife=data_row.get("TyreLife"),
                        freshTyre=data_row.get("FreshTyre"),
                        team=data_row.get("Team"),
                        lapStartTime=data_row.get("LapStartTime").total_seconds() if data_row.get("LapStartTime") is not None else None,
                        lapStartDate=data_row.get("LapStartDate"),
                        trackStatus=str(data_row.get("TrackStatus")),
                        position=data_row.get("Position"),
                        deleted=data_row.get("Deleted"),
                        deletedReason=data_row.get("DeletedReason"),
                        fastF1Generated=data_row.get("FastF1Generated"),
                        isAccurate=data_row.get("IsAccurate")
                    )
                )

            # with Session() as session:
            #     session.add_all(lap_data)
            #     session.commit()

            # driver results 
            driver_results = event.results
            driver_results = driver_results.replace({np.nan: None}) # converting NAN to NONE values
            # driver_data = []
            for driver in range(len(driver_results["Abbreviation"])):
                data_row = driver_results.iloc[driver]
                driver_data.append(
                    Driver(
                        driver_number=str(data_row.get("DriverNumber")),
                        abbreviation=data_row.get("Abbreviation"),
                        full_name=data_row.get("FullName"),
                        race_id=round,
                        session=event_session,
                        Q1=data_row.get("Q1").total_seconds() if data_row.get("Q1") is not None else None,
                        Q2=data_row.get("Q2").total_seconds() if data_row.get("Q2") is not None else None,
                        Q3=data_row.get("Q3").total_seconds() if data_row.get("Q3") is not None else None,
                        team=data_row.get("TeamName"),
                        position=int(data_row.get("Position")) if data_row.get("Position") is not None else None,
                        classified_pos=str(data_row.get("ClassifiedPosition")) if data_row.get("ClassifiedPosition") not in [None, ""] else None,
                        grid_pos=int(data_row.get("GridPosition")) if data_row.get("GridPosition") is not None else None,
                        time=data_row.get("Time").total_seconds() if data_row.get("Time") is not None else None,
                        status=data_row.get("Status"),
                        points=data_row.get("Points") if data_row.get("Points") is not None else None,
                        laps=data_row.get("Laps") if data_row.get("Points") is not None else None
                    )
                )
            # with Session() as session:
            #     session.add_all(driver_data)
            #     session.commit()

            unique_drivers_list = driver_results['Abbreviation'].unique().tolist() # get list of unigue driver Abreviations for the current session
            # remove them from list if they have empty lap data
            for drv in unique_drivers_list:
                drv_laps = event.laps.pick_drivers(drv)
                if drv_laps.empty:
                    unique_drivers_list.remove(drv)

            # car_data = []
            # car_pos_data = []
            for driver in unique_drivers_list:
                # car data -- by driver
                car_telem = event.laps.pick_drivers(f"{driver}").get_car_data().add_distance().add_driver_ahead().add_relative_distance().add_differential_distance()
                car_telem = car_telem.replace({np.nan: None}) # converting NAN to NONE values
                for row in range(len(car_telem["Date"])):
                    data_row = car_telem.iloc[row]
                    car_data.append(
                        CarTelemetry(
                            race_id=round,
                            session=event_session,
                            driver_abr=driver,
                            date=data_row.get("Date"),
                            rpm=data_row.get("RPM"),
                            speed=data_row.get("Speed"),
                            gear=data_row.get("nGear"),
                            throttle=data_row.get("Throttle"),
                            brake=data_row.get("Brake"),
                            drs=data_row.get("DRS"),
                            source=data_row.get("Source"),
                            time=data_row.get("Time").total_seconds() if data_row.get("Time") is not None else None,
                            sessionTime=data_row.get("SessionTime").total_seconds() if data_row.get("SessionTime") is not None else None,
                            distance=data_row.get("Distance"),
                            driverAhead=data_row.get("DriverAhead") if data_row.get("DriverAhead") not in [None,""] else None,
                            distanceToDriverAhead=data_row.get("DistanceToDriverAhead") if data_row.get("DistanceToDriverAhead") is not None else None,
                            relativeDistance=data_row.get("RelativeDistance"),
                            differentialDistance=data_row.get("DifferentialDistance")
                        )
                    )

                #  postion data -- by driver
                car_pos = event.laps.pick_drivers(f"{driver}").get_pos_data()
                car_pos = car_pos.replace({np.nan: None}) # converting NAN to NONE values
                for row in range(len(car_pos["Date"])):
                    data_row = car_pos.iloc[row]
                    car_pos_data.append(
                        CarPosition(
                            race_id=round,
                            session=event_session,
                            driver_abr=driver,
                            date=data_row.get("Date"),
                            status=data_row.get("Status"),
                            x_pos=data_row.get("X"),
                            y_pos=data_row.get("Y"),
                            z_pos=data_row.get("Z"),
                            source=data_row.get("Source"),
                            time=data_row.get("Time").total_seconds() if data_row.get("Time") is not None else None,
                            sessionTime=data_row.get("SessionTime").total_seconds() if data_row.get("SessionTime") is not None else None
                        )
                    )


            # with Session() as session:
            #     session.add_all(car_data)
            #     session.add_all(car_pos_data)
            #     session.commit()

            # weather data
            weather_stats = event.weather_data
            weather_stats = weather_stats.replace({np.nan: None}) # converting NAN to NONE values
            # weather_data = []
            for row in range(len(weather_stats["Time"])):
                data_row = weather_stats.iloc[row]
                weather_data.append(
                    Weather(
                        race_id=round,
                        session=event_session,
                        time=data_row.get("Time").total_seconds() if data_row.get("Time") is not None else None,
                        airTemp=data_row.get("AirTemp"),
                        humidity=data_row.get("Humidity"),
                        pressure=data_row.get("Pressure"),
                        rainfall=data_row.get("Rainfall"),
                        trackTemp=data_row.get("TrackTemp"),
                        windDirection=data_row.get("WindDirection"),
                        windSpeed=data_row.get("WindSpeed")
                    )
                )
        # with Session() as session:
        #     session.add_all(lap_data)
        #     session.add_all(driver_data)
        #     session.add_all(car_data)
        #     session.add_all(car_pos_data)
        #     session.add_all(weather_data)
        #     session.commit()

        # circuit data 
        corner_stats = event.get_circuit_info().corners
        corner_stats = corner_stats.replace({np.nan: None}) # converting NAN to NONE values
        corner_data = []
        rotation=event.get_circuit_info().rotation
        for row in range(len(corner_stats["Number"])):
            data_row = corner_stats.iloc[row]
            corner_data.append(
                TrackCorners(
                    race_id = round,
                    x_cord =data_row.get("X"),
                    y_cord =data_row.get("Y"),
                    number =data_row.get("Number"),
                    angle =data_row.get("Angle"),
                    distance =data_row.get("Distance"),
                    rotation = rotation
                )
            )
        with Session() as session:
            session.add_all(lap_data)
            session.add_all(driver_data)
            session.add_all(car_data)
            session.add_all(car_pos_data)
            session.add_all(weather_data)
            session.add_all(corner_data)
            session.commit()




pull_weekly_updates()

