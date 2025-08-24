from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean, TIME


Base = declarative_base()



class Race(Base):
    __tablename__ = 'races'
    id = Column(Integer, primary_key=True, autoincrement=False)
    RoundNumber = Column(String(2))
    name = Column(String(100))
    eventFormat = Column(String(50))
    date = Column(DateTime)
    location = Column(String(50))
    country = Column(String(50))
    session1 = Column(String(50))
    session1Date = Column(DateTime)
    session2 = Column(String(50))
    session2Date = Column(DateTime)
    session3 = Column(String(50))
    session3Date = Column(DateTime)
    session4 = Column(String(50))
    session4Date = Column(DateTime)
    session5 = Column(String(50))
    session5Date = Column(DateTime)

    corners = relationship("TrackCorners", back_populates="race")
    driver_results = relationship("Driver", back_populates="race")
    weather = relationship("Weather", back_populates="race")
    

class Driver(Base):
    __tablename__ = 'drivers'
    id = Column(Integer, primary_key=True)
    driver_number = Column(String(2))
    abbreviation = Column(String(3))
    full_name = Column(String(50))
    race_id = Column(Integer, ForeignKey("races.id"))
    session = Column(String(50))
    Q1 = Column(Float, nullable=True)
    Q2 = Column(Float, nullable=True)
    Q3 = Column(Float, nullable=True)
    team = Column(String(50))
    position = Column(Integer, nullable=True)
    classified_pos = Column(String(10), nullable=True)
    grid_pos = Column(Integer, nullable=True)
    time = Column(Float, nullable=True)
    status = Column(String(50))
    points = Column(Float, nullable=True)
    laps = Column(Float, nullable=True)

    race = relationship("Race", back_populates="driver_results")

class Lap(Base):
    __tablename__ = 'laps'
    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey('races.id'))
    session = Column(String(50))
    driver_number = Column(Integer)
    driver_abr = Column(String(3))
    time = Column(Float)
    lap_time = Column(Float, nullable=True)
    lap_number = Column(Integer, nullable=True)
    stint = Column(Float)
    pitOutTime = Column(Float, nullable=True)
    pitInTime = Column(Float, nullable=True)
    sector1Time=Column(Float, nullable=True)
    sector2Time=Column(Float, nullable=True)
    sector3Time=Column(Float, nullable=True)
    sector1SessionTime=Column(Float, nullable=True)
    secto2SessionTime=Column(Float, nullable=True)
    sector3SessionTime=Column(Float, nullable=True)
    speedI1=Column(Float, nullable=True)
    speedI2=Column(Float, nullable=True)
    speedFL=Column(Float, nullable=True)
    speedST=Column(Float, nullable=True)
    isPersonalBest=Column(Boolean, nullable=True)
    compound=Column(String(20))
    tyreLife=Column(Float)
    freshTyre=Column(Boolean)
    team=Column(String(50))
    lapStartTime=Column(Float, nullable=True)
    lapStartDate=Column(DateTime)
    trackStatus=Column(String(10))
    position=Column(Float, nullable=True)
    deleted=Column(Boolean)
    deletedReason=Column(String(100), nullable=True)
    fastF1Generated=Column(Boolean)
    isAccurate=Column(Boolean)



class CarTelemetry(Base):
    __tablename__ = 'telemetry'
    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey('races.id'))
    session = Column(String(50))
    driver_abr = Column(String(3))
    date = Column(DateTime)
    rpm = Column(Float)
    speed = Column(Float)
    gear = Column(Integer)
    throttle = Column(Float)
    brake = Column(Boolean)
    drs = Column(Integer)
    source = Column(String(20))
    time = Column(Float, nullable=True)
    sessionTime = Column(Float, nullable=True)
    distance = Column(Float)
    driverAhead = Column(String(2), nullable=True)
    distanceToDriverAhead = Column(Float, nullable=True)
    relativeDistance = Column(Float)
    differentialDistance = Column(Float)

class CarPosition(Base):
    __tablename__ = 'position'
    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"))
    session=Column(String(50))
    driver_abr = Column(String(3))
    date=Column(DateTime)
    status=Column(String(50))
    x_pos=Column(Float)
    y_pos=Column(Float)
    z_pos=Column(Float)
    source=Column(String(20))
    time=Column(Float, nullable=True)
    sessionTime=Column(Float, nullable=True)


class Weather(Base):
    __tablename__ = 'weather'
    id=Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"))
    session = Column(String(50))
    time=Column(Float, nullable=True)
    airTemp=Column(Float)
    humidity=Column(Float)
    pressure=Column(Float)
    rainfall=Column(Boolean)
    trackTemp=Column(Float)
    windDirection=Column(Integer)
    windSpeed=Column(Float)

    race = relationship("Race", back_populates="weather")



class TrackCorners(Base):
    __tablename__ = 'track_corners'
    id = Column(Integer, primary_key=True)
    race_id = Column(Integer, ForeignKey("races.id"))
    x_cord = Column(Float)
    y_cord = Column(Float)
    number = Column(Integer)
    angle = Column(Float)
    distance = Column(Float)
    rotation = Column(Float)

    race = relationship("Race", back_populates="corners")



