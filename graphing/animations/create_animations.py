import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from dotenv import load_dotenv
from db.db_connection import get_db_conn_alc
from sqlalchemy.orm import sessionmaker
from sqlalchemy import func
from db.models import Driver
import json
import subprocess

# connect to sql, pull latest race_id number

load_dotenv()  # Load from .env
engine = get_db_conn_alc() # connect to mysql f1_data database
Session = sessionmaker(bind=engine)


with Session() as session:
    max_race_id = session.query(func.max(Driver.race_id)).scalar()
 


# iterate through races.json to see if animation has been created -- end value is last race_id that has data downloaded.
with open("graphing/animations/races.json", "r") as openfile:
    races = json.load(openfile)

for i in range(1,max_race_id+1):
    if races[str(i)] == False:
        race = i,
        with open("graphing/animations/race_id.json", "w") as outfile:
            json.dump(race, outfile, indent=2)
        subprocess.run(["python3", "graphing/animations/qualy_lap_two.py"], check=True)  
        races[str(i)] = True
    

with open("graphing/animations/races.json", "w") as outfile:
    json.dump(races, outfile, indent=2)



