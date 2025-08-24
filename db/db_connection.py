
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = os.getenv("DB_PORT")



def get_db_conn_alc():
    # Build connection string
    # Format: mysql+pymysql://username:password@host/db_name
    connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Create engine
    engine = create_engine(connection_string)
    return engine





