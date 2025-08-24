
from db_connection import get_db_conn_alc
from models import Base  # This includes all model class definitions


engine = get_db_conn_alc()

# One line to create all tables
Base.metadata.create_all(engine)