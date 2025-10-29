# Using SQLAlchemy to connect to database in connect_database.py 
import os
import sys
import pyodbc
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

# Get variables from environment
server = os.getenv('db_host')
database = os.getenv('db_database')
username = os.getenv('db_username')
password = os.getenv('db_password')

def connect_database():
    try:
        conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};'
            'SERVER=' + server + ';'
            'DATABASE=' + database + ';'
            'UID=' + username + ';'
            'PWD=' + password
        )
        print("Database connection....OK...")
        return conn
    except pyodbc.Error as err:
        print(err)
        print("Cannot connect to the database.")
        sys.exit()

# Function to create SQLAlchemy session
def create_sqlalchemy_session():
    conn = connect_database()  # get the pyodbc connection
    connection_string = f"mssql+pyodbc:///?odbc_connect={pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';UID='+username+';PWD='+password)}"
    
    # Use SQLAlchemy's engine to connect
    engine = create_engine(connection_string, echo=True)
    Session = sessionmaker(bind=engine)
    session = Session()  # create a session

     
    return session
