import os
import urllib.parse
import pyodbc
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

# Get variables from environment
server = os.getenv('db_host')
database = os.getenv('db_database')
username = os.getenv('db_username')
password = os.getenv('db_password')

def connect_database():
    """Create a direct PyODBC connection"""
    try:
        conn_str = (
            'DRIVER={ODBC Driver 17 for SQL Server};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID={username};'
            f'PWD={password}'
        )
        conn = pyodbc.connect(conn_str)
        print("Database connection....OK...")
        return conn
    except pyodbc.Error as err:
        print(f"Error connecting to database: {err}")
        return None

def create_sqlalchemy_session():
    """Create SQLAlchemy session with proper connection string and transaction handling
    
    Returns a context manager that properly handles session lifecycle.
    Usage: with create_sqlalchemy_session() as session:
    """
    from contextlib import contextmanager
    
    @contextmanager
    def _session_scope():
        """Provide a transactional scope around a series of operations."""
        try:
            # First create the connection string
            conn_str = (
                'DRIVER={ODBC Driver 17 for SQL Server};'
                f'SERVER={server};'
                f'DATABASE={database};'
                f'UID={username};'
                f'PWD={password}'
            )

            # URL encode the connection string
            encoded_conn_str = urllib.parse.quote_plus(conn_str)

            # Create the full SQLAlchemy URL
            connection_url = f"mssql+pyodbc:///?odbc_connect={encoded_conn_str}"

            # Create the engine with connection pooling and timeout settings
            engine = create_engine(
                connection_url,
                echo=False,  # Set to False to reduce noise
                pool_pre_ping=True,  # Test connections before use
                pool_recycle=3600,  # Recycle connections after 1 hour
                connect_args={'timeout': 30}  # Connection timeout
            )

            # Test the connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                print("Database connection test successful")

            Session = sessionmaker(bind=engine)
            session = Session()
            
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()
                engine.dispose()
                
        except Exception as e:
            print(f"Error creating SQLAlchemy session: {e}")
            raise
    
    return _session_scope()
