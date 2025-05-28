import os
import logging
from contextlib import contextmanager
from typing import Optional
from dataclasses import dataclass
import urllib.parse

from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import sql

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

@dataclass
class MovingRequest:
    request_id: str
    customer_name: str
    email: str
    phone_number: str
    phone_type: str  # cell, home, or work
    from_address: str
    from_building_type: str  # house or apartment
    from_bedrooms: int
    to_address: str
    move_date: str
    flexible_date: bool
    assist_car: bool
    car_year: Optional[str]
    car_make: Optional[str]
    car_model: Optional[str]

class DatabaseDriver:
    def __init__(self, db_url: str = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable must be set for PostgreSQL.")
        
        # Parse the URL to ensure it's valid
        try:
            parsed = urllib.parse.urlparse(self.db_url)
            logger.info(f"Connecting to PostgreSQL at: {parsed.hostname}:{parsed.port}")
        except Exception as e:
            logger.error(f"Invalid DATABASE_URL format: {e}")
            raise ValueError(f"Invalid DATABASE_URL format: {e}")
        
        self._init_db()

    @contextmanager
    def _get_connection(self):
        conn = None
        try:
            logger.debug(f"Establishing connection to database")
            conn = psycopg2.connect(
                self.db_url, 
                cursor_factory=RealDictCursor,
                connect_timeout=30,
                sslmode='require'  # Ensure SSL for Render.com
            )
            conn.autocommit = False
            yield conn
        except psycopg2.OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
                logger.debug("Database connection closed")

    def _init_db(self):
        logger.info("Initializing database schema...")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create moving_requests table with all required fields
                create_table_query = """
                    CREATE TABLE IF NOT EXISTS moving_requests (
                        request_id TEXT PRIMARY KEY,
                        customer_name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        phone_number TEXT NOT NULL,
                        phone_type TEXT NOT NULL CHECK (phone_type IN ('cell', 'home', 'work')),
                        from_address TEXT NOT NULL,
                        from_building_type TEXT NOT NULL CHECK (from_building_type IN ('house', 'apartment')),
                        from_bedrooms INTEGER NOT NULL CHECK (from_bedrooms > 0),
                        to_address TEXT NOT NULL,
                        move_date TEXT NOT NULL,
                        flexible_date BOOLEAN NOT NULL DEFAULT FALSE,
                        assist_car BOOLEAN NOT NULL DEFAULT FALSE,
                        car_year TEXT,
                        car_make TEXT,
                        car_model TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                cursor.execute(create_table_query)
                
                # Create an index on request_id for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_moving_requests_request_id 
                    ON moving_requests(request_id)
                """)
                
                # Create a trigger to update updated_at timestamp
                cursor.execute("""
                    CREATE OR REPLACE FUNCTION update_updated_at_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ language 'plpgsql'
                """)
                
                cursor.execute("""
                    DROP TRIGGER IF EXISTS update_moving_requests_updated_at ON moving_requests
                """)
                
                cursor.execute("""
                    CREATE TRIGGER update_moving_requests_updated_at 
                    BEFORE UPDATE ON moving_requests 
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()
                """)
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def test_connection(self) -> bool:
        """Test the database connection."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                logger.info("Database connection test successful")
                return result is not None
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def create_moving_request(
        self, 
        request_id: str,
        customer_name: str,
        email: str,
        phone_number: str,
        phone_type: str,
        from_address: str,
        from_building_type: str,
        from_bedrooms: int,
        to_address: str,
        move_date: str,
        flexible_date: bool,
        assist_car: bool,
        car_year: Optional[str] = None,
        car_make: Optional[str] = None,
        car_model: Optional[str] = None
    ) -> Optional[MovingRequest]:
        logger.info(f"Creating moving request with ID: {request_id}")
        
        # Validate inputs
        if phone_type.lower() not in ['cell', 'home', 'work']:
            raise ValueError(f"Invalid phone_type: {phone_type}. Must be 'cell', 'home', or 'work'")
        
        if from_building_type.lower() not in ['house', 'apartment']:
            raise ValueError(f"Invalid from_building_type: {from_building_type}. Must be 'house' or 'apartment'")
        
        if from_bedrooms <= 0:
            raise ValueError(f"Invalid from_bedrooms: {from_bedrooms}. Must be greater than 0")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if request_id already exists
                cursor.execute("SELECT request_id FROM moving_requests WHERE request_id = %s", (request_id,))
                if cursor.fetchone():
                    logger.warning(f"Request ID {request_id} already exists, updating instead")
                    return self.update_moving_request(
                        request_id, customer_name, email, phone_number, phone_type,
                        from_address, from_building_type, from_bedrooms, to_address,
                        move_date, flexible_date, assist_car, car_year, car_make, car_model
                    )
                
                # Insert new record
                insert_query = """
                    INSERT INTO moving_requests 
                    (request_id, customer_name, email, phone_number, phone_type,
                     from_address, from_building_type, from_bedrooms, to_address,
                     move_date, flexible_date, assist_car, car_year, car_make, car_model) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_query, (
                    request_id, customer_name.strip(), email.strip(), phone_number.strip(), phone_type.lower(),
                    from_address.strip(), from_building_type.lower(), from_bedrooms, to_address.strip(),
                    move_date.strip(), flexible_date, assist_car, car_year, car_make, car_model
                ))
                
                conn.commit()
                logger.info(f"Successfully created moving request: {request_id}")
                
                # Return the created record
                return self.get_moving_request(request_id)
                
        except psycopg2.IntegrityError as e:
            logger.error(f"Integrity error creating moving request: {e}")
            raise ValueError(f"Database integrity error: {e}")
        except Exception as e:
            logger.error(f"Error creating moving request: {e}")
            raise

    def update_moving_request(
        self, 
        request_id: str,
        customer_name: str,
        email: str,
        phone_number: str,
        phone_type: str,
        from_address: str,
        from_building_type: str,
        from_bedrooms: int,
        to_address: str,
        move_date: str,
        flexible_date: bool,
        assist_car: bool,
        car_year: Optional[str] = None,
        car_make: Optional[str] = None,
        car_model: Optional[str] = None
    ) -> Optional[MovingRequest]:
        logger.info(f"Updating moving request with ID: {request_id}")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                update_query = """
                    UPDATE moving_requests SET
                        customer_name = %s, email = %s, phone_number = %s, phone_type = %s,
                        from_address = %s, from_building_type = %s, from_bedrooms = %s, to_address = %s,
                        move_date = %s, flexible_date = %s, assist_car = %s, car_year = %s, car_make = %s, car_model = %s
                    WHERE request_id = %s
                """
                
                cursor.execute(update_query, (
                    customer_name.strip(), email.strip(), phone_number.strip(), phone_type.lower(),
                    from_address.strip(), from_building_type.lower(), from_bedrooms, to_address.strip(),
                    move_date.strip(), flexible_date, assist_car, car_year, car_make, car_model, request_id
                ))
                
                if cursor.rowcount == 0:
                    logger.warning(f"No moving request found with ID: {request_id}")
                    return None
                
                conn.commit()
                logger.info(f"Successfully updated moving request: {request_id}")
                
                # Return the updated record
                return self.get_moving_request(request_id)
                
        except Exception as e:
            logger.error(f"Error updating moving request: {e}")
            raise

    def get_moving_request(self, request_id: str) -> Optional[MovingRequest]:
        logger.info(f"Looking up moving request with ID: {request_id}")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM moving_requests WHERE request_id = %s", (request_id,))
                row = cursor.fetchone()
                if not row:
                    logger.info(f"No moving request found with ID: {request_id}")
                    return None
                
                logger.info(f"Found moving request: {request_id}")
                return MovingRequest(
                    request_id=row['request_id'],
                    customer_name=row['customer_name'],
                    email=row['email'],
                    phone_number=row['phone_number'],
                    phone_type=row['phone_type'],
                    from_address=row['from_address'],
                    from_building_type=row['from_building_type'],
                    from_bedrooms=row['from_bedrooms'],
                    to_address=row['to_address'],
                    move_date=row['move_date'],
                    flexible_date=row['flexible_date'],
                    assist_car=row['assist_car'],
                    car_year=row['car_year'],
                    car_make=row['car_make'],
                    car_model=row['car_model']
                )
        except Exception as e:
            logger.error(f"Error retrieving moving request: {e}")
            raise

    def delete_moving_request(self, request_id: str) -> bool:
        """Delete a moving request by ID."""
        logger.info(f"Deleting moving request with ID: {request_id}")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM moving_requests WHERE request_id = %s", (request_id,))
                deleted = cursor.rowcount > 0
                conn.commit()
                
                if deleted:
                    logger.info(f"Successfully deleted moving request: {request_id}")
                else:
                    logger.warning(f"No moving request found to delete with ID: {request_id}")
                
                return deleted
        except Exception as e:
            logger.error(f"Error deleting moving request: {e}")
            raise

    def list_all_requests(self) -> list[MovingRequest]:
        """List all moving requests."""
        logger.info("Retrieving all moving requests")
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM moving_requests ORDER BY created_at DESC")
                rows = cursor.fetchall()
                
                requests = []
                for row in rows:
                    requests.append(MovingRequest(
                        request_id=row['request_id'],
                        customer_name=row['customer_name'],
                        email=row['email'],
                        phone_number=row['phone_number'],
                        phone_type=row['phone_type'],
                        from_address=row['from_address'],
                        from_building_type=row['from_building_type'],
                        from_bedrooms=row['from_bedrooms'],
                        to_address=row['to_address'],
                        move_date=row['move_date'],
                        flexible_date=row['flexible_date'],
                        assist_car=row['assist_car'],
                        car_year=row['car_year'],
                        car_make=row['car_make'],
                        car_model=row['car_model']
                    ))
                
                logger.info(f"Retrieved {len(requests)} moving requests")
                return requests
        except Exception as e:
            logger.error(f"Error retrieving all moving requests: {e}")
            raise