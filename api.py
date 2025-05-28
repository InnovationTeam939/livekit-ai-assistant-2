from livekit.agents import llm
import enum
from typing import Annotated, Optional
import logging
from db_driver import DatabaseDriver
import random

logger = logging.getLogger("user-data")
logger.setLevel(logging.INFO)

# Initialize database connection
try:
    DB = DatabaseDriver()
    # Test the connection
    if DB.test_connection():
        logger.info("Database connection established successfully")
    else:
        logger.error("Database connection test failed")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    raise

def generate_request_id() -> str:
    """Generate a random 6-digit request ID."""
    return str(random.randint(100000, 999999))

class MovingDetails(enum.Enum):
    REQUEST_ID = "request_id"
    CUSTOMER_NAME = "customer_name"
    EMAIL = "email"
    PHONE_NUMBER = "phone_number"
    PHONE_TYPE = "phone_type"
    FROM_ADDRESS = "from_address"
    FROM_BUILDING_TYPE = "from_building_type"
    FROM_BEDROOMS = "from_bedrooms"
    TO_ADDRESS = "to_address"
    MOVE_DATE = "move_date"
    FLEXIBLE_DATE = "flexible_date"
    ASSIST_CAR = "assist_car"
    CAR_YEAR = "car_year"
    CAR_MAKE = "car_make"
    CAR_MODEL = "car_model"

class AssistantFnc(llm.FunctionContext):
    def __init__(self):
        super().__init__()
        self._current_request_id = generate_request_id()
        logger.info(f"Initialized new session with request ID: {self._current_request_id}")
    
    def get_current_request_id(self) -> str:
        """Get the current request ID."""
        return self._current_request_id
    
    def get_moving_request_str(self, request_id: str) -> str:
        """Format the moving request details in a clear, readable way."""
        try:
            result = DB.get_moving_request(request_id)
            if result is None:
                return "Moving request not found. Please check your request ID and try again."
            
            # Format the output with exact spacing and formatting as specified in prompts
            request_str = "Here are your moving request details:\n"
            request_str += f"Request ID: {result.request_id}\n"
            request_str += f"Customer Name: {result.customer_name}\n"
            request_str += f"Email: {result.email}\n"
            request_str += f"Phone number: {result.phone_number}\n"
            request_str += f"From Address: {result.from_address}\n"
            request_str += f"Number of Bedrooms: {result.from_bedrooms}\n"
            request_str += f"To Address: {result.to_address}\n"
            request_str += f"Move Date: {result.move_date}\n"
            request_str += f"Flexible Date: {'Yes' if result.flexible_date else 'No'}\n"
            request_str += f"Car Transport: {'Yes' if result.assist_car else 'No'}\n"
            
            # Add car details if car transport is needed
            if result.assist_car and result.car_year and result.car_make and result.car_model:
                request_str += f"Car Details: {result.car_year} {result.car_make} {result.car_model}\n"
            
            # Add a clear question with proper spacing
            request_str += "\nWould you like to make any changes to these details?"
            
            return request_str
            
        except Exception as e:
            logger.error(f"Error formatting moving request: {e}")
            return "I encountered an error retrieving your moving request details. Please try again."
    
    @llm.ai_callable(description="lookup a moving request by its ID")
    def lookup_moving_request(self, request_id: Annotated[str, llm.TypeInfo(description="The ID of the moving request to lookup")]):
        logger.info("lookup moving request - request_id: %s", request_id)
        try:
            result_str = self.get_moving_request_str(request_id)
            return f"The moving request details are: {result_str}"
        except Exception as e:
            logger.error(f"Error in lookup_moving_request: {e}")
            return "I encountered an error looking up your request. Please verify your request ID and try again."
    
    @llm.ai_callable(description="get the details of the current moving request")
    def get_moving_request_details(self):
        logger.info("get moving request details")
        try:
            result_str = self.get_moving_request_str(self._current_request_id)
            return f"The moving request details are: {result_str}"
        except Exception as e:
            logger.error(f"Error in get_moving_request_details: {e}")
            return "I encountered an error retrieving your request details. Please try again."
    
    @llm.ai_callable(description="create a new moving request")
    def create_moving_request(
        self, 
        customer_name: Annotated[str, llm.TypeInfo(description="The name of the customer")],
        email: Annotated[str, llm.TypeInfo(description="The email of the customer")],
        phone_number: Annotated[str, llm.TypeInfo(description="The phone number of the customer")],
        phone_type: Annotated[str, llm.TypeInfo(description="The type of phone (cell, home, or work)")],
        from_address: Annotated[str, llm.TypeInfo(description="The address to move from")],
        from_building_type: Annotated[str, llm.TypeInfo(description="The type of building (house or apartment)")],
        from_bedrooms: Annotated[int, llm.TypeInfo(description="The number of bedrooms")],
        to_address: Annotated[str, llm.TypeInfo(description="The address to move to")],
        move_date: Annotated[str, llm.TypeInfo(description="The date of the move")],
        flexible_date: Annotated[bool, llm.TypeInfo(description="Whether the move date is flexible")],
        assist_car: Annotated[bool, llm.TypeInfo(description="Whether car transportation is needed")],
        car_year: Annotated[Optional[str], llm.TypeInfo(description="The year of the car (if car transportation is needed)")] = None,
        car_make: Annotated[Optional[str], llm.TypeInfo(description="The make of the car (if car transportation is needed)")] = None,
        car_model: Annotated[Optional[str], llm.TypeInfo(description="The model of the car (if car transportation is needed)")] = None
    ):
        request_id = self._current_request_id
        logger.info("create moving request - request_id: %s", request_id)
        
        try:
            # Validate required car details if car transport is needed
            if assist_car and (not car_year or not car_make or not car_model):
                return "Car transportation details are incomplete. Please provide the car year, make, and model."
            
            # Normalize inputs
            phone_type = phone_type.lower().strip()
            from_building_type = from_building_type.lower().strip()
            
            # Validate phone_type
            if phone_type not in ['cell', 'home', 'work']:
                return f"Invalid phone type '{phone_type}'. Please specify 'cell', 'home', or 'work'."
            
            # Validate building type
            if from_building_type not in ['house', 'apartment']:
                return f"Invalid building type '{from_building_type}'. Please specify 'house' or 'apartment'."
            
            result = DB.create_moving_request(
                request_id, customer_name, email, phone_number, phone_type,
                from_address, from_building_type, from_bedrooms, to_address,
                move_date, flexible_date, assist_car, car_year, car_make, car_model
            )
            
            if result is None:
                return "Failed to create moving request. Please try again."
            
            logger.info(f"Successfully created moving request: {request_id}")
            return f"Moving request created! Your request ID is: {request_id}. Please save this ID for future reference."
            
        except ValueError as e:
            logger.error(f"Validation error creating moving request: {e}")
            return f"Error creating request: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error creating moving request: {e}")
            return "I encountered an error creating your moving request. Please try again."
    
    @llm.ai_callable(description="update an existing moving request")
    def update_moving_request(
        self,
        request_id: Annotated[str, llm.TypeInfo(description="The ID of the moving request to update")],
        customer_name: Annotated[str, llm.TypeInfo(description="The name of the customer")],
        email: Annotated[str, llm.TypeInfo(description="The email of the customer")],
        phone_number: Annotated[str, llm.TypeInfo(description="The phone number of the customer")],
        phone_type: Annotated[str, llm.TypeInfo(description="The type of phone (cell, home, or work)")],
        from_address: Annotated[str, llm.TypeInfo(description="The address to move from")],
        from_building_type: Annotated[str, llm.TypeInfo(description="The type of building (house or apartment)")],
        from_bedrooms: Annotated[int, llm.TypeInfo(description="The number of bedrooms")],
        to_address: Annotated[str, llm.TypeInfo(description="The address to move to")],
        move_date: Annotated[str, llm.TypeInfo(description="The date of the move")],
        flexible_date: Annotated[bool, llm.TypeInfo(description="Whether the move date is flexible")],
        assist_car: Annotated[bool, llm.TypeInfo(description="Whether car transportation is needed")],
        car_year: Annotated[Optional[str], llm.TypeInfo(description="The year of the car (if car transportation is needed)")] = None,
        car_make: Annotated[Optional[str], llm.TypeInfo(description="The make of the car (if car transportation is needed)")] = None,
        car_model: Annotated[Optional[str], llm.TypeInfo(description="The model of the car (if car transportation is needed)")] = None
    ):
        logger.info("update moving request - request_id: %s", request_id)
        
        try:
            # Validate required car details if car transport is needed
            if assist_car and (not car_year or not car_make or not car_model):
                return "Car transportation details are incomplete. Please provide the car year, make, and model."
            
            # Normalize inputs
            phone_type = phone_type.lower().strip()
            from_building_type = from_building_type.lower().strip()
            
            # Validate phone_type
            if phone_type not in ['cell', 'home', 'work']:
                return f"Invalid phone type '{phone_type}'. Please specify 'cell', 'home', or 'work'."
            
            # Validate building type
            if from_building_type not in ['house', 'apartment']:
                return f"Invalid building type '{from_building_type}'. Please specify 'house' or 'apartment'."
            
            result = DB.update_moving_request(
                request_id, customer_name, email, phone_number, phone_type,
                from_address, from_building_type, from_bedrooms, to_address,
                move_date, flexible_date, assist_car, car_year, car_make, car_model
            )
            
            if result is None:
                return f"Moving request with ID {request_id} not found or failed to update."
            
            logger.info(f"Successfully updated moving request: {request_id}")
            return f"Moving request {request_id} has been updated successfully!"
            
        except ValueError as e:
            logger.error(f"Validation error updating moving request: {e}")
            return f"Error updating request: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error updating moving request: {e}")
            return "I encountered an error updating your moving request. Please try again."
    
    def has_moving_request(self) -> bool:
        """Check if we have a complete moving request in the database."""
        try:
            result = DB.get_moving_request(self._current_request_id)
            if result is None:
                return False
                
            # Check all required fields are present and not empty
            required_fields = [
                result.customer_name,
                result.email,
                result.phone_number,
                result.phone_type,
                result.from_address,
                result.from_building_type,
                result.to_address,
                result.move_date,
            ]
            
            # Check if all required fields have values
            has_required = all(field and str(field).strip() for field in required_fields)
            
            # Check bedroom count is valid
            has_bedrooms = result.from_bedrooms and result.from_bedrooms > 0
            
            # Check car details if car transport is needed
            has_car_details = True
            if result.assist_car:
                has_car_details = all([result.car_year, result.car_make, result.car_model])
            
            return has_required and has_bedrooms and has_car_details
            
        except Exception as e:
            logger.error(f"Error checking if moving request exists: {e}")
            return False

    @llm.ai_callable(description="get additional details for a moving request")
    def get_additional_details(self, request_id: Annotated[str, llm.TypeInfo(description="The ID of the moving request")], 
                             field: Annotated[str, llm.TypeInfo(description="The field to get details for (phone_type, building_type, car_details)")]):
        """Get additional details for specific fields when requested."""
        try:
            result = DB.get_moving_request(request_id)
            if result is None:
                return "Moving request not found"
            
            field_lower = field.lower()
            if field_lower == "phone_type":
                return f"Phone type: {result.phone_type}"
            elif field_lower == "building_type":
                return f"Building Type: {result.from_building_type}"
            elif field_lower == "car_details" and result.assist_car:
                if result.car_year and result.car_make and result.car_model:
                    return f"Car Year: {result.car_year}\nCar Make: {result.car_make}\nCar Model: {result.car_model}"
                else:
                    return "Car transport is needed but car details are incomplete."
            elif field_lower == "car_details" and not result.assist_car:
                return "Car transport is not needed for this request."
            else:
                return f"Field '{field}' not found or not available."
        except Exception as e:
            logger.error(f"Error getting additional details: {e}")
            return "I encountered an error retrieving the additional details. Please try again."