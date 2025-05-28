from __future__ import annotations
from livekit.agents import ( # type: ignore
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm
)
from livekit.agents.multimodal import MultimodalAgent # type: ignore
from livekit.plugins import openai
from dotenv import load_dotenv
from api import AssistantFnc
from prompts import WELCOME_MESSAGE, INSTRUCTIONS, LOOKUP_MOVING_INFO
import os
import re
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def entrypoint(ctx: JobContext):
    """Main entry point for the LiveKit agent."""
    logger.info("Starting LiveKit agent...")
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
        await ctx.wait_for_participant()
        
        logger.info("Participant connected, initializing model...")
        
        # Initialize the OpenAI Realtime model
        model = openai.realtime.RealtimeModel(
            instructions=INSTRUCTIONS,
            voice="alloy",
            temperature=0.8,
            modalities=["audio", "text"]
        )
        
        # Initialize assistant functions with error handling
        try:
            assistant_fnc = AssistantFnc()
            logger.info(f"Assistant functions initialized with request ID: {assistant_fnc.get_current_request_id()}")
        except Exception as e:
            logger.error(f"Failed to initialize assistant functions: {e}")
            raise
        
        # Create the multimodal agent
        assistant = MultimodalAgent(model=model, fnc_ctx=assistant_fnc)
        assistant.start(ctx.room)
        
        logger.info("Agent started successfully")
        
        # Get the session and send welcome message
        session = model.sessions[0]
        session.conversation.item.create(
            llm.ChatMessage(
                role="assistant",
                content=WELCOME_MESSAGE
            )
        )
        session.response.create()
        
        logger.info("Welcome message sent")
        
        @session.on("user_speech_committed")
        def on_user_speech_committed(msg: llm.ChatMessage):
            """Handle user speech input."""
            logger.info(f"User speech committed: {msg.content}")
            
            try:
                # Handle list content (images, etc.)
                if isinstance(msg.content, list):
                    msg.content = "\n".join("[image]" if isinstance(x, llm.ChatImage) else str(x) for x in msg.content)
                
                # Ensure content is a string
                if not isinstance(msg.content, str):
                    msg.content = str(msg.content)
                
                # Route the message based on content
                content_lower = msg.content.lower()
                
                # Check if user wants to look up their details
                if any(keyword in content_lower for keyword in ["check", "look up", "my details", "request id", "lookup"]):
                    handle_lookup_request(msg)
                else:
                    # Check if we have a complete moving request
                    if assistant_fnc.has_moving_request():
                        handle_query(msg)
                    else:
                        collect_moving_info(msg)
                        
            except Exception as e:
                logger.error(f"Error processing user message: {str(e)}")
                send_error_response("I apologize, but I encountered an error processing your request. Could you please try again?")
        
        def send_error_response(message: str):
            """Send an error response to the user."""
            try:
                session.conversation.item.create(
                    llm.ChatMessage(
                        role="system",
                        content=message
                    )
                )
                session.response.create()
            except Exception as e:
                logger.error(f"Failed to send error response: {e}")
        
        def handle_lookup_request(msg: llm.ChatMessage):
            """Handle request ID lookup."""
            logger.info("Handling lookup request")
            
            try:
                # Extract request ID if present in the message
                request_id_match = re.search(r'\b\d{6}\b', msg.content)
                if request_id_match:
                    request_id = request_id_match.group(0)
                    logger.info(f"Looking up request ID: {request_id}")
                    
                    try:
                        result = assistant_fnc.lookup_moving_request(request_id)
                        session.conversation.item.create(
                            llm.ChatMessage(
                                role="system",
                                content=f"Looking up request ID: {request_id}\n{result}"
                            )
                        )
                    except Exception as e:
                        logger.error(f"Error looking up request: {str(e)}")
                        session.conversation.item.create(
                            llm.ChatMessage(
                                role="system",
                                content="I encountered an error looking up your request. Please verify your request ID and try again."
                            )
                        )
                else:
                    session.conversation.item.create(
                        llm.ChatMessage(
                            role="system",
                            content="I'll need your request ID to look up your details. Could you please provide your 6-digit request ID?"
                        )
                    )
                
                session.response.create()
                
            except Exception as e:
                logger.error(f"Error in handle_lookup_request: {e}")
                send_error_response("I encountered an error processing your lookup request. Please try again.")
        
        def collect_moving_info(msg: llm.ChatMessage):
            """Collect moving information from user."""
            logger.info("Collecting moving information")
            
            try:
                session.conversation.item.create(
                    llm.ChatMessage(
                        role="system",
                        content=LOOKUP_MOVING_INFO(msg)
                    )
                )
                session.response.create()
            except Exception as e:
                logger.error(f"Error collecting moving info: {str(e)}")
                send_error_response("I apologize, but I encountered an error while processing your information. Could you please repeat that?")
            
        def handle_query(msg: llm.ChatMessage):
            """Handle general queries when we have a complete moving request."""
            logger.info("Handling general query")
            
            try:
                session.conversation.item.create(
                    llm.ChatMessage(
                        role="user",
                        content=msg.content
                    )
                )
                session.response.create()
            except Exception as e:
                logger.error(f"Error handling query: {str(e)}")
                send_error_response("I apologize, but I encountered an error processing your query. Could you please try again?")
    
    except Exception as e:
        logger.error(f"Critical error in entrypoint: {e}")
        raise

def validate_environment():
    """Validate that all required environment variables are set."""
    required_env_vars = [
        "LIVEKIT_URL", 
        "LIVEKIT_API_KEY", 
        "LIVEKIT_API_SECRET", 
        "OPENAI_API_KEY", 
        "DATABASE_URL"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        value = os.getenv(var)
        if not value:
            missing_vars.append(var)
        else:
            # Log first few characters for debugging (but not the full value for security)
            logger.info(f"{var}: {value[:10]}...")
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please check your .env file and ensure all required variables are set.")
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("All required environment variables are set")
    return True

def test_database_connection():
    """Test database connection before starting the agent."""
    try:
        from db_driver import DatabaseDriver
        db = DatabaseDriver()
        if db.test_connection():
            logger.info("Database connection test successful")
            return True
        else:
            logger.error("Database connection test failed")
            return False
    except Exception as e:
        logger.error(f"Database connection test error: {e}")
        return False

def main():
    """Main function to run the agent."""
    logger.info("Initializing LiveKit agent application...")
    
    try:
        # Validate environment variables
        validate_environment()
        
        # Test database connection
        if not test_database_connection():
            logger.error("Database connection failed. Please check your DATABASE_URL.")
            sys.exit(1)
        
        logger.info("Pre-flight checks passed. Starting LiveKit agent...")
        
        # Run the LiveKit agent
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
        
    except KeyboardInterrupt:
        logger.info("Agent stopped by user")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Agent failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    main()