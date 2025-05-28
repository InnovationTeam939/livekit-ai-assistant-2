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
from livekit import rtc # type: ignore
from dotenv import load_dotenv
from api import AssistantFnc
from prompts import WELCOME_MESSAGE, INSTRUCTIONS, LOOKUP_MOVING_INFO
import os
import re
import logging
import sys
import asyncio
import threading
from flask import Flask, jsonify
import signal

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

class CallSession:
    """Manages a single call session with proper cleanup"""
    
    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.model = None
        self.assistant = None
        self.assistant_fnc = None
        self.session = None
        self.is_active = False
        self.cleanup_done = False
        
    async def initialize(self):
        """Initialize the call session"""
        try:
            logger.info("Initializing new call session...")
            
            # Initialize the OpenAI Realtime model
            self.model = openai.realtime.RealtimeModel(
                instructions=INSTRUCTIONS,
                voice="alloy",
                temperature=0.8,
                modalities=["audio", "text"]
            )
            
            # Initialize assistant functions with error handling
            try:
                self.assistant_fnc = AssistantFnc()
                logger.info(f"Assistant functions initialized with request ID: {self.assistant_fnc.get_current_request_id()}")
            except Exception as e:
                logger.error(f"Failed to initialize assistant functions: {e}")
                raise
            
            # Create the multimodal agent
            self.assistant = MultimodalAgent(model=self.model, fnc_ctx=self.assistant_fnc)
            self.assistant.start(self.ctx.room)
            
            # Get the session and set up event handlers
            self.session = self.model.sessions[0]
            self.setup_event_handlers()
            
            # Send welcome message
            await self.send_welcome_message()
            
            self.is_active = True
            logger.info("Call session initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize call session: {e}")
            await self.cleanup()
            raise
    
    def setup_event_handlers(self):
        """Set up event handlers for the session"""
        
        @self.session.on("user_speech_committed")
        def on_user_speech_committed(msg: llm.ChatMessage):
            """Handle user speech input."""
            if not self.is_active:
                return
                
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
                    self.handle_lookup_request(msg)
                else:
                    # Check if we have a complete moving request
                    if self.assistant_fnc.has_moving_request():
                        self.handle_query(msg)
                    else:
                        self.collect_moving_info(msg)
                        
            except Exception as e:
                logger.error(f"Error processing user message: {str(e)}")
                self.send_error_response("I apologize, but I encountered an error processing your request. Could you please try again?")
    
    async def send_welcome_message(self):
        """Send welcome message to the user"""
        try:
            self.session.conversation.item.create(
                llm.ChatMessage(
                    role="assistant",
                    content=WELCOME_MESSAGE
                )
            )
            self.session.response.create()
            logger.info("Welcome message sent")
        except Exception as e:
            logger.error(f"Failed to send welcome message: {e}")
    
    def send_error_response(self, message: str):
        """Send an error response to the user."""
        if not self.is_active or not self.session:
            return
            
        try:
            self.session.conversation.item.create(
                llm.ChatMessage(
                    role="system",
                    content=message
                )
            )
            self.session.response.create()
        except Exception as e:
            logger.error(f"Failed to send error response: {e}")
    
    def handle_lookup_request(self, msg: llm.ChatMessage):
        """Handle request ID lookup."""
        if not self.is_active:
            return
            
        logger.info("Handling lookup request")
        
        try:
            # Extract request ID if present in the message
            request_id_match = re.search(r'\b\d{6}\b', msg.content)
            if request_id_match:
                request_id = request_id_match.group(0)
                logger.info(f"Looking up request ID: {request_id}")
                
                try:
                    result = self.assistant_fnc.lookup_moving_request(request_id)
                    self.session.conversation.item.create(
                        llm.ChatMessage(
                            role="system",
                            content=f"Looking up request ID: {request_id}\n{result}"
                        )
                    )
                except Exception as e:
                    logger.error(f"Error looking up request: {str(e)}")
                    self.session.conversation.item.create(
                        llm.ChatMessage(
                            role="system",
                            content="I encountered an error looking up your request. Please verify your request ID and try again."
                        )
                    )
            else:
                self.session.conversation.item.create(
                    llm.ChatMessage(
                        role="system",
                        content="I'll need your request ID to look up your details. Could you please provide your 6-digit request ID?"
                    )
                )
            
            self.session.response.create()
            
        except Exception as e:
            logger.error(f"Error in handle_lookup_request: {e}")
            self.send_error_response("I encountered an error processing your lookup request. Please try again.")
    
    def collect_moving_info(self, msg: llm.ChatMessage):
        """Collect moving information from user."""
        if not self.is_active:
            return
            
        logger.info("Collecting moving information")
        
        try:
            self.session.conversation.item.create(
                llm.ChatMessage(
                    role="system",
                    content=LOOKUP_MOVING_INFO(msg)
                )
            )
            self.session.response.create()
        except Exception as e:
            logger.error(f"Error collecting moving info: {str(e)}")
            self.send_error_response("I apologize, but I encountered an error while processing your information. Could you please repeat that?")
        
    def handle_query(self, msg: llm.ChatMessage):
        """Handle general queries when we have a complete moving request."""
        if not self.is_active:
            return
            
        logger.info("Handling general query")
        
        try:
            self.session.conversation.item.create(
                llm.ChatMessage(
                    role="user",
                    content=msg.content
                )
            )
            self.session.response.create()
        except Exception as e:
            logger.error(f"Error handling query: {str(e)}")
            self.send_error_response("I apologize, but I encountered an error processing your query. Could you please try again?")
    
    async def cleanup(self):
        """Clean up resources when call ends"""
        if self.cleanup_done:
            return
            
        logger.info("Cleaning up call session...")
        
        try:
            self.is_active = False
            
            if self.assistant:
                try:
                    # Stop the assistant gracefully
                    await asyncio.wait_for(self.assistant.aclose(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Assistant cleanup timed out")
                except Exception as e:
                    logger.error(f"Error stopping assistant: {e}")
                finally:
                    self.assistant = None
            
            # Clear references
            self.model = None
            self.session = None
            self.assistant_fnc = None
            
            self.cleanup_done = True
            logger.info("Call session cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global variables for managing the agent
agent_task = None
agent_running = False
active_sessions = {}

async def entrypoint(ctx: JobContext):
    """Main entry point for the LiveKit agent - handles multiple calls"""
    global agent_running, active_sessions
    logger.info("Starting LiveKit agent...")
    
    try:
        # Connect to the room
        await ctx.connect(auto_subscribe=AutoSubscribe.SUBSCRIBE_ALL)
        logger.info("Connected to room, waiting for participants...")
        agent_running = True
        
        @ctx.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            """Handle new participant connection"""
            logger.info(f"Participant connected: {participant.identity}")
            
            # Create a new session for this participant
            session = CallSession(ctx)
            active_sessions[participant.identity] = session
            
            # Initialize the session asynchronously
            asyncio.create_task(initialize_session(session, participant))
        
        @ctx.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            """Handle participant disconnection"""
            logger.info(f"Participant disconnected: {participant.identity}")
            
            # Clean up the session for this participant
            if participant.identity in active_sessions:
                session = active_sessions[participant.identity]
                asyncio.create_task(session.cleanup())
                del active_sessions[participant.identity]
        
        # Keep the agent running
        while agent_running:
            try:
                await asyncio.sleep(1)
                
                # Clean up any disconnected sessions
                disconnected_sessions = []
                for identity, session in active_sessions.items():
                    if not session.is_active and not session.cleanup_done:
                        disconnected_sessions.append(identity)
                
                for identity in disconnected_sessions:
                    if identity in active_sessions:
                        await active_sessions[identity].cleanup()
                        del active_sessions[identity]
                        
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    except Exception as e:
        logger.error(f"Critical error in entrypoint: {e}")
        raise
    finally:
        # Clean up all active sessions
        logger.info("Cleaning up all active sessions...")
        cleanup_tasks = []
        for session in active_sessions.values():
            cleanup_tasks.append(session.cleanup())
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        agent_running = False
        logger.info("Agent shutdown complete")

async def initialize_session(session: CallSession, participant):
    """Initialize a session for a participant"""
    try:
        await session.initialize()
        logger.info(f"Session initialized for participant: {participant.identity}")
    except Exception as e:
        logger.error(f"Failed to initialize session for {participant.identity}: {e}")
        await session.cleanup()

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

def run_agent():
    """Run the LiveKit agent in a separate thread"""
    global agent_task
    
    try:
        logger.info("Starting LiveKit agent in background thread...")
        
        # Validate environment variables
        validate_environment()
        
        # Test database connection
        if not test_database_connection():
            logger.error("Database connection failed. Please check your DATABASE_URL.")
            return
        
        logger.info("Pre-flight checks passed. Starting LiveKit agent...")
        
        # Run the LiveKit agent
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
        
    except Exception as e:
        logger.error(f"Agent failed with error: {str(e)}")

# Flask web server for health checks and status
app = Flask(__name__)

@app.route('/health')
def health_check():
    """Health check endpoint for Render.com"""
    global agent_running
    
    try:
        # Test database connection
        db_healthy = test_database_connection()
        
        status = {
            "status": "healthy" if (agent_running and db_healthy) else "unhealthy",
            "agent_running": agent_running,
            "database_connected": db_healthy,
            "active_sessions": len(active_sessions),
            "timestamp": asyncio.get_event_loop().time() if agent_running else None
        }
        
        return jsonify(status), 200 if status["status"] == "healthy" else 503
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

@app.route('/status')
def status():
    """Status endpoint"""
    global agent_running, active_sessions
    
    return jsonify({
        "agent_running": agent_running,
        "active_sessions": len(active_sessions),
        "session_ids": list(active_sessions.keys())
    })

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        "service": "LiveKit Voice Assistant",
        "status": "running" if agent_running else "stopped",
        "endpoints": ["/health", "/status"]
    })

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global agent_running
    logger.info(f"Received signal {signum}, shutting down...")
    agent_running = False
    sys.exit(0)

def main():
    """Main function to run both the web server and agent."""
    global agent_task
    
    logger.info("Initializing LiveKit agent web service...")
    
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the LiveKit agent in a separate thread
        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()
        
        # Give the agent a moment to start
        import time
        time.sleep(2)
        
        # Get port from environment or default to 10000
        port = int(os.getenv('PORT', 10000))
        
        logger.info(f"Starting Flask web server on port {port}...")
        
        # Run the Flask app
        app.run(host='0.0.0.0', port=port, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service failed with error: {str(e)}")
        raise
    finally:
        global agent_running
        agent_running = False
        logger.info("Service shutdown complete")

if __name__ == "__main__":
    main()