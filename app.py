import asyncio
import os
import time
from flask import Flask, request, jsonify, make_response
from flask_restx import Api, Resource, fields, Namespace
from modules.conversation import Conversation
from modules.nlp_processor import NLPProcessor
from modules.mental_health_response_generator import MentalHealthResponseGenerator
from modules.safety_checker import SafetyChecker
from modules.mental_health_filter import MentalHealthFilter
from modules.user_auth import User, AuthToken
from config import Config
import logging
from datetime import datetime
import torch
from asgiref.wsgi import WsgiToAsgi
from asgiref.sync import async_to_sync
from functools import wraps

# Setup NLTK first before other imports
print("Setting up NLTK data...")
try:
    import setup_nltk
    setup_nltk.setup_nltk_data()
    print("✅ NLTK setup completed")
except Exception as e:
    print(f"⚠️ NLTK setup failed: {e}")

# Enforce WindowsSelectorEventLoopPolicy before any asyncio operations
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Also log to console for development
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logging.getLogger('').addHandler(console_handler)

app = Flask(__name__)

# ============================================================================
# SWAGGER API SETUP AND DOCUMENTATION
# ============================================================================

# Initialize Flask-RESTX for Swagger documentation
api = Api(
    app,
    version='1.0.0',
    title='NeuralEase Mental Health Chatbot API',
    description='''
    A comprehensive mental health support chatbot API with user authentication and session management.
    
    ## 🧠 Features
    - **AI-powered conversations**: Intelligent mental health support using advanced NLP
    - **Secure authentication**: JWT token-based user authentication system
    - **Session management**: Persistent chat sessions with full history
    - **Safety filtering**: Content moderation and crisis detection
    - **User profiles**: Personalized experience and preference management
    - **Mental health focus**: Specialized domain filtering and responses
    
    ## 🚀 Getting Started
    1. **Register**: Create a new account using `/api/auth/register`
    2. **Login**: Authenticate and get your JWT token at `/api/auth/login`
    3. **Create Session**: Start a new chat session at `/api/sessions`
    4. **Send Messages**: Begin chatting at `/api/sessions/{session_id}/messages`
    
    ## 🔐 Authentication
    Most endpoints require authentication. Include your JWT token in the Authorization header:
    ```
    Authorization: Bearer <your_jwt_token>
    ```
    
    ## 🆘 Crisis Resources
    **If you're experiencing a mental health crisis, please reach out immediately:**
    - **US Suicide & Crisis Lifeline**: 988
    - **Crisis Text Line**: Text HOME to 741741
    - **Emergency Services**: 911
    - **International**: Visit findahelpline.com
    
    ## 📊 System Status
    - **Health Check**: `/health` - Basic system health
    - **Detailed Status**: `/status` - Component-level status
    ''',
    doc='/docs',  # Swagger UI will be available at /docs
    authorizations={
        'Bearer': {
            'type': 'apiKey',
            'in': 'header',
            'name': 'Authorization',
            'description': 'JWT token. Format: Bearer &lt;your_token&gt;'
        }
    },
    security='Bearer'
)

# Create namespaces for better organization
auth_ns = Namespace('auth', description='🔐 Authentication Operations', path='/api/auth')
user_ns = Namespace('user', description='👤 User Profile Management', path='/api/user')
sessions_ns = Namespace('sessions', description='💬 Chat Session Management', path='/api/sessions')
legacy_ns = Namespace('legacy', description='🔄 Legacy Endpoints (Backward Compatibility)', path='/')

api.add_namespace(auth_ns)
api.add_namespace(user_ns)
api.add_namespace(sessions_ns)
api.add_namespace(legacy_ns)

# ============================================================================
# API MODELS FOR REQUEST/RESPONSE DOCUMENTATION
# ============================================================================

# Authentication models
auth_register_model = api.model('RegisterRequest', {
    'username': fields.String(required=True, description='Unique username (3-50 characters)', example='john_doe'),
    'email': fields.String(required=True, description='Valid email address', example='john@example.com'),
    'password': fields.String(required=True, description='Password (minimum 8 characters)', example='securepassword123')
})

auth_login_model = api.model('LoginRequest', {
    'username': fields.String(required=True, description='Your username', example='john_doe'),
    'password': fields.String(required=True, description='Your password', example='securepassword123')
})

auth_response_model = api.model('AuthResponse', {
    'message': fields.String(description='Response message', example='Login successful'),
    'user_id': fields.String(description='Unique user identifier'),
    'username': fields.String(description='Username'),
    'email': fields.String(description='Email address'),
    'token': fields.String(description='JWT authentication token')
})

# Profile models
profile_update_model = api.model('ProfileUpdateRequest', {
    'profile': fields.Raw(description='Profile data object', example={
        'preferences': {'theme': 'dark', 'notifications': True},
        'bio': 'Mental health advocate',
        'timezone': 'UTC'
    })
})

# Session models
session_create_model = api.model('CreateSessionRequest', {
    'title': fields.String(description='Optional session title', example='Daily Check-in')
})

session_update_model = api.model('UpdateSessionRequest', {
    'title': fields.String(required=True, description='New session title', example='Anxiety Support Session')
})

session_response_model = api.model('SessionResponse', {
    'session_id': fields.String(description='Unique session identifier'),
    'title': fields.String(description='Session title'),
    'created_at': fields.String(description='Creation timestamp'),
    'last_interaction': fields.String(description='Last interaction timestamp'),
    'messages': fields.List(fields.Raw, description='List of messages in session'),
    'user_profile': fields.Raw(description='User profile data')
})

# Message models
message_send_model = api.model('SendMessageRequest', {
    'message': fields.String(required=True, description='Your message to the chatbot', example='I am feeling anxious about work today')
})

message_edit_model = api.model('EditMessageRequest', {
    'content': fields.String(required=True, description='New message content', example='I am feeling better now')
})

message_response_model = api.model('MessageResponse', {
    'response': fields.String(description='Chatbot response'),
    'message_id': fields.String(description='Message identifier'),
    'session_id': fields.String(description='Session identifier'),
    'analysis': fields.Raw(description='NLP analysis results'),
    'user_profile': fields.Raw(description='Updated user profile'),
    'source': fields.String(description='Response source (AI model used)'),
    'timestamp': fields.String(description='Response timestamp')
})

# Legacy models
chat_legacy_model = api.model('ChatRequest', {
    'message': fields.String(required=True, description='Your message', example='Hello, I need someone to talk to'),
    'session_id': fields.String(description='Optional session ID to continue conversation')
})

consent_model = api.model('ConsentRequest', {
    'session_id': fields.String(description='Session ID (optional)'),
    'consent': fields.Boolean(required=True, description='Consent to data storage', example=True)
})

feedback_model = api.model('FeedbackRequest', {
    'session_id': fields.String(required=True, description='Session ID'),
    'satisfaction': fields.Integer(required=True, min=1, max=5, description='Satisfaction rating (1-5)', example=4),
    'comments': fields.String(description='Optional feedback comments', example='The chatbot was very helpful')
})

# Initialize components
print("Initializing chatbot components...")
try:
    nlp_processor = NLPProcessor()
    response_generator = MentalHealthResponseGenerator()
    safety_checker = SafetyChecker()
    mental_health_filter = MentalHealthFilter()
    print("✅ All components initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize components: {e}")
    # Create simple fallback components
    nlp_processor = None
    response_generator = None
    safety_checker = None
    mental_health_filter = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device set to use {device}")
logging.info("Mental Health Chatbot starting up with enhanced user features")

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Check if token is in headers
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        # If no token in headers, check cookies
        if not token:
            token = request.cookies.get('token')
            
        # If no token in cookies, check JSON body
        if not token and request.is_json:
            token = request.json.get('token')
            
        if not token:
            api.abort(401, 'Authentication token is missing')
            
        # Validate token
        user_id = AuthToken.validate_token(token)
        if not user_id:
            api.abort(401, 'Invalid or expired token')
            
        # Load user
        user = User()
        if not user.load_by_user_id(user_id):
            api.abort(404, 'User not found')
            
        return f(user, *args, **kwargs)
    
    return decorated

# ============================================================================
# ROOT AND STATUS ENDPOINTS
# ============================================================================

@api.route('/')
class APIRoot(Resource):
    @api.doc('api_root')
    def get(self):
        """API Root - Welcome and navigation"""
        return {
            "name": "NeuralEase Mental Health Chatbot API",
            "version": "1.0.0",
            "status": "operational",
            "message": "Welcome to NeuralEase Mental Health Chatbot API",
            "documentation": {
                "swagger_ui": "/docs",
                "description": "Interactive API documentation with try-it-out functionality"
            },
            "endpoints": {
                "health_check": "/health",
                "detailed_status": "/status"
            },
            "quick_start": {
                "1": "Register at /api/auth/register",
                "2": "Login at /api/auth/login",
                "3": "Create session at /api/sessions",
                "4": "Start chatting at /api/sessions/{session_id}/messages"
            },
            "support": "This API provides mental health support through AI-powered conversations"
        }

@api.route('/health')
class HealthCheck(Resource):
    @api.doc('health_check')
    @api.response(200, 'System is healthy')
    def get(self):
        """Health check - Verify API is running"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": "running",
            "components": {
                "nlp_processor": "available" if nlp_processor else "unavailable",
                "response_generator": "available" if response_generator else "unavailable",
                "safety_checker": "available" if safety_checker else "unavailable",
                "mental_health_filter": "available" if mental_health_filter else "unavailable"
            }
        }

@api.route('/status')
class StatusCheck(Resource):
    @api.doc('status_check')
    @api.response(200, 'Detailed system status')
    def get(self):
        """Detailed system status and component health"""
        status = {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "environment": os.environ.get('FLASK_ENV', 'production'),
            "components": {
                "nlp_processor": "available" if nlp_processor else "unavailable",
                "response_generator": "available" if response_generator else "unavailable",
                "safety_checker": "available" if safety_checker else "unavailable",
                "mental_health_filter": "available" if mental_health_filter else "unavailable"
            }
        }
        
        if response_generator:
            # Check AI API status
            if hasattr(response_generator, 'gemini_api_key') and response_generator.gemini_api_key:
                status["components"]["gemini_api"] = "configured"
            else:
                status["components"]["gemini_api"] = "not configured"
            
            if hasattr(response_generator, 'openai_client') and response_generator.openai_client:
                status["components"]["openai_api"] = "available"
            else:
                status["components"]["openai_api"] = "unavailable"
        
        return status

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================

@auth_ns.route('/register')
class Register(Resource):
    @auth_ns.expect(auth_register_model)
    @auth_ns.marshal_with(auth_response_model, code=201)
    @auth_ns.response(400, 'Validation error')
    @auth_ns.response(409, 'Username already exists')
    @auth_ns.response(500, 'Server error')
    def post(self):
        """Register a new user account"""
        try:
            data = request.get_json()
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            
            if not username or not email or not password:
                api.abort(400, "Username, email, and password are required")
                
            # Validate password strength
            if len(password) < 8:
                api.abort(400, "Password must be at least 8 characters long")
                
            # Create new user
            user = User()
            if not user.create_user(username, email, password):
                api.abort(409, "Username already exists")
                
            # Generate authentication token
            token = AuthToken.generate_token(user.user_id)
            
            # Return token with user information
            response = make_response(jsonify({
                "message": "User registered successfully",
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "token": token
            }), 201)
            
            # Set token cookie
            response.set_cookie(
                'token', 
                token, 
                httponly=True, 
                max_age=Config.TOKEN_EXPIRY_HOURS * 3600,
                secure=Config.SECURE_COOKIES
            )
            
            return response
            
        except Exception as e:
            logging.error(f"Registration error: {str(e)}")
            api.abort(500, f"Server error: {str(e)}")

@auth_ns.route('/login')
class Login(Resource):
    @auth_ns.expect(auth_login_model)
    @auth_ns.marshal_with(auth_response_model)
    @auth_ns.response(400, 'Missing credentials')
    @auth_ns.response(401, 'Invalid credentials')
    @auth_ns.response(500, 'Server error')
    def post(self):
        """Login with username and password"""
        try:
            data = request.get_json()
            username = data.get('username')
            password = data.get('password')
            
            if not username or not password:
                api.abort(400, "Username and password are required")
                
            # Authenticate user
            user = User()
            if not user.authenticate(username, password):
                api.abort(401, "Invalid username or password")
                
            # Generate authentication token
            token = AuthToken.generate_token(user.user_id)
            
            # Return token with user information
            response = make_response(jsonify({
                "message": "Login successful",
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "token": token
            }))
            
            # Set token cookie
            response.set_cookie(
                'token', 
                token, 
                httponly=True, 
                max_age=Config.TOKEN_EXPIRY_HOURS * 3600,
                secure=Config.SECURE_COOKIES
            )
            
            return response
            
        except Exception as e:
            logging.error(f"Login error: {str(e)}")
            api.abort(500, f"Server error: {str(e)}")

@auth_ns.route('/logout')
class Logout(Resource):
    @auth_ns.response(200, 'Logout successful')
    def post(self):
        """Logout user by clearing authentication token"""
        response = make_response(jsonify({"message": "Logout successful"}))
        response.delete_cookie('token')
        return response

@auth_ns.route('/refresh')
class RefreshToken(Resource):
    @auth_ns.marshal_with(auth_response_model)
    @auth_ns.response(401, 'Invalid token')
    @auth_ns.response(404, 'User not found')
    @auth_ns.doc(security='Bearer')
    @token_required
    def post(self, user):
        """Refresh authentication token"""
        # Generate a new token
        token = AuthToken.generate_token(user.user_id)
        
        # Return new token
        response = make_response(jsonify({
            "message": "Token refreshed",
            "token": token
        }))
        
        # Set token cookie
        response.set_cookie(
            'token', 
            token, 
            httponly=True, 
            max_age=Config.TOKEN_EXPIRY_HOURS * 3600,
            secure=Config.SECURE_COOKIES
        )
        
        return response

# ============================================================================
# USER PROFILE ENDPOINTS
# ============================================================================

@user_ns.route('/profile')
class UserProfile(Resource):
    @user_ns.doc(security='Bearer')
    @user_ns.response(200, 'User profile data')
    @user_ns.response(401, 'Unauthorized')
    @token_required
    def get(self, user):
        """Get user profile information"""
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "profile": user.profile,
            "created_at": user.created_at,
            "last_login": user.last_login
        }

    @user_ns.expect(profile_update_model)
    @user_ns.doc(security='Bearer')
    @user_ns.response(200, 'Profile updated successfully')
    @user_ns.response(400, 'Invalid profile data')
    @user_ns.response(401, 'Unauthorized')
    @user_ns.response(500, 'Server error')
    @token_required
    def put(self, user):
        """Update user profile information"""
        try:
            data = request.get_json()
            profile_data = data.get('profile', {})
            
            # Update profile
            if user.update_profile(profile_data):
                return {
                    "message": "Profile updated successfully",
                    "profile": user.profile
                }
            else:
                api.abort(500, "Failed to update profile")
                
        except Exception as e:
            logging.error(f"Profile update error: {str(e)}")
            api.abort(500, f"Server error: {str(e)}")

# ============================================================================
# CHAT SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@sessions_ns.route('')
class SessionsList(Resource):
    @sessions_ns.doc(security='Bearer')
    @sessions_ns.response(200, 'List of user sessions')
    @sessions_ns.response(401, 'Unauthorized')
    @token_required
    def get(self, user):
        """Get all chat sessions for the authenticated user"""
        sessions = user.get_all_sessions()
        return {
            "sessions": sessions,
            "count": len(sessions)
        }

    @sessions_ns.expect(session_create_model)
    @sessions_ns.doc(security='Bearer')
    @sessions_ns.response(201, 'Session created successfully')
    @sessions_ns.response(401, 'Unauthorized')
    @sessions_ns.response(500, 'Server error')
    @token_required
    def post(self, user):
        """Create a new chat session"""
        try:
            data = request.get_json() or {}
            title = data.get('title', 'New Conversation')
            
            # Create a new conversation
            conversation = Conversation(Config.ENCRYPTION_KEY)
            conversation.set_user_id(user.user_id)
            conversation.set_title(title)
            conversation.set_consent(True)  # Auto-consent for logged in users
            
            # Add welcome message
            welcome_msg = f"Hello! I'm NeuralEase, a mental health support chatbot. How can I help you today?"
            conversation.add_message("system", welcome_msg, {"source": "welcome", "model": "builtin"})
            
            # Save the conversation
            conversation.save_session()
            
            # Add session to user's sessions
            user.add_session(conversation.session_id)
            
            return {
                "message": "Chat session created",
                "session_id": conversation.session_id,
                "title": conversation.title
            }, 201
            
        except Exception as e:
            logging.error(f"Session creation error: {str(e)}")
            api.abort(500, f"Server error: {str(e)}")

@sessions_ns.route('/<string:session_id>')
class Session(Resource):
    @sessions_ns.doc(security='Bearer')
    @sessions_ns.marshal_with(session_response_model)
    @sessions_ns.response(200, 'Session data')
    @sessions_ns.response(401, 'Unauthorized')
    @sessions_ns.response(404, 'Session not found')
    @sessions_ns.response(500, 'Server error')
    @token_required
    def get(self, user, session_id):
        """Get detailed information about a specific chat session"""
        try:
            # Check if session belongs to user
            if session_id not in user.sessions:
                api.abort(404, "Session not found")
                
            # Load the session
            conversation = Conversation(Config.ENCRYPTION_KEY)
            if not conversation.load_session(session_id):
                api.abort(500, "Failed to load session")
                
            # Return session data
            return {
                "session_id": conversation.session_id,
                "title": conversation.title,
                "created_at": conversation.created_at,
                "last_interaction": conversation.last_interaction,
                "messages": conversation.messages,
                "user_profile": conversation.user_profile
            }
            
        except Exception as e:
            logging.error(f"Get session error: {str(e)}")
            api.abort(500, f"Server error: {str(e)}")

    @sessions_ns.expect(session_update_model)
    @sessions_ns.doc(security='Bearer')
    @sessions_ns.response(200, 'Session updated successfully')
    @sessions_ns.response(400, 'Invalid title')
    @sessions_ns.response(401, 'Unauthorized')
    @sessions_ns.response(404, 'Session not found')
    @sessions_ns.response(500, 'Server error')
    @token_required
    def put(self, user, session_id):
        """Update chat session (rename)"""
        try:
            # Check if session belongs to user
            if session_id not in user.sessions:
                api.abort(404, "Session not found")
                
            data = request.get_json()
            title = data.get('title')
            
            if not title:
                api.abort(400, "Title is required")
                
            # Load the session
            conversation = Conversation(Config.ENCRYPTION_KEY)
            if not conversation.load_session(session_id):
                api.abort(500, "Failed to load session")
                
            # Update title
            conversation.set_title(title)
            conversation.save_session()
            
            return {
                "message": "Session updated",
                "session_id": session_id,
                "title": title
            }
            
        except Exception as e:
            logging.error(f"Update session error: {str(e)}")
            api.abort(500, f"Server error: {str(e)}")

    @sessions_ns.doc(security='Bearer')
    @sessions_ns.response(200, 'Session deleted successfully')
    @sessions_ns.response(401, 'Unauthorized')
    @sessions_ns.response(404, 'Session not found')
    @sessions_ns.response(500, 'Server error')
    @token_required
    def delete(self, user, session_id):
        """Delete a chat session"""
        try:
            # Check if session belongs to user
            if session_id not in user.sessions:
                api.abort(404, "Session not found")
                
            # Load the session
            conversation = Conversation(Config.ENCRYPTION_KEY)
            if not conversation.load_session(session_id):
                api.abort(500, "Failed to load session")
                
            # Mark as deleted
            conversation.mark_deleted()
            
            # Remove from user's sessions
            user.remove_session(session_id)
            
            return {
                "message": "Session deleted",
                "session_id": session_id
            }
            
        except Exception as e:
            logging.error(f"Delete session error: {str(e)}")
            api.abort(500, f"Server error: {str(e)}")

@sessions_ns.route('/<string:session_id>/messages')
class Messages(Resource):
    @sessions_ns.expect(message_send_model)
    @sessions_ns.marshal_with(message_response_model)
    @sessions_ns.doc(security='Bearer')
    @sessions_ns.response(200, 'Message sent successfully')
    @sessions_ns.response(400, 'Invalid message')
    @sessions_ns.response(401, 'Unauthorized')
    @sessions_ns.response(404, 'Session not found')
    @sessions_ns.response(500, 'Server error')
    @token_required
    def post(self, user, session_id):
        """Send a message to a chat session"""
        # Use async_to_sync to run the async function properly
        return async_to_sync(_async_send_message)(user, session_id)

@sessions_ns.route('/<string:session_id>/messages/<string:message_id>')
class Message(Resource):
    @sessions_ns.expect(message_edit_model)
    @sessions_ns.doc(security='Bearer')
    @sessions_ns.response(200, 'Message edited successfully')
    @sessions_ns.response(400, 'Invalid content')
    @sessions_ns.response(401, 'Unauthorized')
    @sessions_ns.response(404, 'Session or message not found')
    @sessions_ns.response(500, 'Server error')
    @token_required
    def put(self, user, session_id, message_id):
        """Edit a message in a chat session"""
        try:
            # Check if session belongs to user
            if session_id not in user.sessions:
                api.abort(404, "Session not found")
                
            data = request.get_json()
            new_content = data.get('content')
            
            if not new_content:
                api.abort(400, "No content provided")
                
            # Load the session
            conversation = Conversation(Config.ENCRYPTION_KEY)
            if not conversation.load_session(session_id):
                api.abort(500, "Failed to load session")
                
            # Edit the message
            if not conversation.edit_message(message_id, new_content):
                api.abort(404, "Message not found")
                
            # Save the session
            conversation.save_session()
            
            return {
                "message": "Message edited",
                "session_id": session_id,
                "message_id": message_id
            }
            
        except Exception as e:
            logging.error(f"Edit message error: {str(e)}")
            api.abort(500, f"Server error: {str(e)}")

    @sessions_ns.doc(security='Bearer')
    @sessions_ns.response(200, 'Message deleted successfully')
    @sessions_ns.response(401, 'Unauthorized')
    @sessions_ns.response(404, 'Session or message not found')
    @sessions_ns.response(500, 'Server error')
    @token_required
    def delete(self, user, session_id, message_id):
        """Delete a message in a chat session"""
        try:
            # Check if session belongs to user
            if session_id not in user.sessions:
                api.abort(404, "Session not found")
                
            # Load the session
            conversation = Conversation(Config.ENCRYPTION_KEY)
            if not conversation.load_session(session_id):
                api.abort(500, "Failed to load session")
                
            # Delete the message
            if not conversation.delete_message(message_id):
                api.abort(404, "Message not found")
                
            # Save the session
            conversation.save_session()
            
            return {
                "message": "Message deleted",
                "session_id": session_id,
                "message_id": message_id
            }
            
        except Exception as e:
            logging.error(f"Delete message error: {str(e)}")
            api.abort(500, f"Server error: {str(e)}")

# ============================================================================
# LEGACY ENDPOINTS (Backward Compatibility) - WITH SWAGGER DOCS
# ============================================================================

@legacy_ns.route('/consent')
class ConsentEndpoint(Resource):
    @legacy_ns.expect(consent_model)
    @legacy_ns.doc('set_consent', description='Set user consent for data storage (legacy endpoint)')
    @legacy_ns.response(200, 'Consent updated successfully')
    @legacy_ns.response(400, 'Invalid session ID')
    @legacy_ns.response(500, 'Server error')
    def post(self):
        """Set user consent for data storage"""
        try:
            data = request.get_json()
            session_id = data.get('session_id', '')
            consent = data.get('consent', False)
            
            conversation = Conversation(Config.ENCRYPTION_KEY)
            if session_id:
                if not conversation.load_session(session_id):
                    logging.warning(f"Invalid session ID: {session_id}")
                    return {"error": "Invalid session ID", "session_id": session_id}, 400
            else:
                session_id = conversation.session_id
            
            conversation.set_consent(consent)
            conversation.save_session()
            
            logging.info(f"Audit: Consent set to {consent} for session {session_id}")
            return {
                "message": "Consent updated",
                "session_id": session_id,
                "consent": consent
            }
        except Exception as e:
            logging.error(f"Consent endpoint error: {str(e)}")
            return {"error": f"Server error: {str(e)}"}, 500

@legacy_ns.route('/feedback')
class FeedbackEndpoint(Resource):
    @legacy_ns.expect(feedback_model)
    @legacy_ns.doc('submit_feedback', description='Submit feedback about a chat session')
    @legacy_ns.response(200, 'Feedback recorded successfully')
    @legacy_ns.response(400, 'Invalid feedback data')
    @legacy_ns.response(500, 'Server error')
    def post(self):
        """Submit feedback about a chat session"""
        try:
            data = request.get_json()
            session_id = data.get('session_id', '')
            satisfaction = data.get('satisfaction', None)
            comments = data.get('comments', '')

            if not session_id or satisfaction is None:
                logging.warning(f"Invalid feedback request: session_id={session_id}, satisfaction={satisfaction}")
                return {"error": "Session ID and satisfaction score required"}, 400

            if not isinstance(satisfaction, int) or satisfaction < 1 or satisfaction > 5:
                logging.warning(f"Invalid satisfaction score: {satisfaction}")
                return {"error": "Satisfaction must be an integer between 1 and 5"}, 400

            conversation = Conversation(Config.ENCRYPTION_KEY)
            if not conversation.load_session(session_id):
                logging.warning(f"Invalid session ID for feedback: {session_id}")
                return {"error": "Invalid session ID", "session_id": session_id}, 400

            logging.info(f"Audit: Feedback received for session {session_id}: satisfaction={satisfaction}, comments={comments}")
            return {
                "message": "Feedback recorded",
                "session_id": session_id,
                "satisfaction": satisfaction,
                "comments": comments
            }
        except Exception as e:
            logging.error(f"Feedback endpoint error: {str(e)}")
            return {"error": f"Server error: {str(e)}"}, 500

@legacy_ns.route('/chat')
class ChatEndpoint(Resource):
    @legacy_ns.expect(chat_legacy_model)
    @legacy_ns.doc('legacy_chat', description='Legacy chat endpoint for backward compatibility')
    @legacy_ns.response(200, 'Chat response')
    @legacy_ns.response(400, 'Invalid input')
    @legacy_ns.response(403, 'Consent required')
    @legacy_ns.response(500, 'Server error')
    def post(self):
        """Legacy chat endpoint for backward compatibility"""
        return async_to_sync(_async_chat)()

# ============================================================================
# ASYNC IMPLEMENTATION FUNCTIONS
# ============================================================================

def get_fallback_response(intent, user_input):
    """Simple fallback when AI components are unavailable"""
    fallback_responses = {
        "greeting": "Hello! I'm NeuralEase, here to support you with mental health concerns. How are you feeling today?",
        "emotional_support": "I hear that you're going through a difficult time. Your feelings are valid, and you're not alone. Would you like to talk more about what you're experiencing?",
        "crisis": "I'm concerned about your wellbeing. If you're in crisis, please call 988 for immediate support from the Suicide & Crisis Lifeline. They're available 24/7.",
        "default": "I'm here to listen and support you with mental health concerns. Could you tell me a bit more about how you're feeling or what's on your mind?"
    }
    
    # Simple keyword-based intent detection
    user_lower = user_input.lower()
    if any(word in user_lower for word in ["hello", "hi", "hey"]):
        return fallback_responses["greeting"]
    elif any(word in user_lower for word in ["suicide", "kill myself", "end my life"]):
        return fallback_responses["crisis"]
    elif any(word in user_lower for word in ["sad", "depressed", "anxious", "stressed"]):
        return fallback_responses["emotional_support"]
    else:
        return fallback_responses["default"]

async def _async_send_message(user, session_id):
    """Async implementation of send_message"""
    start_time = datetime.now()
    try:
        # Check if session belongs to user
        if session_id not in user.sessions:
            return jsonify({"error": "Session not found"}), 404
            
        data = request.get_json()
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
            
        # Load the session
        conversation = Conversation(Config.ENCRYPTION_KEY)
        if not conversation.load_session(session_id):
            return jsonify({"error": "Failed to load session"}), 500
            
        logging.debug(f"Session {session_id}: Received input: {user_input}")

        # Use fallback if components are not available
        if not all([nlp_processor, response_generator, safety_checker, mental_health_filter]):
            logging.warning("Using fallback mode - some components unavailable")
            response = get_fallback_response("default", user_input)
            
            # Add basic metadata
            analysis = {"intent": {"intent": "general"}, "sentiment": {"label": "neutral"}, "emotions": "none"}
            conversation.add_message("user", user_input, analysis)
            conversation.add_message("system", response, {"source": "fallback", "model": "builtin"})
            conversation.save_session()
            
            return jsonify({
                "response": response,
                "session_id": session_id,
                "source": "fallback",
                "timestamp": datetime.now().isoformat()
            })

        # Safety check
        if not safety_checker.is_safe(user_input):
            logging.warning(f"Session {session_id}: Unsafe input detected: {user_input}")
            response = "I'm sorry, but that input contains inappropriate content. Please rephrase."
            conversation.add_message("user", user_input, {"is_safe": False})
            conversation.add_message("system", response, None)
            conversation.save_session()
            return jsonify({
                "response": response,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_profile": conversation.get_user_profile()
            })

        # Mental health domain check
        if not mental_health_filter.is_mental_health_related(user_input):
            logging.info(f"Session {session_id}: Non-mental health query detected: {user_input}")
            response = mental_health_filter.get_redirection_message(user_input)
            conversation.add_message("user", user_input, {"is_mental_health": False})
            conversation.add_message("system", response, {"source": "filter", "model": "rule-based"})
            conversation.save_session()
            return jsonify({
                "response": response,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_profile": conversation.get_user_profile(),
                "filtered": True
            })

        # Analyze text
        analysis = await nlp_processor.analyze_text(user_input)
        
        # Extract intent and other analysis
        intent_dict = analysis.get('intent', {})
        intent = intent_dict.get('intent', 'general')
        
        sentiment_dict = analysis.get('sentiment', {})
        sentiment = sentiment_dict.get('label', 'neutral')
        
        emotions = analysis.get('emotions', 'none')
        
        # Get conversation context
        context = conversation.get_context()
        user_profile = conversation.get_user_profile()
        user_profile["last_input"] = user_input

        # Add user message
        conversation.add_message("user", user_input, analysis)

        # Generate response
        async with response_generator:
            response = await response_generator.generate_response(intent, sentiment, emotions, context, user_profile)

        # Add system response
        system_metadata = {
            "intent": analysis['intent'],
            "source": response_generator._last_source,
            "model": response_generator._last_source
        }
        conversation.add_message("system", response, system_metadata)
        conversation.save_session()

        response_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Session {session_id}: Response generated in {response_time}s")

        return jsonify({
            "response": response,
            "message_id": conversation.messages[-1]["id"],
            "session_id": session_id,
            "analysis": analysis,
            "user_profile": user_profile,
            "source": response_generator._last_source,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Send message error: {str(e)}", exc_info=True)
        # Fallback response for errors
        response = "I'm having trouble processing that right now. How are you feeling today?"
        return jsonify({
            "response": response,
            "session_id": session_id,
            "source": "error_fallback",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

async def _async_chat():
    """Async implementation of the legacy chat endpoint"""
    start_time = datetime.now()
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        session_id = data.get('session_id', '')

        if not user_input:
            logging.warning(f"Session {session_id}: Empty input received")
            return jsonify({"error": "No input provided", "session_id": session_id}), 400

        logging.debug(f"Session {session_id}: Received input: {user_input}")

        conversation = Conversation(Config.ENCRYPTION_KEY)
        if session_id:
            if not conversation.load_session(session_id):
                session_id = conversation.session_id
        else:
            session_id = conversation.session_id

        if not conversation.user_profile["consent_given"]:
            logging.warning(f"Session {session_id}: Consent required")
            return jsonify({
                "error": "Consent required",
                "session_id": session_id,
                "message": "Please provide consent to store conversation data via /consent endpoint"
            }), 403

        # Use fallback if components are not available
        if not all([nlp_processor, response_generator, safety_checker, mental_health_filter]):
            logging.warning("Using fallback mode - some components unavailable")
            response = get_fallback_response("default", user_input)
            
            # Add basic metadata
            analysis = {"intent": {"intent": "general"}, "sentiment": {"label": "neutral"}, "emotions": "none"}
            conversation.add_message("user", user_input, analysis)
            conversation.add_message("system", response, {"source": "fallback", "model": "builtin"})
            conversation.save_session()
            
            return jsonify({
                "response": response,
                "session_id": session_id,
                "source": "fallback",
                "timestamp": datetime.now().isoformat()
            })

        # Full processing with all components
        # Safety check
        if not safety_checker.is_safe(user_input):
            logging.warning(f"Session {session_id}: Unsafe input detected: {user_input}")
            response = "I'm sorry, but that input contains inappropriate content. Please rephrase."
            conversation.add_message("user", user_input, {"is_safe": False})
            conversation.add_message("system", response, None)
            conversation.save_session()
            return jsonify({
                "response": response,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_profile": conversation.get_user_profile()
            })

        # Mental health domain check
        if not mental_health_filter.is_mental_health_related(user_input):
            logging.info(f"Session {session_id}: Non-mental health query detected: {user_input}")
            response = mental_health_filter.get_redirection_message(user_input)
            conversation.add_message("user", user_input, {"is_mental_health": False})
            conversation.add_message("system", response, {"source": "filter", "model": "rule-based"})
            conversation.save_session()
            return jsonify({
                "response": response,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "user_profile": conversation.get_user_profile(),
                "filtered": True
            })

        # Analyze text
        analysis = await nlp_processor.analyze_text(user_input)
        
        # Extract analysis components
        intent_dict = analysis.get('intent', {})
        intent = intent_dict.get('intent', 'general')
        
        sentiment_dict = analysis.get('sentiment', {})
        sentiment = sentiment_dict.get('label', 'neutral')
        
        emotions = analysis.get('emotions', 'none')
        
        # Get context and user profile
        context = conversation.get_context()
        user_profile = conversation.get_user_profile()
        user_profile["last_input"] = user_input

        # Add user message
        conversation.add_message("user", user_input, analysis)

        # Generate response
        async with response_generator:
            response = await response_generator.generate_response(intent, sentiment, emotions, context, user_profile)

        # Add system response
        system_metadata = {
            "intent": analysis['intent'],
            "source": response_generator._last_source,
            "model": response_generator._last_source
        }
        conversation.add_message("system", response, system_metadata)
        conversation.save_session()

        response_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Session {session_id}: Response generated in {response_time}s")

        return jsonify({
            "response": response,
            "session_id": session_id,
            "analysis": analysis,
            "user_profile": user_profile,
            "source": response_generator._last_source,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        # Fallback response for errors
        response = "I'm having trouble processing that right now. How are you feeling today?"
        return jsonify({
            "response": response,
            "session_id": session_id or "",
            "source": "error_fallback",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# Create ASGI app for deployment
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    # Get port from environment variable (Render sets this automatically)
    port = int(os.environ.get('PORT', 5000))  # Default for local development
    
    print(f"🚀 Starting Mental Health Chatbot on port {port}")
    print(f"💚 Health check available at: http://localhost:{port}/health")
    print(f"📊 Status check available at: http://localhost:{port}/status")
    print(f"📋 Swagger Documentation available at: http://localhost:{port}/docs")
    print(f"🌐 API Root: http://localhost:{port}/")
    print(f"")
    print(f"🎯 Quick Test URLs:")
    print(f"   - API Info: http://localhost:{port}/")
    print(f"   - Interactive Docs: http://localhost:{port}/docs")
    print(f"   - Health: http://localhost:{port}/health")
    
    # Bind to 0.0.0.0 for Render (not just localhost)
    app.run(host='0.0.0.0', port=port, debug=True)  # Set debug=True for local development